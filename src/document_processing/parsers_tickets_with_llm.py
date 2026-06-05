"""
parsers_tickets.py
==================
Парсер JSON-тикетов с портала технической поддержки РегЛаб (через API).

Что добавлено/изменено:
  - Прямое чтение JSON вместо парсинга HTML.
  - Разметка ролей (Клиент/Инженер).
  - Умная саммаризация через Groq/Ollama с возвратом строгого JSON (symptoms/solution).
  - Экспоненциальная задержка и ротация ключей для API Groq.
"""

import os
import re
import json
import time
import logging
import requests
from html import unescape
from pathlib import Path
from typing import Any, List, Dict

from langchain_core.documents import Document

try:
    from src.config import settings
except ImportError:
    class DummySettings:
        """Минимальные настройки для запуска парсера тикетов вне основного проекта."""

        enable_smart_metadata = True
        ticket_indexing_prefix = "Техническое обращение: диагностика и решение проблемы."
        ticket_active_llm = "groq"
        ticket_llm_model_name = "llama-3.1-8b-instant"
        ollama_url = "http://localhost:11434/v1"
        groq_api_keys = [os.getenv("GROQ_API_KEY")]
    settings = DummySettings()

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TAG_TAXONOMY_PATH = PROJECT_ROOT / "data" / "ticket_tag_taxonomy.json"

# Глобальный индекс для ротации ключей Groq
_current_key_idx = 0
_api_provider_idx = 0
_api_key_indices: dict[str, int] = {}
_api_exhausted_providers: set[str] = set()

# --- РЕГУЛЯРКИ ---
_FIO_PATTERN = re.compile(r'\b([А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?)\s+([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+(?:вич|вна))\b', re.UNICODE)
_ADMIN_PHRASES = re.compile(r'(?i)(С уважением[:,\s]*|С наилучшими пожеланиями[:,\s]*|Спасибо[:,\s]*|Добрый день[:,\s]*|Здравствуйте[:,\s]*|Служба технической поддержки|инженер технической поддержки|Специалист технической поддержки|ООО "Прософт-Системы"|Тел\.|E-mail|Пожалуйста, оцените качество)')
_SYSTEM_NOISE = re.compile(r'^(SimpleLogic:|Статус изменен на|Работа остановлена|Работа окончена|Задача закрыта|Задача назначена на|Резолюция:|Ждем ответа:|Оценка:|Дата оценки:|Категория обращения|Изменен состав наблюдателей|Наблюдатели:|Соисполнители:|Установлена связь с задачей|Актив:|Контрагент:|Создано$|Серийный номер оборудования:|Завершите работу и поставьте статус|В задаче добавлено решение)', re.IGNORECASE)
_WHITESPACE_RE = re.compile(r'[ \t\r\f\v]+')

TECHNICAL_FIELD_LABELS = {
    "cf_tip_oborud_reg_name": "Оборудование",
    "cf_kategoriya_or_name": "Категория",
    "cf_tip_or": "Тип обращения",
    "cf_artikul": "Артикул",
    "cf_serijnyj_nomer_": "Серийный номер",
    "cf_opisanie_otkaza": "Описание отказа",
    "cf_prichina_otkaza": "Причина отказа",
    "cf_kommentarij_k_o": "Комментарий к отказу",
    "cf_kd_po_ustraneni": "Корректирующее действие",
    "cf_kod_por": "Код причины отказа",
    "cf_status_rr": "Статус ремонта",
    "cf_data_p": "Дата производства",
    "cf_data_prodazhi": "Дата продажи",
    "cf_data_otkaza": "Дата отказа",
}
EQUIPMENT_CODE_GROUPS = {
    "r01": "regul_r050_r100_r200_r400_r500_r600",
    "r02": "regul_r500s",
    "r03": "astraregul_platform",
}

_TAG_TAXONOMY_CACHE: dict[str, Any] | None = None

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def _strip_html(text: str) -> str:
    """Удаляет HTML-разметку из поля тикета и нормализует пробелы."""

    if not text or not isinstance(text, str):
        return ""
    text = unescape(text)
    text = re.sub(r'<\s*br\s*/?\s*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</\s*(p|div|li|tr|h[1-6])\s*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = _WHITESPACE_RE.sub(' ', text)
    text = re.sub(r'\n\s+', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _remove_pii(text: str) -> str:
    """Маскирует ФИО, email, телефоны и служебные подписи в тексте комментария."""

    if not text: return text
    text = _FIO_PATTERN.sub("[ФИО СКРЫТО]", text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'\b\+?7[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b', '[ТЕЛЕФОН]', text)
    return _ADMIN_PHRASES.sub("", text)

def _format_date(value: Any) -> str:
    if not value:
        return ""
    return str(value).split("T", 1)[0]


def _clean_scalar(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, bool):
        return "да" if value else ""
    if isinstance(value, dict):
        for key in ("name", "code", "alias", "title"):
            if value.get(key):
                return _strip_html(str(value[key]))
        return ""
    if isinstance(value, list):
        items = [_clean_scalar(item) for item in value]
        return ", ".join(item for item in items if item)
    return _strip_html(str(value))


def _load_tag_taxonomy() -> dict[str, Any]:
    global _TAG_TAXONOMY_CACHE
    if _TAG_TAXONOMY_CACHE is not None:
        return _TAG_TAXONOMY_CACHE
    try:
        with TAG_TAXONOMY_PATH.open("r", encoding="utf-8") as file:
            taxonomy = json.load(file)
        if not isinstance(taxonomy, dict):
            taxonomy = {}
    except FileNotFoundError:
        logger.warning("Tag taxonomy file not found: %s", TAG_TAXONOMY_PATH)
        taxonomy = {}
    _TAG_TAXONOMY_CACHE = taxonomy
    return taxonomy


def _norm_match_text(value: str) -> str:
    value = value.lower().replace("ё", "е")
    value = re.sub(r"[_\-.]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _contains_term(text_norm: str, term: str) -> bool:
    term_norm = _norm_match_text(term)
    if not term_norm:
        return False
    pattern = r"(?<![0-9a-zа-я])" + re.escape(term_norm) + r"(?![0-9a-zа-я])"
    return re.search(pattern, text_norm, re.IGNORECASE) is not None


def _allowed_quality_tags(taxonomy: dict[str, Any] | None = None) -> list[str]:
    taxonomy = taxonomy or _load_tag_taxonomy()
    tags: list[str] = []
    for group in taxonomy.get("groups", []):
        if not isinstance(group, dict):
            continue
        for tag in group.get("tags", []):
            if isinstance(tag, str) and tag not in tags:
                tags.append(tag)
    return tags


def _normalize_quality_tags(tags: Any, taxonomy: dict[str, Any] | None = None) -> list[str]:
    taxonomy = taxonomy or _load_tag_taxonomy()
    allowed_tags = _allowed_quality_tags(taxonomy)
    alias_to_tag: dict[str, str] = {_norm_match_text(tag): tag for tag in allowed_tags}
    for tag, aliases in taxonomy.get("aliases", {}).items():
        if tag not in allowed_tags:
            continue
        for alias in aliases or []:
            alias_to_tag[_norm_match_text(str(alias))] = tag

    if not isinstance(tags, list):
        tags = [tags] if tags else []

    normalized: list[str] = []
    for raw_tag in tags:
        tag = alias_to_tag.get(_norm_match_text(str(raw_tag)))
        if tag and tag not in normalized:
            normalized.append(tag)
    return normalized


def _auto_tag_ticket(ticket_data: dict, title: str, description: str, comments: list[str], technical_lines: list[str]) -> dict[str, list[str]]:
    taxonomy = _load_tag_taxonomy()
    product_searchable = "\n".join(
        [
            title,
            description,
            "\n".join(comments[:20]),
            _clean_scalar(ticket_data.get("cf_artikul")),
        ]
    )
    tag_searchable = "\n".join([product_searchable, "\n".join(technical_lines)])
    product_text_norm = _norm_match_text(product_searchable)
    tag_text_norm = _norm_match_text(tag_searchable)

    matched_group_ids: list[str] = []
    matched_group_titles: list[str] = []
    matched_products: list[str] = []
    matched_tags: list[str] = []

    aliases = taxonomy.get("aliases", {})
    equipment_group_id = EQUIPMENT_CODE_GROUPS.get(str(ticket_data.get("cf_tip_oborud_reg") or "").lower())
    for group in taxonomy.get("groups", []):
        if not isinstance(group, dict):
            continue

        group_id = str(group.get("id") or "")
        group_title = str(group.get("title") or group_id)
        group_product_match = equipment_group_id == group_id
        for product in group.get("products", []):
            if _contains_term(product_text_norm, str(product)):
                group_product_match = True
                product_str = str(product)
                if product_str not in matched_products:
                    matched_products.append(product_str)

        group_tags = [str(tag) for tag in group.get("tags", []) if isinstance(tag, str)]
        for tag in group_tags:
            terms = [tag] + [str(alias) for alias in aliases.get(tag, [])]
            if any(_contains_term(tag_text_norm, term) for term in terms):
                if tag not in matched_tags:
                    matched_tags.append(tag)

        if group_product_match:
            if group_id and group_id not in matched_group_ids:
                matched_group_ids.append(group_id)
            if group_title and group_title not in matched_group_titles:
                matched_group_titles.append(group_title)

    return {
        "ticket_product_groups": matched_group_ids,
        "ticket_product_group_titles": matched_group_titles,
        "ticket_products": matched_products,
        "quality_tags": _normalize_quality_tags(matched_tags, taxonomy),
    }


def _extract_technical_details(ticket_data: dict) -> tuple[list[str], dict[str, str]]:
    lines: list[str] = []
    metadata: dict[str, str] = {}

    for field, label in TECHNICAL_FIELD_LABELS.items():
        value = _clean_scalar(ticket_data.get(field))
        if not value:
            continue
        lines.append(f"{label}: {value}")
        metadata[field] = value

    tags = _clean_scalar(ticket_data.get("tags"))
    if tags:
        lines.append(f"Теги: {tags}")
        metadata["tags"] = tags

    assets = ticket_data.get("assets")
    if isinstance(assets, list) and assets:
        asset_names = []
        for asset in assets[:5]:
            asset_text = _clean_scalar(asset)
            if asset_text:
                asset_names.append(asset_text)
        if asset_names:
            value = "; ".join(asset_names)
            lines.append(f"Активы/оборудование из карточки: {value}")
            metadata["assets"] = value

    return lines, metadata


def _extract_comments(ticket_data: dict) -> List[str]:
    """Достает пользовательские и инженерные комментарии из JSON тикета."""

    comments = []
    raw_comments = ticket_data.get('comments_list', [])
    if not isinstance(raw_comments, list): return comments

    for c in raw_comments:
        if not isinstance(c, dict): continue
        text = c.get('text') or c.get('content') or ''
        if isinstance(text, dict): text = text.get('text') or text.get('html') or ''
        if not text or not isinstance(text, str): continue
            
        clean_text = _strip_html(text)
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()
        
        if _SYSTEM_NOISE.match(clean_text) or len(clean_text) <= 10: continue
        safe_text = _remove_pii(clean_text)
        
        role = c.get('author_role', 'Unknown')
        role_label = "[КЛИЕНТ]" if role == 'Client' else "[ИНЖЕНЕР]" if role == 'Engineer' else "[СООБЩЕНИЕ]"
        date = _format_date(c.get("cmf_created_at"))
        date_prefix = f"{date} " if date else ""
        comments.append(f"{date_prefix}{role_label}: {safe_text}")
    return comments

# --- ФУНКЦИЯ LLM (Ollama & Groq с ротацией и JSON) ---
def _enrich_ticket_with_llm(title: str, description: str, comments: List[str], technical_details: List[str] | None = None) -> Dict[str, str]:
    """Получает от LLM краткие симптомы и решение для fact-документа тикета."""

    global _current_key_idx
    provider = getattr(settings, "ticket_active_llm", "ollama").lower()
    
    full_text = f"ТЕМА: {title}\n"
    if description: full_text += f"ОПИСАНИЕ: {description[:1000]}\n"
    if technical_details: full_text += "ТЕХНИЧЕСКИЕ ПОЛЯ:\n" + "\n".join(technical_details[:12]) + "\n"
    if comments: full_text += "ПЕРЕПИСКА:\n" + "\n".join(comments[:15])
        
    allowed_tags_text = ", ".join(_allowed_quality_tags())

    # Жесткий системный промпт
    sys_prompt = (
        "Проанализируй переписку технической поддержки. Выдели техническую суть проблемы и итоговое решение. "
        "Ответь СТРОГО в формате JSON с ключами: 'symptoms', 'solution' и 'tags'. "
        "Значениями ключей должны быть массивы ПРОСТЫХ СТРОК (не объекты). "
        "В 'tags' выбирай только теги из закрытого списка, не придумывай новые. "
        f"Закрытый список тегов: {allowed_tags_text}. "
        "Пример: {\"symptoms\": [\"Обрыв связи по шине B1\", \"Потери пакетов в логах\"], "
        "\"solution\": [\"Поменять модули ПЛК местами\"], \"tags\": [\"Redundancy\", \"Modbus_TCP master\"]}"
    )
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": full_text}
    ]
    
    model_name = getattr(settings, "ticket_llm_model_name", "llama-3.1-8b-instant")
    
    try:
        if provider == "api":
            return _enrich_ticket_with_api_rotation(messages)

        if provider == "groq":
            return _enrich_ticket_with_groq(messages)

        if provider == "gemini":
            return _enrich_ticket_with_gemini(messages)

        elif provider == "ollama":
            endpoint = "http://localhost:11434/api/chat"
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.1,
                "format": "json",
                "stream": False,
            }

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(endpoint, json=payload, timeout=120)

                    if response.status_code == 404 and "model" in response.text:
                        logger.error(
                            f"❌ Модель '{model_name}' не найдена в Ollama! "
                            f"Введите: ollama pull {model_name}"
                        )
                        return {}

                    response.raise_for_status()

                    raw_body = response.text.strip()
                    if not raw_body:
                        # Ollama вернула пустой ответ — пауза и повтор
                        wait = 5 * (attempt + 1)
                        logger.warning(
                            f"⚠️ Ollama вернула пустой ответ (попытка {attempt + 1}/{max_retries}). "
                            f"Жду {wait}с..."
                        )
                        time.sleep(wait)
                        continue

                    raw_json = response.json().get("message", {}).get("content", "").strip()
                    if not raw_json:
                        wait = 5 * (attempt + 1)
                        logger.warning(
                            f"⚠️ Ollama вернула пустой content (попытка {attempt + 1}/{max_retries}). "
                            f"Жду {wait}с..."
                        )
                        time.sleep(wait)
                        continue

                    # Иногда модель оборачивает JSON в ```json ... ``` — снимаем обёртку
                    raw_json = re.sub(r"^```(?:json)?\s*", "", raw_json)
                    raw_json = re.sub(r"\s*```$", "", raw_json).strip()

                    result = json.loads(raw_json)
                    time.sleep(1)  # Небольшая пауза между тикетами — даём модели "остыть"
                    return result

                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    wait = 10 * (attempt + 1)
                    logger.warning(
                        f"⚠️ Ollama недоступна (попытка {attempt + 1}/{max_retries}): {e}. "
                        f"Жду {wait}с..."
                    )
                    time.sleep(wait)
                    continue

                except json.JSONDecodeError as e:
                    wait = 3 * (attempt + 1)
                    logger.warning(
                        f"⚠️ Не удалось распарсить JSON от Ollama (попытка {attempt + 1}/{max_retries}): {e}. "
                        f"Жду {wait}с..."
                    )
                    time.sleep(wait)
                    continue

            logger.error(f"❌ Ollama: все {max_retries} попытки исчерпаны.")
            return {}

        else:
            logger.warning(f"⚠️ Неизвестный провайдер LLM: {provider}")
            return {}

    except Exception as e:
        logger.error(f"⚠️ Ошибка вызова LLM ({provider}) или парсинга JSON: {e}")
        return {}


def _safe_parse_llm_json(raw_json: str) -> dict:
    raw_json = (raw_json or "{}").strip()
    raw_json = re.sub(r"^```(?:json)?\s*", "", raw_json)
    raw_json = re.sub(r"\s*```$", "", raw_json).strip()
    parsed = json.loads(raw_json or "{}")
    return parsed if isinstance(parsed, dict) else {}


def _gemini_keys() -> list[str]:
    return list(getattr(settings, "google_api_keys", []) or [])


def _groq_keys() -> list[str]:
    return list(getattr(settings, "groq_api_keys", []) or [])


def _provider_keys(provider: str) -> list[str]:
    if provider == "gemini":
        return _gemini_keys()
    if provider == "groq":
        return _groq_keys()
    return []


def _next_provider_with_keys(provider_order: list[str]) -> str | None:
    global _api_provider_idx
    available = [
        provider
        for provider in provider_order
        if provider not in _api_exhausted_providers and _provider_keys(provider)
    ]
    if not available:
        return None

    for _ in range(len(provider_order)):
        provider = provider_order[_api_provider_idx % len(provider_order)]
        _api_provider_idx += 1
        if provider in available:
            return provider
    return available[0]


def _mark_provider_exhausted(provider: str) -> None:
    _api_exhausted_providers.add(provider)
    logger.warning("⚠️ Все ключи %s для обработки тикетов исчерпаны или недоступны.", provider)


def _enrich_ticket_with_api_rotation(messages: list[dict[str, str]]) -> dict:
    provider_order = list(getattr(settings, "ticket_api_provider_order", ["gemini", "groq"]) or ["gemini", "groq"])
    provider_order = [provider.lower() for provider in provider_order if provider.lower() in {"gemini", "groq"}]
    if not provider_order:
        provider_order = ["gemini", "groq"]

    total_keys = sum(len(_provider_keys(provider)) for provider in provider_order)
    if total_keys == 0:
        logger.error("⚠️ Для режима api не найдены GOOGLE_API_KEYS/GROQ_API_KEYS.")
        return {}

    attempts_left = max(total_keys * 2, 1)
    while attempts_left > 0:
        provider = _next_provider_with_keys(provider_order)
        if not provider:
            logger.error("❌ Все API-провайдеры для ticket enrichment исчерпаны.")
            return {}

        if provider == "gemini":
            result, rotate_reason = _call_gemini_with_rotation(messages)
        else:
            result, rotate_reason = _call_groq_with_rotation(messages)

        if result:
            return result

        if rotate_reason == "provider_exhausted":
            _mark_provider_exhausted(provider)

        attempts_left -= 1

    logger.error("❌ API-ротация ticket enrichment завершилась без успешного ответа.")
    return {}


def _call_gemini_with_rotation(messages: list[dict[str, str]]) -> tuple[dict, str | None]:
    keys = _gemini_keys()
    if not keys:
        return {}, "provider_exhausted"

    key_index = _api_key_indices.get("gemini", 0)
    model = getattr(settings, "ticket_provider_model_name", lambda provider: "gemini-2.5-flash")("gemini")
    content = "\n\n".join(f"{msg['role'].upper()}:\n{msg['content']}" for msg in messages)

    for _ in range(len(keys)):
        current_index = key_index % len(keys)
        key = keys[current_index]
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": content}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        try:
            response = requests.post(endpoint, params={"key": key}, json=payload, timeout=90)
            if response.status_code in (401, 403, 429):
                logger.warning(
                    "⚠️ Gemini key #%s недоступен или исчерпал лимит: HTTP %s. Переключаюсь.",
                    current_index + 1,
                    response.status_code,
                )
                key_index += 1
                _api_key_indices["gemini"] = key_index
                time.sleep(2)
                continue

            if response.status_code in (500, 502, 503, 504):
                logger.warning("⚠️ Gemini временно недоступен: HTTP %s.", response.status_code)
                key_index += 1
                _api_key_indices["gemini"] = key_index
                time.sleep(3)
                continue

            response.raise_for_status()
            data = response.json()
            raw_json = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "{}")
            )
            time.sleep(2)
            return _safe_parse_llm_json(raw_json), None

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            logger.warning("⚠️ Сетевая ошибка Gemini key #%s: %s. Переключаюсь.", current_index + 1, exc)
            key_index += 1
            _api_key_indices["gemini"] = key_index
            time.sleep(3)
        except Exception as exc:
            logger.error("⚠️ Ошибка Gemini key #%s: %s", current_index + 1, exc)
            key_index += 1
            _api_key_indices["gemini"] = key_index

    return {}, "provider_exhausted"


def _call_groq_with_rotation(messages: list[dict[str, str]]) -> tuple[dict, str | None]:
    keys = _groq_keys()
    if not keys:
        return {}, "provider_exhausted"

    key_index = _api_key_indices.get("groq", 0)
    model = getattr(settings, "ticket_provider_model_name", lambda provider: "llama-3.1-8b-instant")("groq")
    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    for _ in range(len(keys)):
        current_index = key_index % len(keys)
        key = keys[current_index]
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=90)
            if response.status_code in (401, 403, 429):
                logger.warning(
                    "⚠️ Groq key #%s недоступен или исчерпал лимит: HTTP %s. Переключаюсь.",
                    current_index + 1,
                    response.status_code,
                )
                key_index += 1
                _api_key_indices["groq"] = key_index
                time.sleep(2)
                continue

            if response.status_code in (500, 502, 503, 504):
                logger.warning("⚠️ Groq временно недоступен: HTTP %s.", response.status_code)
                key_index += 1
                _api_key_indices["groq"] = key_index
                time.sleep(3)
                continue

            response.raise_for_status()
            raw_json = response.json().get("choices", [{}])[0].get("message", {}).get("content", "{}")
            time.sleep(2)
            return _safe_parse_llm_json(raw_json), None

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            logger.warning("⚠️ Сетевая ошибка Groq key #%s: %s. Переключаюсь.", current_index + 1, exc)
            key_index += 1
            _api_key_indices["groq"] = key_index
            time.sleep(3)
        except Exception as exc:
            logger.error("⚠️ Ошибка Groq key #%s: %s", current_index + 1, exc)
            key_index += 1
            _api_key_indices["groq"] = key_index

    return {}, "provider_exhausted"


def _enrich_ticket_with_groq(messages: list[dict[str, str]]) -> dict:
    result, _ = _call_groq_with_rotation(messages)
    return result


def _enrich_ticket_with_gemini(messages: list[dict[str, str]]) -> dict:
    result, _ = _call_gemini_with_rotation(messages)
    return result


def _build_fallback_fact_body(
    description: str,
    comments: List[str],
    resolution: str,
    technical_details: List[str],
) -> str:
    problem_parts: list[str] = []
    if description:
        problem_parts.append(description[:1200])
    if technical_details:
        problem_parts.append("Технические поля:\n" + "\n".join(technical_details[:12]))

    client_comments = [comment for comment in comments if "[КЛИЕНТ]" in comment]
    engineer_comments = [comment for comment in comments if "[ИНЖЕНЕР]" in comment]

    if client_comments:
        problem_parts.append("Сообщения клиента:\n" + "\n".join(client_comments[:5]))

    solution_parts: list[str] = []
    if resolution:
        solution_parts.append(resolution)
    if engineer_comments:
        solution_parts.append("Ответы инженера:\n" + "\n".join(engineer_comments[-8:]))

    if not solution_parts and comments:
        solution_parts.append("Финальные сообщения переписки:\n" + "\n".join(comments[-5:]))

    return (
        "ПРОБЛЕМА:\n"
        + ("\n\n".join(problem_parts).strip() or "Нет явного описания проблемы.")
        + "\n\nРЕШЕНИЕ / ХОД РАЗБОРА:\n"
        + ("\n\n".join(solution_parts).strip() or "Нет явно выделенного решения.")
    )


# --- ОСНОВНАЯ ФУНКЦИЯ ПАРСИНГА ---
def process_ticket_file(file_path: Path, source_type: str = "support_tickets", portal_base_url: str = "https://support.prosyst.ru") -> List[Document]:
    """Преобразует JSON тикета в card/fact документы для индексации в Qdrant."""

    file_name = file_path.name
    ticket_id_match = re.search(r'\[([A-Z]+-\d+)\]', file_name)
    ticket_id = ticket_id_match.group(1) if ticket_id_match else "UNKNOWN"
    documents = []

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_data = json.load(f)

        ticket_data = raw_data.get('result', {})
        if isinstance(ticket_data, list) and len(ticket_data) > 0: ticket_data = ticket_data[0]

        title = _strip_html(ticket_data.get('name') or ticket_id)
        description = _strip_html(ticket_data.get('text') or ticket_data.get('description') or '')
        resolution = _strip_html(ticket_data.get('resolution') or '')
        comments = _extract_comments(ticket_data)
        technical_lines, technical_meta = _extract_technical_details(ticket_data)
        auto_tags = _auto_tag_ticket(ticket_data, title, description, comments, technical_lines)
        
        status = (ticket_data.get('cache_status_type') or 'Unknown').capitalize()
        equipment_type = ticket_data.get('cf_tip_oborud_reg_name') or "General"
        category = ticket_data.get('cf_kategoriya_or_name') or "Без категории"
        
        base_meta = {
            'source_file': file_name,
            'source_type': source_type,
            'format': 'ticket',
            'ticket_id': ticket_id,
            'ticket_url': f"{portal_base_url.rstrip('/')}/project/Task/{ticket_id}",
            'page_title': title,
            'equipment_type': equipment_type,
            'category': category,
            'status': status,
            'created_at': ticket_data.get('cmf_created_at'),
            'closed_at': ticket_data.get('status_closed_at'),
            'comments_count': len(comments),
            'doc_level': 'card',
            **technical_meta,
            **auto_tags,
        }

        # ДОКУМЕНТ: CARD
        prefix = getattr(settings, "ticket_indexing_prefix", "Техническое обращение: диагностика и решение проблемы.")
        card_lines = [
            f"{prefix}",
            f"[ТИКЕТ: {ticket_id}] [{equipment_type}]",
            f"[ТИП: {category}] | [СТАТУС: {status}]",
            f"ТЕМА: {title}\n"
        ]
        if technical_lines:
            card_lines.append("ТЕХНИЧЕСКИЕ ПОЛЯ:\n" + "\n".join(technical_lines) + "\n")
        if auto_tags.get("quality_tags"):
            card_lines.append("ТЕГИ КАЧЕСТВА:\n" + ", ".join(auto_tags["quality_tags"]) + "\n")
        if description: card_lines.append(f"ОПИСАНИЕ:\n{description}\n")
        if comments: card_lines.append(f"ПЕРЕПИСКА:\n" + "\n---\n".join(comments))
        if resolution: card_lines.append(f"\nИТОГОВОЕ РЕШЕНИЕ:\n{resolution}")

        card_content = "\n".join(card_lines).strip()
        documents.append(Document(page_content=card_content, metadata=base_meta))

        # ДОКУМЕНТ: FACT
        ticket_is_resolved = status.upper() in ('ЗАКРЫТО', 'РЕШЕНО', 'CLOSED', 'SOLVED')
        
        if (description or comments) and ticket_is_resolved:
            fact_meta = {**base_meta, 'doc_level': 'fact'}
            
            if getattr(settings, "enable_smart_metadata", False):
                llm_data = _enrich_ticket_with_llm(title, description, comments, technical_lines)
                symptoms = llm_data.get('symptoms', [])
                solution = llm_data.get('solution', [])
                llm_tags = _normalize_quality_tags(llm_data.get('tags', []))
                if llm_tags:
                    merged_tags = list(dict.fromkeys([*auto_tags.get("quality_tags", []), *llm_tags]))
                    fact_meta["quality_tags"] = merged_tags
                    fact_meta["llm_quality_tags"] = llm_tags
                
                if symptoms or solution:
                    fact_meta['llm_symptoms'] = symptoms
                    fact_meta['llm_solution'] = solution
                    
                    # ПРЕВРАЩАЕМ СПИСКИ В КРАСИВЫЙ ТЕКСТ С БУЛЛИТАМИ
                    sym_text = "\n".join([f"- {s}" if isinstance(s, str) else f"- {str(s)}" for s in symptoms]) if isinstance(symptoms, list) else str(symptoms)
                    sol_text = "\n".join([f"- {s}" if isinstance(s, str) else f"- {str(s)}" for s in solution]) if isinstance(solution, list) else str(solution)
                    
                    fact_body = f"СИМПТОМЫ ПРОБЛЕМЫ:\n{sym_text}\n\nРЕШЕНИЕ:\n{sol_text}"
                else:
                    fact_body = _build_fallback_fact_body(description, comments, resolution, technical_lines)
            else:
                fact_body = _build_fallback_fact_body(description, comments, resolution, technical_lines)

            fact_lines = [
                f"{prefix}", 
                f"[ТИКЕТ: {ticket_id}] [{equipment_type}]",
                f"ТЕМА: {title}\n",
                ("ТЕГИ КАЧЕСТВА: " + ", ".join(fact_meta.get("quality_tags", [])) + "\n")
                if fact_meta.get("quality_tags") else "",
                fact_body,
            ]
            
            documents.append(Document(page_content='\n'.join(line for line in fact_lines if line).strip(), metadata=fact_meta))

    except Exception as e:
        logger.error(f"❌ Ошибка парсинга {file_name}: {e}")

    return documents
