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
from pathlib import Path
from typing import List, Dict

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

# Глобальный индекс для ротации ключей Groq
_current_key_idx = 0

# --- РЕГУЛЯРКИ ---
_FIO_PATTERN = re.compile(r'\b([А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?)\s+([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+(?:вич|вна))\b', re.UNICODE)
_ADMIN_PHRASES = re.compile(r'(?i)(С уважением[:,\s]*|С наилучшими пожеланиями[:,\s]*|Спасибо[:,\s]*|Добрый день[:,\s]*|Здравствуйте[:,\s]*|Служба технической поддержки|инженер технической поддержки|Специалист технической поддержки|ООО "Прософт-Системы"|Тел\.|E-mail|Пожалуйста, оцените качество)')
_SYSTEM_NOISE = re.compile(r'^(SimpleLogic:|Статус изменен на|Работа остановлена|Работа окончена|Задача закрыта|Задача назначена на|Резолюция:|Ждем ответа:|Оценка:|Дата оценки:|Категория обращения|Изменен состав наблюдателей|Наблюдатели:|Соисполнители:|Установлена связь с задачей|Актив:|Контрагент:|Создано$|Серийный номер оборудования:|Завершите работу и поставьте статус|В задаче добавлено решение)', re.IGNORECASE)

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def _strip_html(text: str) -> str:
    """Удаляет HTML-разметку из поля тикета и нормализует пробелы."""

    if not text or not isinstance(text, str): return ""
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def _remove_pii(text: str) -> str:
    """Маскирует ФИО, email, телефоны и служебные подписи в тексте комментария."""

    if not text: return text
    text = _FIO_PATTERN.sub("[ФИО СКРЫТО]", text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'\b\+?7[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b', '[ТЕЛЕФОН]', text)
    return _ADMIN_PHRASES.sub("", text)

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
        prefix = "👤 [КЛИЕНТ]" if role == 'Client' else "🛠️ [ИНЖЕНЕР]" if role == 'Engineer' else "💬 [СООБЩЕНИЕ]"
        comments.append(f"{prefix}: {safe_text}")
    return comments

# --- ФУНКЦИЯ LLM (Ollama & Groq с ротацией и JSON) ---
def _enrich_ticket_with_llm(title: str, description: str, comments: List[str]) -> Dict[str, str]:
    """Получает от LLM краткие симптомы и решение для fact-документа тикета."""

    global _current_key_idx
    provider = getattr(settings, "ticket_active_llm", "ollama").lower()
    
    full_text = f"ТЕМА: {title}\n"
    if description: full_text += f"ОПИСАНИЕ: {description[:1000]}\n"
    if comments: full_text += "ПЕРЕПИСКА:\n" + "\n".join(comments[:15])
        
    # Жесткий системный промпт
    sys_prompt = (
        "Проанализируй переписку технической поддержки. Выдели техническую суть проблемы и итоговое решение. "
        "Ответь СТРОГО в формате JSON с двумя ключами: 'symptoms' и 'solution'. "
        "Значениями ключей должны быть массивы ПРОСТЫХ СТРОК (не объекты). "
        "Пример: {\"symptoms\": [\"Обрыв связи по шине B1\", \"Потери пакетов в логах\"], \"solution\": [\"Поменять модули ПЛК местами\"]}"
    )
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": full_text}
    ]
    
    model_name = getattr(settings, "ticket_llm_model_name", "llama-3.1-8b-instant")
    
    try:
        if provider == "groq":
            keys = getattr(settings, 'groq_api_keys', [])
            if not keys:
                logger.error("⚠️ Список groq_api_keys пуст!")
                return {}

            endpoint = "https://api.groq.com/openai/v1/chat/completions"
            max_retries = len(keys) * 2 
            
            for attempt in range(max_retries):
                current_key = keys[_current_key_idx % len(keys)]
                headers = {"Authorization": f"Bearer {current_key}", "Content-Type": "application/json"}
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"}
                }

                response = requests.post(endpoint, headers=headers, json=payload, timeout=60)

                if response.status_code == 429:
                    logger.warning(f"⚠️ [РОТАЦИЯ] Ключ №{_current_key_idx % len(keys) + 1} исчерпал лимит. Переключаюсь...")
                    _current_key_idx += 1 
                    time.sleep(2) 
                    continue 

                if response.status_code == 401:
                    logger.error(f"❌ Ключ №{_current_key_idx % len(keys) + 1} не авторизован. Пропускаю.")
                    _current_key_idx += 1
                    continue

                response.raise_for_status()
                
                time.sleep(2) # Пауза для сохранения лимитов
                raw_json = response.json().get("choices", [{}])[0].get("message", {}).get("content", "{}")
                return json.loads(raw_json)

            logger.error("❌ Все ключи Groq исчерпали лимиты.")
            return {}

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
            'doc_level': 'card'
        }

        # ДОКУМЕНТ: CARD
        prefix = getattr(settings, "ticket_indexing_prefix", "Техническое обращение: диагностика и решение проблемы.")
        card_lines = [
            f"{prefix}",
            f"[ТИКЕТ: {ticket_id}] [{equipment_type}]",
            f"[ТИП: {category}] | [СТАТУС: {status}]",
            f"ТЕМА: {title}\n"
        ]
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
                llm_data = _enrich_ticket_with_llm(title, description, comments)
                symptoms = llm_data.get('symptoms', [])
                solution = llm_data.get('solution', [])
                
                if symptoms or solution:
                    fact_meta['llm_symptoms'] = symptoms
                    fact_meta['llm_solution'] = solution
                    
                    # ПРЕВРАЩАЕМ СПИСКИ В КРАСИВЫЙ ТЕКСТ С БУЛЛИТАМИ
                    sym_text = "\n".join([f"- {s}" if isinstance(s, str) else f"- {str(s)}" for s in symptoms]) if isinstance(symptoms, list) else str(symptoms)
                    sol_text = "\n".join([f"- {s}" if isinstance(s, str) else f"- {str(s)}" for s in solution]) if isinstance(solution, list) else str(solution)
                    
                    fact_body = f"СИМПТОМЫ ПРОБЛЕМЫ:\n{sym_text}\n\nРЕШЕНИЕ:\n{sol_text}"
                else:
                    fact_body = f"ПРОБЛЕМА:\n{description[:500]}\n\nРЕШЕНИЕ:\n{resolution}"
            else:
                fact_body = f"ПРОБЛЕМА:\n{description[:500]}\n\nРЕШЕНИЕ:\n{resolution}"

            fact_lines = [
                f"{prefix}", 
                f"[ТИКЕТ: {ticket_id}] [{equipment_type}]",
                f"ТЕМА: {title}\n",
                fact_body
            ]
            
            documents.append(Document(page_content='\n'.join(fact_lines).strip(), metadata=fact_meta))

    except Exception as e:
        logger.error(f"❌ Ошибка парсинга {file_name}: {e}")

    return documents
