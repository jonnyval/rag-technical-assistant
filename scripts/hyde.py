import re
from typing import Optional
from langchain_core.language_models import BaseChatModel
from src.logger import log


# Промпт заточен под стиль документации РегЛаб:
# технические термины, конкретные значения, без воды
HYDE_PROMPT = """Ты — технический писатель компании РегЛаб.
Напиши короткий фрагмент (4-6 предложений) из технической документации, который содержал бы ответ на вопрос ниже.

Требования:
- Пиши в стиле сухой документации.
- Используй технические термины из вопроса.
- КРИТИЧЕСКИ ВАЖНО: Если ты не знаешь точного ответа, НЕ выдумывай названия переменных, параметров, функций или кодов ошибок (например, не пиши NET_STAT или AutoReconnect, если их нет в вопросе).
- Не пиши вводных фраз.

Вопрос: {query}

Фрагмент документации:"""


def generate_hypothetical_document(
    query: str,
    llm: BaseChatModel,
) -> tuple[str, bool]:
    """
    Генерирует гипотетический фрагмент документации для поиска (HyDE).

    Возвращает (текст_для_поиска, использован_ли_fallback).
    При любой ошибке или коротком ответе возвращает оригинальный запрос
    с fallback=True, чтобы система продолжила работу без сбоя.
    """
    try:
        response = llm.invoke(HYDE_PROMPT.format(query=query))
        raw = response.content if hasattr(response, "content") else str(response)

        # Убираем блоки <think>…</think> (DeepSeek-R1, некоторые локальные модели)
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        if len(cleaned) < 50:
            log.warning(
                f"HyDE: слишком короткий результат ({len(cleaned)} симв.), "
                "используем оригинальный запрос"
            )
            return query, True

        log.debug(f"HyDE сгенерировал: {cleaned[:120]}…")
        return cleaned, False

    except Exception as exc:
        log.warning(f"HyDE: ошибка генерации ({exc}), fallback на оригинальный запрос")
        return query, True