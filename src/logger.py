"""Логгер RAG системы.

Два канала:
  - Основной логгер "RegLabRAG" — INFO/DEBUG для всех компонентов.
  - Аудит-логгер "RegLabRAG.audit" — детальная запись каждого вопроса в отдельный файл.

Использование аудита в engine.py:
    from src.logger import log, log_query_audit

    log_query_audit(
        query="Как настроить R500?",
        equipment_filter=["R500"],
        retrieved_docs=docs,
        response=response,
        elapsed_sec=1.23,
    )
"""

import logging
import os
import json
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional, Any

# Определяем путь к папке логов
LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE        = LOG_DIR / "rag_system.log"
AUDIT_LOG_FILE  = LOG_DIR / "query_audit.jsonl"   # Один JSON на строку — легко парсить


# ─────────────────────────────────────────────
# ОСНОВНОЙ ЛОГГЕР
# ─────────────────────────────────────────────
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("RegLabRAG")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Файл: всё подряд
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Консоль: только INFO+
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# ─────────────────────────────────────────────
# АУДИТ-ЛОГГЕР (отдельный файл, JSONL-формат)
# ─────────────────────────────────────────────
def setup_audit_logger() -> logging.Logger:
    """Настраивает отдельный логгер для полного аудита запросов.
    
    Каждая запись — валидный JSON в одну строку.
    Ротация: 3 файла по 10 МБ.
    """
    audit_logger = logging.getLogger("RegLabRAG.audit")
    audit_logger.setLevel(logging.DEBUG)
    audit_logger.propagate = False  # Не дублировать в основной логгер

    handler = RotatingFileHandler(
        AUDIT_LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler.setLevel(logging.DEBUG)
    # Формат: просто сообщение (JSON уже готов внутри log_query_audit)
    handler.setFormatter(logging.Formatter("%(message)s"))

    if audit_logger.hasHandlers():
        audit_logger.handlers.clear()
    audit_logger.addHandler(handler)
    return audit_logger


# ─────────────────────────────────────────────
# ПУБЛИЧНЫЕ ОБЪЕКТЫ
# ─────────────────────────────────────────────
log         = setup_logger()
audit_log   = setup_audit_logger()


def log_query_audit(
    query: str,
    equipment_filter: Optional[List[str]],
    retrieved_docs: Optional[List[Any]],
    response: Optional[Any],
    elapsed_sec: float = 0.0,
    extra: Optional[dict] = None,
) -> None:
    """Записывает полный аудит одного запроса в query_audit.jsonl.

    Вызывать из RAGEngine.process_query() после получения ответа.

    Args:
        query:            Исходный вопрос пользователя.
        equipment_filter: Список фильтров оборудования (или None).
        retrieved_docs:   Список документов из ретривера (LangChain Document).
        response:         Объект RAGReasoningSchema (или None при ошибке).
        elapsed_sec:      Время обработки в секундах.
        extra:            Любые дополнительные поля (например, {"ticket_id": "TS-123"}).

    Пример записи в файле:
        {
          "ts": "2026-05-05T14:22:01",
          "query": "Как настроить Modbus TCP на R500?",
          "equipment_filter": ["R500"],
          "elapsed_sec": 1.84,
          "retrieved_docs": [
            {"source": "R500_modbus.docx", "score": 0.92, "preview": "..."}
          ],
          "user_intent": "Настройка Modbus TCP",
          "final_answer": "...",
          "extracted_facts": [{"source_file": "...", "fact": "..."}],
          "missing_context": "Всего хватает"
        }
    """
    from datetime import datetime

    # ── Документы ──────────────────────────────────────
    docs_summary = []
    if retrieved_docs:
        for d in retrieved_docs:
            meta = getattr(d, "metadata", {})
            docs_summary.append({
                "source":    meta.get("source_file", "unknown"),
                "equipment": meta.get("equipment_type", ""),
                "score":     round(float(meta.get("rerank_score", 0.0)), 4),
                "preview":   (getattr(d, "page_content", "") or "")[:200],
            })

    # ── Ответ ───────────────────────────────────────────
    answer_data: dict = {}
    if response is not None:
        answer_data = {
            "user_intent":     getattr(response, "user_intent", ""),
            "final_answer":    getattr(response, "final_answer", ""),
            "missing_context": getattr(response, "missing_context", ""),
            "extracted_facts": [
                {"source_file": f.source_file, "fact": f.fact}
                for f in (getattr(response, "extracted_facts", None) or [])
            ],
            "relevant_images": getattr(response, "relevant_images", []),
        }

    record = {
        "ts":               datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "query":            query,
        "equipment_filter": equipment_filter or [],
        "elapsed_sec":      round(elapsed_sec, 3),
        "n_docs_retrieved": len(docs_summary),
        "retrieved_docs":   docs_summary,
        **answer_data,
        **(extra or {}),
    }

    try:
        audit_log.info(json.dumps(record, ensure_ascii=False))
    except Exception as e:
        log.warning(f"Не удалось записать аудит запроса: {e}")
