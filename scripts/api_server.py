# scripts/api_server.py
# Добавь этот блок к существующему файлу — новые импорты и эндпоинты

import sys
from pathlib import Path
import asyncio

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from typing import List, Optional, Literal
import uvicorn
import time

from src.engine import RAGEngine
from src.logger import log

app = FastAPI(title="RegLab RAG API", version="1.0")
rag_engine = None
inference_semaphore = asyncio.Semaphore(2)


# ==========================================
# СУЩЕСТВУЮЩИЕ СХЕМЫ (без изменений)
# ==========================================

class TicketRequest(BaseModel):
    """Запрос портала ТП на генерацию приватной подсказки по заявке."""

    ticket_id: str
    text: str
    equipment: Optional[List[str]] = None

class SimilarTicketResponse(BaseModel):
    """Краткое описание похожего обращения, найденного в векторной базе тикетов."""

    ticket_id: str
    source_file: str
    problem_summary: str
    solution_summary: str
    relevance_reason: str

class TicketResponse(BaseModel):
    """Ответ API с приватной подсказкой ИИ и раздельными блоками docs/tickets."""

    ticket_id: str
    user_intent: str
    docs_answer: str
    related_topics: List[str]
    similar_tickets: List[SimilarTicketResponse]
    missing_context: str
    draft_private_comment: str
    confidence: str
    # Backward-compatible fields for older clients.
    final_answer: str
    extracted_facts: List[str]


# ==========================================
# НОВЫЕ СХЕМЫ — OpenAI-совместимый формат
# ==========================================

class ChatMessage(BaseModel):
    """Сообщение в OpenAI-совместимом формате chat completions."""

    role: Literal["user", "assistant", "system"]
    content: str

class ChatCompletionRequest(BaseModel):
    """OpenAI-совместимый запрос для интеграций вроде Open WebUI."""

    model: str = "reglab-rag"
    messages: List[ChatMessage]
    stream: bool = False
    # Поле для передачи фильтра оборудования через system-сообщение или metadata
    # Open WebUI передаёт equipment через system prompt если настроить


# ==========================================
# STARTUP
# ==========================================

@app.on_event("startup")
def startup_event():
    """Инициализирует RAGEngine один раз при запуске FastAPI-приложения."""

    global rag_engine
    log.info("🚀 Запуск API: Инициализация RAG движка...")
    rag_engine = RAGEngine()
    log.info("✅ Движок готов к работе.")


# ==========================================
# СУЩЕСТВУЮЩИЙ ЭНДПОИНТ (без изменений)
# ==========================================

@app.post("/api/v1/analyze_ticket", response_model=TicketResponse)
async def analyze_ticket(request: TicketRequest):
    """Генерирует приватный комментарий ИИ для новой или текущей заявки ТП."""

    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG движок не инициализирован")

    try:
        log.info(f"Получен тикет {request.ticket_id} для анализа.")
        async with inference_semaphore:
            result = await run_in_threadpool(rag_engine.process_support_ticket, request.text)

        return TicketResponse(
            ticket_id=request.ticket_id,
            user_intent=result.user_intent,
            docs_answer=result.docs_answer,
            related_topics=result.related_topics,
            similar_tickets=[
                SimilarTicketResponse(**t.dict()) for t in result.similar_tickets
            ],
            missing_context=result.missing_context,
            draft_private_comment=result.draft_private_comment,
            confidence=result.confidence,
            final_answer=result.draft_private_comment,
            extracted_facts=[f"Документация: {result.docs_answer}"],
        )
    except Exception as e:
        log.error(f"Ошибка обработки тикета {request.ticket_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# НОВЫЙ ЭНДПОИНТ — для Open WebUI
# ==========================================

@app.get("/v1/models")
async def list_models():
    """Open WebUI вызывает этот эндпоинт при подключении — возвращаем список моделей."""
    return {
        "object": "list",
        "data": [
            {
                "id": "reglab-rag",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "reglab",
                "description": "RegLab RAG — поиск по технической документации и тикетам",
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-совместимый эндпоинт.
    Open WebUI шлёт сюда сообщения в стандартном формате,
    мы прогоняем последнее user-сообщение через RAGEngine.
    """
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG движок не инициализирован")

    try:
        # Извлекаем последнее сообщение пользователя
        user_messages = [m for m in request.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="Нет user-сообщения")

        query = user_messages[-1].content

        log.info(f"[OpenAI endpoint] query='{query[:80]}...'")

        async with inference_semaphore:
            result = await run_in_threadpool(rag_engine.process_support_ticket, query)

        tickets_block = ""
        if result.similar_tickets:
            ticket_lines = "\n".join(
                f"- **{t.ticket_id}**: {t.problem_summary} Решение/действия: {t.solution_summary}"
                for t in result.similar_tickets
            )
            tickets_block = f"\n\n---\n**Похожие обращения:**\n{ticket_lines}"

        full_answer = (
            f"{result.draft_private_comment}"
            f"\n\n---\n**Информация из документации:**\n{result.docs_answer}"
            f"{tickets_block}"
            f"\n\n**Уверенность:** {result.confidence}"
        )

        # Возвращаем в OpenAI-формате
        return {
            "id": f"chatcmpl-reglab-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_answer,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                # RAGEngine не считает токены — ставим заглушку
                "prompt_tokens": len(query.split()),
                "completion_tokens": len(full_answer.split()),
                "total_tokens": len(query.split()) + len(full_answer.split()),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[OpenAI endpoint] Ошибка: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# ЗАПУСК
# ==========================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
