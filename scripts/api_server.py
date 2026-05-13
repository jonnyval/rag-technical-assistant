# scripts/api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from src.retrieval.engine import RAGEngine
from src.logger import log

app = FastAPI(title="RegLab RAG API", version="1.0")

# Глобальная переменная для движка
rag_engine = None

# Схемы для API
class TicketRequest(BaseModel):
    ticket_id: str
    text: str
    equipment: Optional[List[str]] = None

class TicketResponse(BaseModel):
    ticket_id: str
    user_intent: str
    final_answer: str
    extracted_facts: List[str]

@app.on_event("startup")
async def startup_event():
    global rag_engine
    log.info("🚀 Запуск API: Инициализация RAG движка (загрузка моделей в память)...")
    rag_engine = RAGEngine()
    log.info("✅ Движок готов к работе.")

@app.post("/api/v1/analyze_ticket", response_model=TicketResponse)
async def analyze_ticket(request: TicketRequest):
    try:
        log.info(f"Получен тикет {request.ticket_id} для анализа.")
        
        # Передаем текст тикета в ядро
        result = rag_engine.process_query(
            query=request.text,
            equipment_filter=request.equipment
        )
        
        # Формируем красивый JSON ответ для портала
        return TicketResponse(
            ticket_id=request.ticket_id,
            user_intent=result.user_intent,
            final_answer=result.final_answer,
            extracted_facts=[f"{f.source_file}: {f.fact}" for f in result.extracted_facts]
        )
    except Exception as e:
        log.error(f"Ошибка обработки тикета {request.ticket_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Запуск сервера на 8000 порту
    uvicorn.run(app, host="0.0.0.0", port=8000)