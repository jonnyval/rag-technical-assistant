import sys
import json
import time
import warnings
from pathlib import Path

# Добавляем корень проекта для корректных импортов
sys.path.append(str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_openai import ChatOpenAI
from langchain_gigachat import GigaChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import List

from src.config import settings
from src.retrieval.rag import RegLabHybridRetriever
from src.retrieval.bm25_utils import load_bm25_index
from src.logger import log
from langchain_core.callbacks import BaseCallbackHandler

# 1. Обработчик ротации ключей
class KeyRotationCallbackHandler(BaseCallbackHandler):
    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        log.warning(f"⚠️ [РОТАЦИЯ] Ошибка API: {type(error).__name__}. Переключаюсь на запасной ключ...")

# ==========================================
# 🧩 КОПИЯ SGR СХЕМЫ (из app.py)
# ==========================================
# ==========================================
# 🧩 КОПИЯ SGR СХЕМЫ (из app.py)
# ==========================================
# ==========================================
# 🧩 КОПИЯ SGR СХЕМЫ (из app.py)
# ==========================================
class FactExtraction(BaseModel):
    """Извлечение конкретных фактов из контекста."""
    source_file: str = Field(description="Название файла-источника из предоставленного контекста")
    fact: str = Field(description="Конкретный технический факт или шаг, полезный для ответа")

class RAGReasoningSchema(BaseModel):
    """Строгая схема мышления для RAG: анализ, сбор фактов и формирование ответа."""
    user_intent: str = Field(description="Кратко переформулируйте, что именно хочет узнать пользователь.")
    extracted_facts: List[FactExtraction] = Field(description="Массив полезных фактов. Оставь пустым, если прямых ответов нет.")
    missing_context: str = Field(description="Чего не хватает в контексте.")
    final_answer: str = Field(description="Итоговый ответ. Выбери только правильные варианты из предложенных пользователем. Не угадывай, если фактов нет.")

def format_docs(docs):
    formatted = []
    for d in docs:
        meta = d.metadata
        formatted.append(f"[{meta.get('equipment_type')} | Файл: {meta.get('source_file')} | Раздел: {meta.get('breadcrumb_raw')}]\n{d.page_content}")
    return "\n\n".join(formatted)

# ==========================================
# 🚀 ИНИЦИАЛИЗАЦИЯ RAG (без Streamlit)
# ==========================================
# ==========================================
# 🚀 ИНИЦИАЛИЗАЦИЯ RAG (Универсальная)
# ==========================================
def init_rag_system():
    log.info(f"🔌 Инициализация БД: {settings.db_path}...")
    chroma_client = chromadb.PersistentClient(path=settings.db_path)
    collection = chroma_client.get_collection(name=settings.collection_name)
    bm25_model, bm25_corpus = load_bm25_index(settings.bm25_cache)

    log.info("🧠 Загрузка моделей поиска...")
    embed_model = SentenceTransformer(settings.embedding_model_name, device=settings.device)
    rerank_model = CrossEncoder(settings.reranker_model_name, device=settings.device)

    retriever = RegLabHybridRetriever(
        chroma_collection=collection,
        embedding_model=embed_model,
        reranker_model=rerank_model,
        bm25=bm25_model,
        bm25_corpus_map=bm25_corpus,
        top_k_retrieval=settings.top_k_retrieval,
        top_k_final=settings.top_k_final,
        rerank_threshold=settings.rerank_threshold,
        use_litm=settings.use_litm
    )

    # ====================================================
    # 🤖 ВЫБОР И ПОДКЛЮЧЕНИЕ LLM В ЗАВИСИМОСТИ ОТ КОНФИГА
    # ====================================================
    if settings.active_llm == "gigachat":
        log.info("🤖 Подключение к LLM (GigaChat)...")
        if not settings.gigachat_credentials:
            raise ValueError("Токен GIGACHAT_CREDENTIALS не найден в .env")

        llm = GigaChat(
            credentials=settings.gigachat_credentials,
            verify_ssl_certs=False, # Важно, если нет сертификатов Минцифры
            model="GigaChat-2",   # Для сложных SGR-схем GigaChat-Pro справляется лучше базового
            temperature=0.2
        )
        # GigaChat корректно работает со структурированным выводом без явного указания method
        structured_llm = llm.with_structured_output(RAGReasoningSchema)

    else:
        # По умолчанию работает Groq с ротацией ключей
        log.info("🤖 Подключение к LLM (GROQ) с ротацией ключей...")
        keys = settings.groq_api_keys
        if not keys:
            raise ValueError("Ключи GROQ не найдены в .env")

        log.info(f"🔑 Загружено ключей: {len(keys)}")

        llms = [
            ChatOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=key,
                model="llama-3.3-70b-versatile",
                temperature=0.2,
                max_retries=1,
                callbacks=[KeyRotationCallbackHandler()]
            ) for key in keys
        ]

        primary_llm = llms[0]
        fallback_llms = llms[1:]
        robust_llm = primary_llm.with_fallbacks(fallback_llms) if fallback_llms else primary_llm
        
        # Для Groq обязательно принудительно указывать method="function_calling"
        structured_llm = robust_llm.with_structured_output(RAGReasoningSchema, method="function_calling")

    # ====================================================
    # 🧩 СБОРКА SGR ПАЙПЛАЙНА
    # ====================================================
    prompt = ChatPromptTemplate.from_template("""
    Ты ведущий технический эксперт компании "РегЛаб". 
    Твоя задача — проанализировать контекст и ответить на вопрос пользователя, строго следуя заданной JSON схеме.
    
    ВАЖНО: Если пользователь дает варианты ответов, твоя задача — выбрать только правильные на основе контекста.
    
    Контекст:
    {context}
    
    Вопрос пользователя: {input}
    """)
    
    sgr_chain = ({"context": retriever | format_docs, "input": RunnablePassthrough()} | prompt | structured_llm)

    return sgr_chain

# ==========================================
# ⚙️ ЛОГИКА АВТОТЕСТИРОВАНИЯ
# ==========================================
def main():
    INPUT_DIR = Path("data/50_questions")
    OUTPUT_DIR = Path("result_test_hands") / settings.active_db_name
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not INPUT_DIR.exists():
        log.error(f"Папка с исходными вопросами не найдена: {INPUT_DIR}")
        return

    sgr_chain = init_rag_system()
    db_prefix = settings.active_db_name

    json_files = list(INPUT_DIR.glob("*.json"))
    log.info(f"📂 Найдено файлов для тестов: {len(json_files)}")

    for file_path in json_files:
        output_file = OUTPUT_DIR / f"{db_prefix}_{file_path.name}"
        
        if output_file.exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            log.info(f"Продолжаем работу с существующим файлом: {output_file.name}")
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            log.info(f"Начинаем новый файл: {file_path.name}")

        for i, item in enumerate(data):
            # Пропускаем, если ответ уже сгенерирован
            if item.get("Ответ RAG") and str(item.get("Ответ RAG")).strip() != "":
                log.info(f"⏩ Вопрос {i+1}/{len(data)} уже отвечен. Пропускаем.")
                continue

            question = item.get("Вопрос", "")
            options = item.get("Варианты ответа", "")
            full_prompt = f"{question}\n\nВарианты ответа:\n{options}"
            
            log.info(f"▶️ Обработка вопроса {i+1}/{len(data)}...")
            
            try:
                response = sgr_chain.invoke(full_prompt)
                
                item["Ответ RAG"] = response.final_answer
                item["Правильность"] = ""
                
                item["SGR_Audit"] = {
                    "Понятый интент": response.user_intent,
                    "Извлеченные факты": [f"{f.source_file}: {f.fact}" for f in response.extracted_facts]
                }
                
                log.info("✅ Успешный ответ получен.")
                
            except Exception as e:
                log.error(f"❌ Ошибка LLM на вопросе {i+1}: {e}")
                item["Ответ RAG"] = f"ОШИБКА: {e}"
                item["Правильность"] = ""

            # Сразу сохраняем изменения на диск
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
                
            # Ждем 60 секунд (только если это не последний вопрос в файле)
            if i < len(data) - 1:
                log.info("⏱️ Ожидание 60 секунд перед следующим вопросом...")
                time.sleep(60)

    log.info("\n🎉 ВСЕ ТЕСТЫ УСПЕШНО ЗАВЕРШЕНЫ!")

if __name__ == "__main__":
    main()