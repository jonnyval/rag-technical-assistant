import sys
import os
import json
import time
import pickle
import warnings
from pathlib import Path

# Добавляем корень проекта для корректных импортов
sys.path.append(str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_classic.storage import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import MarkdownTextSplitter

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_gigachat import GigaChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field
from typing import List

from src.config import settings
from src.retrieval.rag_qdrant import RegLabQdrantRetriever
from src.logger import log

# 1. Обработчик ротации ключей (с выводом текста ошибки для отладки лимитов)
class KeyRotationCallbackHandler(BaseCallbackHandler):
    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        log.warning(f"⚠️ [РОТАЦИЯ] Ошибка API: {error}. Переключаюсь на запасной ключ...")

# ==========================================
# 🧩 СТРОГАЯ SGR СХЕМА (С обязательным final_answer)
# ==========================================
class FactExtraction(BaseModel):
    source_file: str = Field(description="Название файла-источника из предоставленного контекста")
    fact: str = Field(description="Конкретный технический факт или шаг, полезный для ответа")

class RAGReasoningSchema(BaseModel):
    user_intent: str = Field(description="Кратко переформулируйте, что именно хочет узнать пользователь.")
    extracted_facts: List[FactExtraction] = Field(description="Массив полезных фактов. Пусто, если ничего не найдено.")
    missing_context: str = Field(description="Чего не хватает в контексте для полного ответа.")
    final_answer: str = Field(description="ОБЯЗАТЕЛЬНОЕ ПОЛЕ. Итоговый ответ. Формат Markdown. Выбери правильный вариант из предложенных пользователем.")

def format_docs(docs):
    formatted = []
    for d in docs:
        meta = d.metadata
        formatted.append(f"[{meta.get('equipment_type')} | Файл: {meta.get('source_file')} | Раздел: {meta.get('breadcrumb_raw')}]\n{d.page_content}")
    return "\n\n".join(formatted)

# ==========================================
# 🚀 ИНИЦИАЛИЗАЦИЯ RAG (QDRANT + PARENT-CHILD)
# ==========================================
def init_rag_system():
    log.info("Загрузка баз и моделей Qdrant...")
    
    dense_embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={'device': settings.device},
        encode_kwargs={'normalize_embeddings': True}
    )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        collection_name=settings.collection_name,
        path=settings.db_path,
        retrieval_mode=RetrievalMode.HYBRID
    )

    store = InMemoryStore()
    parent_store_file = os.path.join(settings.parent_store_path, "parents_store.pkl")
    if os.path.exists(parent_store_file):
        with open(parent_store_file, 'rb') as f:
            store.store = pickle.load(f)
    else:
        raise FileNotFoundError(f"Файл {parent_store_file} не найден! Выполните ингест.")

    child_splitter = MarkdownTextSplitter(
        chunk_size=settings.child_chunk_size, 
        chunk_overlap=settings.child_chunk_overlap
    )
    
    parent_retriever = ParentDocumentRetriever(
        vectorstore=qdrant,
        docstore=store,
        child_splitter=child_splitter,
        search_kwargs={"k": settings.top_k_retrieval} 
    )

    rerank_model = CrossEncoder(settings.reranker_model_name, device=settings.device)

    retriever = RegLabQdrantRetriever(
        parent_retriever=parent_retriever,
        reranker_model=rerank_model,
        top_k_final=settings.top_k_final,
        rerank_threshold=settings.rerank_threshold,
        use_litm=settings.use_litm
    )

    # ====================================================
    # 🤖 ПОДКЛЮЧЕНИЕ LLM
    # ====================================================
    if settings.active_llm == "gigachat":
        log.info("🤖 Подключение к LLM (GigaChat)...")
        robust_llm = GigaChat(credentials=settings.gigachat_credentials, verify_ssl_certs=False, model="GigaChat-2-Pro", temperature=0.2)
    elif settings.active_llm == "gemini":
        log.info("🤖 Подключение к LLM (Gemini)...")
        robust_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=settings.google_api_key, temperature=0.2)
    else:
        log.info("🤖 Подключение к LLM (GROQ) с ротацией ключей...")
        keys = settings.groq_api_keys
        llms = [
            ChatOpenAI(base_url="https://api.groq.com/openai/v1", api_key=key, model="llama-3.3-70b-versatile", temperature=0.2, max_retries=1, callbacks=[KeyRotationCallbackHandler()]) for key in keys
        ]
        robust_llm = llms[0].with_fallbacks(llms[1:]) if len(llms) > 1 else llms[0]

    # ====================================================
    # 🧩 СБОРКА SGR ПАЙПЛАЙНА
    # ====================================================
    prompt = ChatPromptTemplate.from_template("""
    Ты ведущий технический эксперт компании "РегЛаб". 
    Твоя задача — проанализировать контекст и ответить на вопрос пользователя, строго следуя заданной JSON схеме.
    
    ВАЖНО: 
    1. Если пользователь дает варианты ответов, выбери правильный.
    2. ДЕЛАЙ ДОПУЩЕНИЯ: Если в вопросе и контексте есть похожие термины (например, Modbus TCP и Modbus Serial), считай, что общая логика статусов совпадает, и используй таблицу из контекста.
    3. Поле `final_answer` — это ЕДИНСТВЕННОЕ, что увидит пользователь. Ты ОБЯЗАН его заполнить, сформулировав красивый итоговый ответ на основе извлеченных фактов. Никогда не возвращай null или пустую строку в этом поле.
    
    Контекст:
    {context}
    
    Вопрос пользователя: {input}
    """)
    
    structured_llm = robust_llm.with_structured_output(RAGReasoningSchema, method="function_calling")
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
            if item.get("Ответ RAG") and str(item.get("Ответ RAG")).strip() != "":
                log.info(f"⏩ Вопрос {i+1}/{len(data)} уже отвечен. Пропускаем.")
                continue

            question = item.get("Вопрос", "")
            options = item.get("Варианты ответа", "")
            full_prompt = f"{question}\n\nВарианты ответа:\n{options}"
            
            log.info(f"▶️ Обработка вопроса {i+1}/{len(data)}...")
            
            try:
                response = sgr_chain.invoke(full_prompt)
                
                # Добавлена защита от пустого ответа модели
                final_text = response.final_answer if response.final_answer else "Модель вернула пустой ответ."
                
                item["Ответ RAG"] = final_text
                item["Правильность"] = ""
                
                item["SGR_Audit"] = {
                    "Понятый интент": response.user_intent,
                    "Извлеченные факты": [f"{f.source_file}: {f.fact}" for f in response.extracted_facts] if response.extracted_facts else [],
                    "Чего не хватило": response.missing_context
                }
                
                log.info("✅ Успешный ответ получен.")
                
            except Exception as e:
                log.error(f"❌ Ошибка LLM на вопросе {i+1}: {e}")
                item["Ответ RAG"] = f"ОШИБКА: {e}"
                item["Правильность"] = ""

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
                
            if i < len(data) - 1:
                # Если вы используете Gemini, можете смело уменьшить это время до 5-10 секунд.
                # Если используете бесплатный Groq, оставляйте 60 секунд.
                sleep_time = 30 if settings.active_llm != "gemini" else 5
                log.info(f"⏱️ Ожидание {sleep_time} секунд перед следующим вопросом...")
                time.sleep(sleep_time)

    log.info("\n🎉 ВСЕ ТЕСТЫ УСПЕШНО ЗАВЕРШЕНЫ!")

if __name__ == "__main__":
    main()