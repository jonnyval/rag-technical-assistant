import sys
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_gigachat import GigaChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnablePassthrough

from src.config import settings
from src.retrieval.rag import RegLabHybridRetriever
from src.retrieval.bm25_utils import load_bm25_index

from pydantic import BaseModel, Field
from typing import List

# ==========================================
# 🧩 SGR СХЕМЫ (Schema-Guided Reasoning)
# ==========================================

class FactExtraction(BaseModel):
    """Паттерн Cycle: Извлечение конкретных фактов из контекста"""
    source_file: str = Field(description="Название файла-источника из предоставленного контекста")
    fact: str = Field(description="Конкретный технический факт или шаг, полезный для ответа")

class RAGReasoningSchema(BaseModel):
    """Паттерн Cascade: Строгая схема мышления для RAG"""
    
    # 1. Анализ
    user_intent: str = Field(description="Кратко переформулируйте, что именно хочет узнать пользователь.")
    
    # 2. Сбор фактов (Cycle) - заставляем модель выписать факты ДО формирования ответа
    extracted_facts: List[FactExtraction] = Field(description="Массив полезных фактов, найденных в контексте. Пусто, если ничего не найдено.")
    
    # 3. Рефлексия
    missing_context: str = Field(description="Чего не хватает в контексте для полного ответа (напишите 'Контекст достаточен', если всего хватает).")
    
    # 4. Итоговый ответ
    final_answer: str = Field(description="Итоговый ответ. Опирайся ТОЛЬКО на извлеченные факты (extracted_facts). ВАЖНО: Если пользователь прислал тестовый вопрос с вариантами ответов, ты ДОЛЖЕН выбрать и написать ТОЛЬКО правильные варианты, отбросив неверные. Формат Markdown, со ссылками на источники.")

# ==========================================
# 🛠 УТИЛИТЫ И ОБРАБОТЧИКИ
# ==========================================

class KeyRotationCallbackHandler(BaseCallbackHandler):
    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        print(f"\n⚠️ [РОТАЦИЯ] Ошибка API: {type(error).__name__}. Переключаюсь на запасной ключ...")

def format_docs(docs):
    """Вспомогательная функция для сборки документов в единый текст для промпта"""
    formatted = []
    for d in docs:
        meta = d.metadata
        formatted.append(f"[{meta.get('equipment_type')} | Файл: {meta.get('source_file')} | Раздел: {meta.get('breadcrumb_raw')}]\n{d.page_content}")
    return "\n\n".join(formatted)

# ==========================================
# 🚀 ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ
# ==========================================

@st.cache_resource(show_spinner="Загрузка моделей...")
def init_rag_system():
    # 1. Инициализация БД
    chroma_client = chromadb.PersistentClient(path=settings.db_path)
    collection = chroma_client.get_collection(name=settings.collection_name)
    bm25_model, bm25_corpus = load_bm25_index(settings.bm25_cache)

    # 2. Инициализация моделей поиска
    embed_model = SentenceTransformer(settings.embedding_model_name, device=settings.device)
    rerank_model = CrossEncoder(settings.reranker_model_name, device=settings.device)

    # 3. Сборка ретривера
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

    # 4. Инициализация LLM
    if settings.active_llm == "gigachat":
        robust_llm = GigaChat(
            credentials=settings.gigachat_credentials,
            verify_ssl_certs=False,
            model="GigaChat-2-Pro",
            temperature=0.2
        )
    elif settings.active_llm == "gemini":
        robust_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.google_api_key,
            temperature=0.2
        )
    else:
        keys = settings.groq_api_keys
        if not keys:
            raise ValueError("Ключи GROQ не найдены.")

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

    # 5. Сборка SGR Промпта и Цепочки
    prompt = ChatPromptTemplate.from_template("""
    Ты ведущий технический эксперт компании "РегЛаб". 
    Твоя задача — проанализировать контекст и ответить на вопрос пользователя, строго следуя заданной JSON схеме.
    
    ВАЖНО: 
    1. Если в источнике указан URL, добавь его в конце ответа.
    2. Если пользователь дает варианты ответов, твоя задача — выбрать только правильные на основе контекста.
    
    Контекст:
    {context}
    
    Вопрос пользователя: {input}
    """)
    
    # Привязываем Pydantic-схему к LLM
    structured_llm = robust_llm.with_structured_output(RAGReasoningSchema, method="function_calling")
    
    # Создаем LCEL пайплайн SGR
    sgr_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | structured_llm
    )

    return retriever, sgr_chain

# ==========================================
# 🎨 STREAMLIT ИНТЕРФЕЙС
# ==========================================

def main():
    st.set_page_config(page_title="RegLab AI", layout="wide")
    st.title("🤖 База знаний РегЛаб")
    st.markdown(f"**База:** `{settings.active_db_name}` | **LLM:** `{settings.active_llm.upper()}`")

    try:
        retriever, sgr_chain = init_rag_system()
    except Exception as e:
        st.error(f"Ошибка: {e}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Отрисовка истории чата
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Ввод нового вопроса
    if prompt := st.chat_input("Ваш вопрос..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Анализ документов (SGR)..."):
                try:
                    # 1. Получаем документы для UI (используем ретривер напрямую)
                    docs = retriever.invoke(prompt)
                    
                    # 2. Запускаем "мозг" (SGR цепочка)
                    # На выходе получаем заполненный объект Pydantic (RAGReasoningSchema)
                    sgr_response = sgr_chain.invoke(prompt)
                    
                    # Выводим финальный ответ пользователю
                    st.markdown(sgr_response.final_answer)
                    
                    # === БЛОК АУДИТА SGR (Открываем "мозги" модели) ===
                    with st.expander("🧠 Процесс мышления (SGR Audit)"):
                        st.markdown(f"**Понятый интент:** {sgr_response.user_intent}")
                        
                        st.markdown("**Извлеченные факты:**")
                        if sgr_response.extracted_facts:
                            for fact in sgr_response.extracted_facts:
                                st.markdown(f"- 📄 `{fact.source_file}`: {fact.fact}")
                        else:
                            st.markdown("- Фактов не найдено.")
                            
                        st.markdown(f"**Чего не хватило:** {sgr_response.missing_context}")

                    # === БЛОК ИСТОЧНИКОВ ===
                    with st.expander("📚 Исходные чанки (Retriever)"):
                        for i, doc in enumerate(docs):
                            meta = doc.metadata
                            st.markdown(f"**{i+1}. {meta.get('source_file')}** ({meta.get('breadcrumb_raw')})")
                            if meta.get('source_url'):
                                st.markdown(f"🔗 [Ссылка]({meta.get('source_url')})")
                            st.markdown(f"Уверенность: {meta.get('rerank_score', 0):.2f}")
                            st.divider()
                            
                    # Сохраняем в историю ТОЛЬКО финальный ответ, чтобы не замусоривать контекст чата
                    st.session_state.messages.append({"role": "assistant", "content": sgr_response.final_answer})
                    
                except Exception as e:
                    st.error(f"Ошибка генерации или парсинга SGR: {e}")

if __name__ == "__main__":
    main()