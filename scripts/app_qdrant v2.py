import sys
import os
import warnings
import re
from pathlib import Path

# Добавляем корень проекта для корректных импортов
sys.path.append(str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import streamlit as st
from sentence_transformers import CrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.storage import SQLStore

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_gigachat import GigaChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import List

from src.config import settings
from src.retrieval.rag_qdrant import RegLabQdrantRetriever

# ==========================================
# 🧩 SGR СХЕМЫ (Schema-Guided Reasoning)
# ==========================================
class FactExtraction(BaseModel):
    """Схема для извлечения отдельного факта из текста."""
    source_file: str = Field(description="Название файла-источника из предоставленного контекста")
    fact: str = Field(description="Конкретный технический факт или шаг, полезный для ответа")

class RAGReasoningSchema(BaseModel):
    """Схема мышления и генерации финального ответа.""" # <-- Обязательно для GigaChat
    user_intent: str = Field(description="Кратко переформулируйте, что именно хочет узнать пользователь.")
    extracted_facts: List[FactExtraction] = Field(description="Массив полезных фактов. Пусто, если ничего не найдено.")
    missing_context: str = Field(description="Чего не хватает в контексте для полного ответа.")
    final_answer: str = Field(description="ОБЯЗАТЕЛЬНОЕ ПОЛЕ. Итоговый ответ. Формат Markdown.")
    
    relevant_images: List[str] = Field(
        default=[], 
        description="Массив путей к изображениям (извлекай пути из разметки ![alt](путь) в контексте). Добавляй сюда пути только если изображение напрямую относится к ответу."
    )

# ==========================================
# 🛠 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
class KeyRotationCallbackHandler(BaseCallbackHandler):
    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        st.error(f"⚠️ [РОТАЦИЯ] Ошибка API: {error}")

def detect_equipment_intent(query: str) -> List[str]:
    """Определяет тип оборудования на основе ключевых слов из запроса. Возвращает список."""
    q_lower = query.lower()
    detected = []

    if re.search(r'\b(astraregul|astra|астра|hmi|ide|historian|сервер|server|окно|проект[аеу]?)\b', q_lower):
        detected.append("AstraRegul")
    
    if re.search(r'\b(r500s|r-500s|safety|безопасност[иь])\b', q_lower):
        detected.append("R500S")
        
    if re.search(r'\b(r500|r-500|regul|регул|плк)\b', q_lower):
        detected.append("R500")

    return detected if detected else ["Все"]

def format_docs(docs):
    formatted = []
    for d in docs:
        meta = d.metadata
        formatted.append(f"[{meta.get('equipment_type')} | Файл: {meta.get('source_file')} | Раздел: {meta.get('breadcrumb_raw')}]\n{d.page_content}")
    return "\n\n".join(formatted)

# ==========================================
# 🚀 ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ
# ==========================================
@st.cache_resource
def init_rag_system():
    dense_embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={'device': settings.device},
        encode_kwargs={'normalize_embeddings': True}
    )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    # 1. Подключение к Docker Qdrant (через URL)
    client = QdrantClient(url=settings.db_url)
    qdrant = QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID
    )

    # 2. Подключение к SQLite (мгновенно, без загрузки в память)
    store = SQLStore(
        namespace="reglab_parents",
        db_url=f"sqlite:///{settings.parent_store_path}"
    )

    child_splitter = MarkdownTextSplitter(chunk_size=settings.child_chunk_size, chunk_overlap=settings.child_chunk_overlap)
    
    parent_retriever = ParentDocumentRetriever(
        vectorstore=qdrant,
        docstore=store,
        child_splitter=child_splitter,
        search_kwargs={"k": settings.top_k_retrieval}
    )

    rerank_model = CrossEncoder(settings.reranker_model_name, device=settings.device)
    
    if settings.active_llm == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=settings.google_api_key, temperature=0.2)
    elif settings.active_llm == "gigachat":
        llm = GigaChat(credentials=settings.gigachat_credentials, verify_ssl_certs=False, model="GigaChat-2", temperature=0.2)
    else:
        llms = [ChatOpenAI(base_url="https://api.groq.com/openai/v1", api_key=k, model="llama-3.3-70b-versatile", temperature=0.2, callbacks=[KeyRotationCallbackHandler()]) for k in settings.groq_api_keys]
        llm = llms[0].with_fallbacks(llms[1:]) if len(llms) > 1 else llms[0]

    retriever = RegLabQdrantRetriever(
        parent_retriever=parent_retriever,
        reranker_model=rerank_model,
        top_k_final=settings.top_k_final,
        rerank_threshold=settings.rerank_threshold,
        use_litm=settings.use_litm
    )

    prompt_template = ChatPromptTemplate.from_template("""
Ты ведущий технический эксперт компании "РегЛаб". 
Твоя задача — проанализировать контекст и ответить на вопрос, строго следуя JSON схеме.

ГЛОССАРИЙ:
- R500: Стандартные ПЛК (РСУ).
- R500S: Контроллеры безопасности (ПСБ, SIL3).
- AstraRegul: Верхний уровень (HMI, Server, Historian).

ПРАВИЛА:
1. Если есть варианты ответов — выбери ВСЕ правильные.
2. ДЕЛАЙ ДОПУЩЕНИЯ: Если контекст говорит о Modbus Serial, а вопрос о Modbus TCP, считай логику статусов одинаковой.
3. Поле final_answer ОБЯЗАТЕЛЬНО. Никогда не оставляй его пустым.
4. ВАЖНО: Если в контексте встречаются изображения (в формате ![alt](путь)), и они иллюстрируют твой ответ, обязательно сохрани их "пути" в массив relevant_images.

Контекст:
{context}

Вопрос: {input}
""")
    
    if settings.active_llm == "gigachat":
        structured_llm = llm.with_structured_output(RAGReasoningSchema)
    else:
        structured_llm = llm.with_structured_output(RAGReasoningSchema, method="function_calling")
        
    sgr_chain = ({"context": retriever | format_docs, "input": RunnablePassthrough()} | prompt_template | structured_llm)

    return retriever, sgr_chain

# ==========================================
# 🖥 ИНТЕРФЕЙС STREAMLIT
# ==========================================
def main():
    st.set_page_config(page_title="RegLab AI", layout="wide")
    st.title("🤖 База знаний РегЛаб (Qdrant Edition)")
    
    st.markdown(f"**База данных:** Docker Qdrant (`{settings.collection_name}`) | **Активная LLM:** `{settings.active_llm.upper()}`")
    st.divider()
    
    st.markdown("**Фильтр поиска:**")
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_equipment = st.multiselect(
            "Принудительный фильтр (оставьте пустым для 'Авто'):",
            options=["AstraRegul", "R500", "R500S"],
            default=[],
            help="Выберите один или несколько типов оборудования. Если оставить пустым, система определит контекст автоматически."
        )
    st.divider()

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = init_rag_system()
        
    retriever, sgr_chain = st.session_state.rag_system

    if prompt := st.chat_input("Ваш вопрос..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Анализ документов через Qdrant..."):
                try:
                    active_filter = selected_equipment
                    
                    if not active_filter:
                        detected_intent = detect_equipment_intent(prompt)
                        active_filter = detected_intent
                        
                        if "Все" not in active_filter:
                            formatted_names = ", ".join(active_filter)
                            st.caption(f"🤖 *Авто-маршрутизация: Ищем в **{formatted_names}***")
                        else:
                            st.caption("🤖 *Авто-маршрутизация: Ищем по всей базе*")

                    # Передаем массив в кастомный ретривер (если вы добавили поле equipment_filter)
                    retriever.equipment_filter = active_filter
                    
                    # 1. Поиск документов
                    docs = retriever.invoke(prompt)
                    
                    # 2. Генерация ответа
                    response = sgr_chain.invoke(prompt)
                    
                    # Вывод основного ответа
                    st.markdown(response.final_answer)

                    # Отрисовка сохраненных картинок
                    if hasattr(response, 'relevant_images') and response.relevant_images:
                        for img_path in response.relevant_images:
                            if os.path.exists(img_path):
                                st.image(img_path, caption="Иллюстрация из документации")
                            else:
                                st.warning(f"Изображение не найдено на диске: {img_path}")
                    
                    # Экспандеры с деталями
                    with st.expander("🧠 Мышление (SGR)"):
                        st.write(f"**Интент:** {response.user_intent}")
                        st.write("**Факты:**")
                        for f in response.extracted_facts:
                            st.write(f"- {f.source_file}: {f.fact}")
                        st.write(f"**Чего не хватило:** {response.missing_context}")

                    with st.expander("📚 Источники"):
                        if not docs:
                            st.write("Документы не найдены (возможно, не совпали фильтры).")
                        for i, d in enumerate(docs):
                            st.write(f"{i+1}. {d.metadata.get('source_file')} (Score: {d.metadata.get('rerank_score', 0):.2f})")
                    
                    st.session_state.messages.append({"role": "assistant", "content": response.final_answer})
                except Exception as e:
                    st.error(f"Ошибка: {e}")

if __name__ == "__main__":
    main()