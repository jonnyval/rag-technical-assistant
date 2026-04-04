import sys
import os
import pickle
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import streamlit as st
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
    source_file: str = Field(description="Название файла-источника из предоставленного контекста")
    fact: str = Field(description="Конкретный технический факт или шаг, полезный для ответа")

class RAGReasoningSchema(BaseModel):
    user_intent: str = Field(description="Кратко переформулируйте, что именно хочет узнать пользователь.")
    extracted_facts: List[FactExtraction] = Field(description="Массив полезных фактов. Пусто, если ничего не найдено.")
    missing_context: str = Field(description="Чего не хватает в контексте для полного ответа.")
    final_answer: str = Field(description="ОБЯЗАТЕЛЬНОЕ ПОЛЕ. Итоговый ответ. Формат Markdown. Выбери правильный вариант из предложенных пользователем.")

# ==========================================
# 🛠 УТИЛИТЫ
# ==========================================
class KeyRotationCallbackHandler(BaseCallbackHandler):
    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        print(f"\n⚠️ [РОТАЦИЯ] Ошибка API: {type(error).__name__}. Переключаюсь на запасной ключ...")

def format_docs(docs):
    formatted = []
    for d in docs:
        meta = d.metadata
        formatted.append(f"[{meta.get('equipment_type')} | Файл: {meta.get('source_file')} | Раздел: {meta.get('breadcrumb_raw')}]\n{d.page_content}")
    return "\n\n".join(formatted)

# ==========================================
# 🚀 ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ (QDRANT + PARENT-CHILD)
# ==========================================
@st.cache_resource(show_spinner="Загрузка баз и моделей Qdrant...")
def init_rag_system():
    # 1. Загрузка моделей эмбеддингов
    dense_embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={'device': settings.device},
        encode_kwargs={'normalize_embeddings': True}
    )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    # 2. Подключение к Qdrant (Гибридный режим)
    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        collection_name=settings.collection_name,
        path=settings.db_path,
        retrieval_mode=RetrievalMode.HYBRID
    )

    # 3. Загрузка хранилища Родителей из Pickle
    store = InMemoryStore()
    parent_store_file = os.path.join(settings.parent_store_path, "parents_store.pkl")
    if os.path.exists(parent_store_file):
        with open(parent_store_file, 'rb') as f:
            store.store = pickle.load(f)
    else:
        raise FileNotFoundError(f"Файл {parent_store_file} не найден! Выполните ингест.")

    # 4. Базовый ретривер (Qdrant -> Родители)
    # Инициализируем сплиттер (он обязателен для структуры ParentDocumentRetriever, даже при поиске)
    child_splitter = MarkdownTextSplitter(
        chunk_size=settings.child_chunk_size, 
        chunk_overlap=settings.child_chunk_overlap
    )
    
    parent_retriever = ParentDocumentRetriever(
        vectorstore=qdrant,
        docstore=store,
        child_splitter=child_splitter,  # <--- Теперь передаем обязательный параметр
        search_kwargs={"k": settings.top_k_retrieval} 
    )

    # 5. Инициализация Реранкера
    rerank_model = CrossEncoder(settings.reranker_model_name, device=settings.device)

    # 6. Сборка финального умного ретривера
    retriever = RegLabQdrantRetriever(
        parent_retriever=parent_retriever,
        reranker_model=rerank_model,
        top_k_final=settings.top_k_final,
        rerank_threshold=settings.rerank_threshold,
        use_litm=settings.use_litm
    )

    # 7. Инициализация LLM
    if settings.active_llm == "gigachat":
        robust_llm = GigaChat(credentials=settings.gigachat_credentials, verify_ssl_certs=False, model="GigaChat-2-Pro", temperature=0.2)
    elif settings.active_llm == "gemini":
        robust_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=settings.google_api_key, temperature=0.2)
    else:
        keys = settings.groq_api_keys
        llms = [
            ChatOpenAI(base_url="https://api.groq.com/openai/v1", api_key=key, model="llama-3.3-70b-versatile", temperature=0.2, max_retries=1, callbacks=[KeyRotationCallbackHandler()]) for key in keys
        ]
        robust_llm = llms[0].with_fallbacks(llms[1:]) if len(llms) > 1 else llms[0]

    # 8. Сборка SGR Промпта и Цепочки
    prompt = ChatPromptTemplate.from_template("""
    Ты ведущий технический эксперт компании "РегЛаб". 
    Твоя задача — проанализировать контекст и ответить на вопрос пользователя, строго следуя заданной JSON схеме.
    
    ВАЖНО: 
    1. Если пользователь дает варианты ответов, выбери правильный.
    2. ДЕЛАЙ ДОПУЩЕНИЯ: Если в вопросе и контексте есть похожие термины, считай, что общая логика статусов совпадает, и используй таблицу из контекста.
    3. Поле `final_answer` — это ЕДИНСТВЕННОЕ, что увидит пользователь. Ты ОБЯЗАН его заполнить, сформулировав красивый итоговый ответ на основе извлеченных фактов. Никогда не возвращай null или пустую строку в этом поле.
    
    Контекст:
    {context}
    
    Вопрос пользователя: {input}
    """)
    
    structured_llm = robust_llm.with_structured_output(RAGReasoningSchema, method="function_calling")
    
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
    st.title("🤖 База знаний РегЛаб (Qdrant Edition)")
    st.markdown(f"**База:** `{settings.active_db_name}` | **LLM:** `{settings.active_llm.upper()}`")

    try:
        retriever, sgr_chain = init_rag_system()
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ваш вопрос..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Анализ документов через Qdrant..."):
                try:
                    # 1. Получаем документы
                    docs = retriever.invoke(prompt)
                    
                    # === ДОБАВЬТЕ ЭТИ ТРИ СТРОЧКИ ДЛЯ ОТЛАДКИ ===
                    print("\n\n" + "="*50)
                    print("ЧТО ВИДИТ LLM В САМОМ ЛУЧШЕМ ДОКУМЕНТЕ:")
                    print(docs[0].page_content if docs else "Документов нет")
                    print("="*50 + "\n\n")
                    # ============================================

                    # 2. Запускаем "мозг" (SGR цепочка)
                    sgr_response = sgr_chain.invoke(prompt)
                    
                    # === ОТЛАДКА СЫРОГО ОТВЕТА ===
                    print("\n--- СЫРОЙ ОТВЕТ МОДЕЛИ ---")
                    print(repr(sgr_response))
                    print("--------------------------\n")

                    # 🔴 ВОТ ЭТА СТРОЧКА ВЫВЕДЕТ ОТВЕТ ПОЛЬЗОВАТЕЛЮ НА ЭКРАН 🔴
                    st.markdown(sgr_response.final_answer)
                    
                    with st.expander("🧠 Процесс мышления (SGR Audit)"):
                        st.markdown(f"**Понятый интент:** {sgr_response.user_intent}")
                        st.markdown("**Извлеченные факты:**")
                        if sgr_response.extracted_facts:
                            for fact in sgr_response.extracted_facts:
                                st.markdown(f"- 📄 `{fact.source_file}`: {fact.fact}")
                        else:
                            st.markdown("- Фактов не найдено.")
                        st.markdown(f"**Чего не хватило:** {sgr_response.missing_context}")

                    with st.expander("📚 Исходные чанки (Retriever)"):
                        for i, doc in enumerate(docs):
                            meta = doc.metadata
                            st.markdown(f"**{i+1}. {meta.get('source_file')}** ({meta.get('breadcrumb_raw')})")
                            if meta.get('source_url'):
                                st.markdown(f"🔗 [Ссылка]({meta.get('source_url')})")
                            st.markdown(f"Уверенность реранкера: {meta.get('rerank_score', 0):.2f}")
                            st.divider()
                            
                    st.session_state.messages.append({"role": "assistant", "content": sgr_response.final_answer})
                    
                except Exception as e:
                    st.error(f"Ошибка генерации: {e}")

if __name__ == "__main__":
    main()