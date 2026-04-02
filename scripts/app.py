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
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from src.config import settings
from src.retrieval.rag import RegLabHybridRetriever
from src.retrieval.bm25_utils import load_bm25_index
from langchain_core.callbacks import BaseCallbackHandler

class KeyRotationCallbackHandler(BaseCallbackHandler):
    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        print(f"\n⚠️ [РОТАЦИЯ] Ошибка API: {type(error).__name__}. Переключаюсь на запасной ключ...")

@st.cache_resource(show_spinner="Загрузка моделей...")
def init_rag_system():
    chroma_client = chromadb.PersistentClient(path=settings.db_path)
    collection = chroma_client.get_collection(name=settings.collection_name)
    bm25_model, bm25_corpus = load_bm25_index(settings.bm25_cache)

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

    if settings.active_llm == "gigachat":
        robust_llm = GigaChat(
            credentials=settings.gigachat_credentials,
            verify_ssl_certs=False,
            model="GigaChat-2",
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
                model="llama-3.1-8b-instant",
                temperature=0.2,
                max_retries=1,
                callbacks=[KeyRotationCallbackHandler()]
            ) for key in keys
        ]
        primary_llm = llms[0]
        fallback_llms = llms[1:]
        robust_llm = primary_llm.with_fallbacks(fallback_llms) if fallback_llms else primary_llm

    prompt = ChatPromptTemplate.from_template("""
Ты ведущий технический эксперт компании "РегЛаб". 
Твоя задача — давать точные ответы на основе документации.
Ссылайся на названия файлов и разделы.

ВАЖНО: Если в источнике указан URL, добавь его в конце ответа.

Контекст:
{context}

Вопрос: {input}

Ответ:
""")
    document_chain = create_stuff_documents_chain(robust_llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

def main():
    st.set_page_config(page_title="RegLab AI", layout="wide")
    st.title("🤖 База знаний РегЛаб")
    st.markdown(f"**База:** `{settings.active_db_name}` | **LLM:** `{settings.active_llm.upper()}`")

    try:
        rag_chain = init_rag_system()
    except Exception as e:
        st.error(f"Ошибка: {e}")
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
            with st.spinner("Генерация ответа..."):
                try:
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    st.markdown(answer)
                    with st.expander("📚 Источники"):
                        for i, doc in enumerate(response["context"]):
                            meta = doc.metadata
                            st.markdown(f"**{i+1}. {meta.get('source_file')}** ({meta.get('breadcrumb_raw')})")
                            if meta.get('source_url'):
                                st.markdown(f"🔗 [Ссылка]({meta.get('source_url')})")
                            st.markdown(f"Уверенность: {meta.get('rerank_score', 0):.2f}")
                            st.divider()
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Ошибка: {e}")

if __name__ == "__main__":
    main()