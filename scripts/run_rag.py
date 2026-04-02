import sys
import warnings
from pathlib import Path

# Добавляем корень проекта, чтобы импорты работали
sys.path.append(str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# Импорты LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Наши собственные модули
from src.config import settings
from src.retrieval.rag import RegLabHybridRetriever
from src.retrieval.bm25_utils import load_bm25_index


def main():
    print("\n" + "="*50)
    print("🚀 ЗАПУСК ПОЛНОЦЕННОЙ RAG-СИСТЕМЫ (Вектора + BM25 + Groq)")
    print("="*50)

    # 1. Загрузка баз данных
    print(f"🔌 1. Подключение к векторной базе: {settings.active_db_name}...")
    chroma_client = chromadb.PersistentClient(path=settings.db_path)
    collection = chroma_client.get_collection(name=settings.collection_name)

    print("📦 2. Загрузка лексического индекса BM25...")
    bm25_model, bm25_corpus = load_bm25_index(settings.bm25_cache)

    # 2. Инициализация нейросетей поиска
    print("🧠 3. Загрузка моделей эмбеддингов и реранкера (BGE-m3 & CrossEncoder)...")
    embed_model = SentenceTransformer(settings.embedding_model_name, device=settings.device)
    rerank_model = CrossEncoder(settings.reranker_model_name, device=settings.device)

    # 3. Создаем наш гибридный ретривер
    print("⚙️ 4. Сборка гибридного ретривера...")
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

    # 4. Подключаем LLM (Groq)
    print("🤖 5. Подключение к облачной LLM (Groq)...")
    if not settings.groq_api_key:
        print("❌ Ошибка: Ключ GROQ_API_KEY не найден в файле .env!")
        return

    llm = ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=settings.groq_api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.1
    )

    # 5. Собираем цепочку LangChain
    prompt = ChatPromptTemplate.from_template("""
Ты ведущий технический эксперт компании "РегЛаб". 
Твоя задача — давать точные, технически грамотные ответы на основе документации.
Обязательно ссылайся на названия файлов и разделы, откуда ты взял информацию.

Контекст из базы знаний:
{context}

Вопрос пользователя: {input}

Ответ:
""")
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    print("\n✅ СИСТЕМА ГОТОВА К РАБОТЕ!\n")

    # ==========================================================
    # ТЕСТОВЫЙ ЗАПРОС
    # ==========================================================
    test_query = "Опиши процесс создания резервной копии проекта в AstraRegul. Какие шаги нужно выполнить?"
    
    print(f"👤 Вопрос: {test_query}\n")
    print("⏳ Нейросеть думает (идет гибридный поиск и генерация)...\n")
    
    response = rag_chain.invoke({"input": test_query})

    print("🤖 ОТВЕТ НЕЙРОСЕТИ:")
    print(response["answer"])
    
    print("\n📚 ИСПОЛЬЗОВАННЫЕ ДОКУМЕНТЫ (Топ-3 после RRF и Реранжирования):")
    for doc in response["context"]:
        meta = doc.metadata
        print(f" - [{meta.get('equipment_type')}] Файл: {meta.get('source_file')} | Уверенность реранкера: {meta.get('rerank_score', 0):.2f}")

if __name__ == "__main__":
    main()