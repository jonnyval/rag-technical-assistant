import numpy as np
import nltk
from typing import List, Any, Dict
from pydantic import ConfigDict, Field

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

# Убеждаемся, что токенизатор загружен
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class RegLabHybridRetriever(BaseRetriever):
    """
    Гибридный RAG-ретривер (Вектора + BM25 + RRF + CrossEncoder Reranking),
    полностью совместимый с архитектурой LangChain.
    """
    chroma_collection: Any
    embedding_model: Any
    reranker_model: Any
    
    # BM25 компоненты (могут быть None, если база пустая или индекс еще не построен)
    bm25: Any = None
    bm25_corpus_map: List[Dict] = Field(default_factory=list)
    
    # Настройки поиска
    top_k_retrieval: int = 30
    top_k_final: int = 3
    rerank_threshold: float = 0.05
    use_litm: bool = True
    
    # Разрешаем LangChain/Pydantic принимать сложные объекты (модели, клиенты БД)
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @property
    def morph_analyzer(self):
        """Ленивая инициализация лемматизатора, чтобы не ломать Pydantic"""
        if not hasattr(self, "_morph"):
            import pymorphy3
            self._morph = pymorphy3.MorphAnalyzer()
        return self._morph

    def _preprocess_text(self, text: str) -> List[str]:
        """Лемматизация текста для BM25"""
        tokens = nltk.word_tokenize(text.lower(), language="russian")
        return [self.morph_analyzer.parse(t)[0].normal_form for t in tokens if t.isalnum()]

    def _hybrid_search(self, query: str) -> List[Dict]:
        """Этап 1: Гибридный поиск и RRF (Reciprocal Rank Fusion)"""
        docs_storage = {}
        vec_ids = []
        bm25_ids = []

        # --- 1. Векторный поиск (Chroma) ---
        query_vec = self.embedding_model.encode(query).tolist()
        vec_results = self.chroma_collection.query(
            query_embeddings=[query_vec], 
            n_results=self.top_k_retrieval, 
            include=["documents", "metadatas"]
        )
        
        if vec_results['ids'] and len(vec_results['ids'][0]) > 0:
            for i in range(len(vec_results['ids'][0])):
                doc_id = vec_results['ids'][0][i]
                if doc_id not in docs_storage:
                    docs_storage[doc_id] = {
                        'content': vec_results['documents'][0][i], 
                        'metadata': vec_results['metadatas'][0][i]
                    }
                vec_ids.append(doc_id)

        # --- 2. Лексический поиск (BM25) ---
        if self.bm25 is not None and self.bm25_corpus_map:
            tokenized_query = self._preprocess_text(query)
            doc_scores = self.bm25.get_scores(tokenized_query)
            
            # Берем топ-K индексов с ненулевым скором
            top_indices = np.argsort(doc_scores)[::-1]
            added_bm25_count = 0
            
            for idx in top_indices:
                if added_bm25_count >= self.top_k_retrieval: break
                if doc_scores[idx] <= 0: continue
                
                mapped_doc = self.bm25_corpus_map[idx]
                doc_id = mapped_doc['id']
                
                if doc_id not in docs_storage:
                    docs_storage[doc_id] = {
                        'content': mapped_doc['document'], 
                        'metadata': mapped_doc['metadata']
                    }
                bm25_ids.append(doc_id)
                added_bm25_count += 1

        # --- 3. Слияние рангов (RRF) ---
        combined_scores = {}
        # Вес 60 — классическая константа для RRF
        for rank, doc_id in enumerate(vec_ids):
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (1 / (60 + rank))
        for rank, doc_id in enumerate(bm25_ids):
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (1 / (60 + rank))
            
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        
        # Передаем на реранжирование чуть больше документов (x1.5), чем нужно в финале
        limit = int(self.top_k_final * 1.5) if self.top_k_final * 1.5 < len(sorted_ids) else len(sorted_ids)
        
        return [docs_storage[doc_id] for doc_id in sorted_ids[:limit]]

    def _reorder_documents(self, documents: List[Dict], query: str) -> List[Dict]:
        """Реранжирование документов с помощью CrossEncoder"""
        if not documents:
            return []

        # 1. Формируем пары "вопрос - текст"
        pairs = [[query, doc['content']] for doc in documents]
        
        # 2. Получаем оценки от реранкера
        scores = self.reranker_model.predict(pairs)
        
        print(f"\n🧠 [RERANKER] Оценил {len(documents)} документов.")
        print(f"🔝 Максимальный балл: {max(scores):.2f}, Минимальный балл: {min(scores):.2f}")

        # 3. ВАЖНО: Записываем оценки прямо в метаданные каждого документа!
        for i, doc in enumerate(documents):
            # Конвертируем из формата numpy в обычный float, чтобы LangChain не ругался
            doc['metadata']['rerank_score'] = float(scores[i])

        # 4. Сортируем документы по убыванию уверенности
        documents.sort(key=lambda x: x['metadata']['rerank_score'], reverse=True)

        # 5. Отсекаем мусор по порогу (если нужно) и берем топ-K
        if self.rerank_threshold is not None:
            documents = [d for d in documents if d['metadata']['rerank_score'] >= self.rerank_threshold]

        # Если используем Lost In The Middle, перемешиваем (опционально, зависит от вашей реализации)
        final_docs = documents[:self.top_k_final]
        
        return final_docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Главный метод, который вызывает LangChain"""
        
        # 1. Ищем гибридом
        hybrid_docs = self._hybrid_search(query)
        
        # 2. Реранжируем
        best_docs = self._reorder_documents(hybrid_docs, query)
        
        # 3. Упаковываем в формат LangChain
        langchain_docs = []
        for d in best_docs:
            meta = d['metadata']
            
            equipment = meta.get('equipment_type', 'General')
            source_file = meta.get('source_file', 'Неизвестный файл')
            breadcrumb = meta.get('breadcrumb_raw', meta.get('page_title', 'Без раздела'))
            source_url = meta.get('source_url', '')
            
            # Формируем строку с URL только если он не пустой
            url_text = f" | URL: {source_url}" if source_url else ""
            
            # Обогащаем сам текст чанка информацией об источнике, чтобы LLM могла на него ссылаться
            enriched_content = (
                f"--- ИСТОЧНИК [Оборудование: {equipment} | Файл: {source_file} | Раздел: {breadcrumb}{url_text}] ---\n"
                f"{d['content']}"
            )
            
            langchain_docs.append(Document(page_content=enriched_content, metadata=meta))
            
        return langchain_docs