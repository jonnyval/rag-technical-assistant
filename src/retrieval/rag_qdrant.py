from typing import List, Any
from pydantic import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from src.utils.profiler import TimeProfiler

class RegLabQdrantRetriever(BaseRetriever):
    """
    Кастомный ретривер для Qdrant. 
    ОПТИМИЗИРОВАННАЯ ВЕРСИЯ: Реранкер работает только по маленьким чанкам, 
    и только для победителей извлекаются полные страницы (Parents).
    """
    parent_retriever: ParentDocumentRetriever = Field(description="Базовый ретривер Parent-Child")
    reranker_model: Any = Field(description="Модель CrossEncoder для перекрестного ранжирования")
    top_k_final: int = 3
    rerank_threshold: float = 0.05
    use_litm: bool = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        
        k = self.parent_retriever.search_kwargs.get("k", 30)
        
        # 1. Поиск Детей (Гибридный поиск Qdrant)
        with TimeProfiler(f"Qdrant Hybrid Search (Топ-{k} детей)"):
            child_docs = self.parent_retriever.vectorstore.similarity_search(query, k=k)

        if not child_docs:
            return []

        # 2. Реранжирование
        with TimeProfiler("CrossEncoder Reranking"):
            pairs = [[query, doc.page_content] for doc in child_docs]
            scores = self.reranker_model.predict(pairs)
            
            scored_children = []
            for doc, score in zip(child_docs, scores):
                if score >= self.rerank_threshold:
                    doc.metadata["rerank_score"] = float(score)
                    scored_children.append(doc)
                    
            scored_children.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
            top_children = scored_children[:self.top_k_final]

        # 3. Извлечение Родителей из RAM
        with TimeProfiler("Извлечение полных страниц (Parents)"):
            parent_ids = []
            for child in top_children:
                parent_id = child.metadata.get("doc_id")
                if parent_id and parent_id not in parent_ids:
                    parent_ids.append(parent_id)
            
            final_docs = []
            if parent_ids:
                parents = self.parent_retriever.docstore.mget(parent_ids)
                for parent, p_id in zip(parents, parent_ids):
                    if parent:
                        best_score = max([c.metadata["rerank_score"] for c in top_children if c.metadata.get("doc_id") == p_id])
                        parent.metadata["rerank_score"] = best_score
                        final_docs.append(parent)

        # 4. LITM Сортировка
        with TimeProfiler("Сортировка Lost-in-the-Middle"):
            if self.use_litm and len(final_docs) > 2:
                reordered = []
                final_docs_copy = final_docs.copy()
                while final_docs_copy:
                    reordered.append(final_docs_copy.pop(0))
                    if final_docs_copy:
                        reordered.append(final_docs_copy.pop(-1))
                final_docs = reordered

        return final_docs