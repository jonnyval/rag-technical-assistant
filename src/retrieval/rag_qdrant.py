from typing import List, Any, Optional
from pydantic import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from src.utils.profiler import TimeProfiler

# Импортируем классы для фильтрации Qdrant
from qdrant_client.http import models

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
    
    # НОВОЕ ПОЛЕ: Для приема фильтра из Streamlit (app_qdrant.py)
    equipment_filter: Optional[List[str]] = Field(default=None, description="Список оборудования для фильтрации")

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        
        k = self.parent_retriever.search_kwargs.get("k", 30)
        
        # === 1. ПОДГОТОВКА ФИЛЬТРА ДЛЯ QDRANT ===
        search_kwargs = {"k": k}
        
        if self.equipment_filter and "Все" not in self.equipment_filter:
            # Формируем нативный фильтр Qdrant (поиск любого совпадения из массива)
            search_kwargs["filter"] = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.equipment_type",
                        match=models.MatchAny(any=self.equipment_filter)
                    )
                ]
            )
        
        # === 2. ПОИСК ДЕТЕЙ ===
        with TimeProfiler(f"Qdrant Hybrid Search (Топ-{k} детей)"):
            # Передаем подготовленные аргументы, включая фильтр
            child_docs = self.parent_retriever.vectorstore.similarity_search(
                query, 
                **search_kwargs
            )

        if not child_docs:
            return []

        # === 3. РЕРАНЖИРОВАНИЕ ===
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

        # === 4. ИЗВЛЕЧЕНИЕ РОДИТЕЛЕЙ ИЗ RAM ===
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

        # === 5. LITM СОРТИРОВКА ===
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