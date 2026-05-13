from typing import List, Any, Optional
from pydantic import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_classic.retrievers import ParentDocumentRetriever
from src.utils.profiler import TimeProfiler
from src.retrieval.hyde import generate_hypothetical_document

# Импортируем классы для фильтрации Qdrant
from qdrant_client.http import models


class RegLabQdrantRetriever(BaseRetriever):
    """
    Кастомный ретривер для Qdrant.

    ОПТИМИЗИРОВАННАЯ ВЕРСИЯ: Реранкер работает только по маленьким чанкам,
    и только для победителей извлекаются полные страницы (Parents).

    HyDE (Hypothetical Document Embeddings):
    Если передан hyde_llm — перед поиском LLM генерирует гипотетический
    фрагмент документации. Поиск идёт по его вектору (семантически близко
    к реальным чанкам), реранкер — всегда по оригинальному запросу.
    """

    parent_retriever: ParentDocumentRetriever = Field(
        description="Базовый ретривер Parent-Child"
    )
    reranker_model: Any = Field(
        description="Модель CrossEncoder для перекрестного ранжирования"
    )
    top_k_final: int = 3
    rerank_threshold: float = 0.05
    use_litm: bool = True

    # Фильтр по типу оборудования, устанавливается из Streamlit
    equipment_filter: Optional[List[str]] = Field(
        default=None,
        description="Список оборудования для фильтрации",
    )

    # LLM для HyDE. None = HyDE выключен, поиск идёт по оригинальному запросу
    hyde_llm: Optional[Any] = Field(
        default=None,
        description="LLM для генерации гипотетического документа (HyDE). "
                    "Если None — HyDE отключён.",
    )

    # Только для отладки: хранит последний HyDE-текст, выводится в UI
    _last_hyde_text: str = ""

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        k = self.parent_retriever.search_kwargs.get("k", 30)

        # === 1. HyDE: заменяем поисковый запрос гипотетическим документом ===
        #
        # Идея: вопрос пользователя и текст документации живут в разных
        # семантических пространствах. Гипотетический ответ, написанный
        # в стиле документации, семантически близок к реальным чанкам.
        #
        # Реранкер на шаге 3 ВСЕГДА получает оригинальный query —
        # он оценивает релевантность к тому, что реально спросил пользователь.

        search_query = query
        hyde_used_fallback = True  # считаем "не использован" пока не доказано обратное

        if self.hyde_llm is not None:
            with TimeProfiler("HyDE генерация гипотетического документа"):
                search_query, hyde_used_fallback = generate_hypothetical_document(
                    query=query,
                    llm=self.hyde_llm,
                )
            self._last_hyde_text = search_query
            if not hyde_used_fallback:
                run_manager.on_text(
                    f"[HyDE] Поиск по гипотетическому тексту ({len(search_query)} симв.)",
                    verbose=True,
                )
        else:
            self._last_hyde_text = ""

        # === 2. ПОДГОТОВКА ФИЛЬТРА ДЛЯ QDRANT ===
        search_kwargs: dict = {"k": k}

        if self.equipment_filter and "Все" not in self.equipment_filter:
            search_kwargs["filter"] = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.equipment_type",
                        match=models.MatchAny(any=self.equipment_filter),
                    )
                ]
            )

        # === 3. ПОИСК ДЕТЕЙ ===
        # Используем search_query (HyDE-текст или оригинал при fallback)
        with TimeProfiler(f"Qdrant Hybrid Search (Топ-{k} детей)"):
            child_docs = self.parent_retriever.vectorstore.similarity_search(
                search_query,
                **search_kwargs,
            )

        if not child_docs:
            return []

        # === 4. РЕРАНЖИРОВАНИЕ ===
        # Реранкер ВСЕГДА получает оригинальный query, а не HyDE-текст.
        # Это принципиально: мы ищем по гипотетическому тексту, но оцениваем
        # релевантность к реальному вопросу пользователя.
        with TimeProfiler("CrossEncoder Reranking"):
            pairs = [[query, doc.page_content] for doc in child_docs]  # <-- query!
            scores = self.reranker_model.predict(pairs)

            scored_children = []
            for doc, score in zip(child_docs, scores):
                if score >= self.rerank_threshold:
                    doc.metadata["rerank_score"] = float(score)
                    scored_children.append(doc)

            scored_children.sort(
                key=lambda x: x.metadata["rerank_score"], reverse=True
            )
            top_children = scored_children[: self.top_k_final]

        # === 5. ИЗВЛЕЧЕНИЕ РОДИТЕЛЕЙ ИЗ RAM ===
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
                        best_score = max(
                            c.metadata["rerank_score"]
                            for c in top_children
                            if c.metadata.get("doc_id") == p_id
                        )
                        parent.metadata["rerank_score"] = best_score
                        final_docs.append(parent)

        # === 6. LITM СОРТИРОВКА ===
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