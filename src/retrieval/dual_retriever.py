"""
dual_retriever.py — параллельный поиск по двум БД (документация + тикеты).

Путь: src/retrieval/dual_retriever.py

Требует в config.yaml:
    storage:
      vector_db:
        active:    "qdrant_v2_docker"      # docs-БД
        second_db: "qdrant_tickets_test"   # tickets-БД
"""

import pickle
import logging
from typing import List, Optional, Any, Dict

from pydantic import Field
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.storage import SQLStore
from langchain_classic.storage import EncoderBackedStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from src.utils.profiler import TimeProfiler

log = logging.getLogger("RegLabRAG")


# ==========================================
# 🔀 DUAL RETRIEVER
# ==========================================

class DualRetriever(BaseRetriever):
    """Параллельный поиск по двум Qdrant-коллекциям с единым реранкингом.

    Схема работы:
      1. Поиск top_k дочерних чанков в БД-1 (документация)
      2. Поиск top_k дочерних чанков в БД-2 (тикеты)
      3. Объединение + реранкинг общего пула
      4. Выбор top_k_final лучших, извлечение родителей из каждой БД
      5. LiTM-сортировка финального набора

    В метаданных каждого документа добавляется поле `db_source`:
      "docs"    — из базы документации
      "tickets" — из базы тикетов

    Параметры:
        docs_retriever      ParentDocumentRetriever для БД документации
        tickets_retriever   ParentDocumentRetriever для БД тикетов
        reranker_model      CrossEncoder из sentence_transformers
        top_k_final         сколько документов отдать LLM (суммарно)
        rerank_threshold    минимальный score реранкера
        use_litm            применять ли Lost-in-the-Middle перестановку
        equipment_filter    фильтр по equipment_type (только для docs-БД)
        tickets_weight      множитель score для тикетов (0.0–1.0);
                            понизьте до 0.7–0.8 если тикеты вытесняют документацию
    """

    docs_retriever:    ParentDocumentRetriever = Field(description="Ретривер документации")
    tickets_retriever: ParentDocumentRetriever = Field(description="Ретривер тикетов")
    reranker_model:    Any                     = Field(default=None, description="CrossEncoder реранкер. None = реранкер отключён.")

    top_k_final:      int   = 5
    rerank_threshold: float = 0.05
    use_litm:         bool  = True
    tickets_weight:   float = 0.7

    equipment_filter: Optional[List[str]] = Field(
        default=None,
        description="Фильтр по equipment_type (только для docs-БД)",
    )

    class Config:
        """Разрешает хранить в Pydantic-модели внешние объекты LangChain и reranker."""

        arbitrary_types_allowed = True

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _build_search_kwargs(self, k: int, apply_equipment_filter: bool) -> Dict:
        """Формирует параметры Qdrant-поиска, включая опциональный фильтр оборудования."""

        kwargs: Dict = {"k": k}
        if (
            apply_equipment_filter
            and self.equipment_filter
            and "Все" not in self.equipment_filter
        ):
            kwargs["filter"] = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.equipment_type",
                        match=qdrant_models.MatchAny(any=self.equipment_filter),
                    )
                ]
            )
        return kwargs

    def _search_children(
        self,
        retriever: ParentDocumentRetriever,
        query: str,
        k: int,
        apply_filter: bool,
        db_label: str,
    ) -> List[Document]:
        """Ищет дочерние чанки в одной БД и помечает их меткой источника."""
        try:
            with TimeProfiler(f"Qdrant Search [{db_label}] top-{k}"):
                docs = retriever.vectorstore.similarity_search(
                    query,
                    **self._build_search_kwargs(k, apply_filter),
                )
            for d in docs:
                d.metadata["db_source"] = db_label
            log.debug(f"  [{db_label}] найдено {len(docs)} чанков")
            return docs
        except Exception as e:
            log.error(f"❌ Ошибка поиска в [{db_label}]: {e}", exc_info=True)
            return []

    def _fetch_parents(
        self,
        retriever: ParentDocumentRetriever,
        top_children: List[Document],
        db_label: str,
    ) -> List[Document]:
        """Извлекает родительские документы из конкретной БД или возвращает сам чанк, если родителя нет."""
        parent_ids = []
        result = []

        # Разделяем чанки на те, у которых есть родитель, и самостоятельные
        for child in top_children:
            if child.metadata.get("db_source") != db_label:
                continue
            
            pid = child.metadata.get("doc_id")
            if pid:
                if pid not in parent_ids:
                    parent_ids.append(pid)
            else:
                # Если doc_id нет (наш случай с тикетами), 
                # просто прокидываем сам чанк дальше как готовый документ
                child.metadata["db_source"] = db_label
                result.append(child)

        # Если есть чанки с doc_id, достаем их родителей из базы
        if parent_ids:
            parents = retriever.docstore.mget(parent_ids)
            for parent, pid in zip(parents, parent_ids):
                if not parent:
                    continue
                best_score = max(
                    c.metadata["rerank_score"]
                    for c in top_children
                    if c.metadata.get("doc_id") == pid
                    and c.metadata.get("db_source") == db_label
                )
                parent.metadata["rerank_score"] = best_score
                parent.metadata["db_source"] = db_label
                result.append(parent)

        log.debug(f"  [{db_label}] извлечено {len(result)} документов (включая документы без родителей)")
        return result

    def _rank_children(self, query: str, children: List[Document], *, limit: int | None = None, use_reranker: bool = True) -> List[Document]:
        """Ранжирует дочерние чанки внутри одного источника, не смешивая документацию и тикеты."""
        if not children:
            return []
        result_limit = limit or self.top_k_final

        if use_reranker and self.reranker_model is not None:
            with TimeProfiler(f"CrossEncoder Reranking ({len(children)} chunks)"):
                pairs = [[query, d.page_content] for d in children]
                scores = self.reranker_model.predict(pairs)

                scored: List[Document] = []
                for doc, score in zip(children, scores):
                    score = float(score)
                    if score >= self.rerank_threshold:
                        doc.metadata["rerank_score"] = score
                        scored.append(doc)

                scored.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
                return scored[:result_limit]

        for doc in children:
            doc.metadata.setdefault("rerank_score", 0.0)
        return children[:result_limit]

    @staticmethod
    def _unique_parent_children(children: List[Document], limit: int) -> List[Document]:
        """Keep the best child per parent so one ticket cannot consume the whole limit.

        Tickets are short and their symptoms and solution often become separate
        child chunks. Parent retrieval deduplicates them later, but doing that
        only after applying ``final_limit`` can return fewer unique tickets than
        requested. Deduplicate ranked children first while preserving rank.
        """
        result: List[Document] = []
        seen = set()
        for child in children:
            metadata = child.metadata if hasattr(child, "metadata") else {}
            key = (
                metadata.get("doc_id")
                or metadata.get("ticket_id")
                or metadata.get("source_file")
                or id(child)
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(child)
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _multi_query_child_key(document: Document) -> tuple:
        """Stable key for a child chunk returned by several query variants."""
        metadata = document.metadata if hasattr(document, "metadata") else {}
        return (
            metadata.get("db_source"),
            metadata.get("doc_id") or metadata.get("ticket_id") or metadata.get("source_file"),
            document.page_content,
        )

    def _fuse_multi_query_children(
        self,
        query_results: List[List[Document]],
        *,
        rrf_k: int,
        candidate_limit: int,
    ) -> List[Document]:
        """Fuse ranked vector-search lists with Reciprocal Rank Fusion."""
        fused: Dict[tuple, Document] = {}
        scores: Dict[tuple, float] = {}
        for result in query_results:
            for rank, document in enumerate(result, start=1):
                key = self._multi_query_child_key(document)
                fused.setdefault(key, document)
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

        ordered = sorted(fused, key=lambda key: scores[key], reverse=True)[:candidate_limit]
        children = [fused[key] for key in ordered]
        for key, child in zip(ordered, children):
            child.metadata["fusion_score"] = scores[key]
        return children

    def retrieve_docs_multi_query(
        self,
        queries: List[str],
        *,
        rerank_query: str,
        use_reranker: bool = True,
        rrf_k: int = 60,
        candidate_limit: int = 60,
    ) -> List[Document]:
        """RAG-Fusion over documentation, followed by one shared rerank pass."""
        unique_queries = list(dict.fromkeys(query.strip() for query in queries if query and query.strip()))
        if len(unique_queries) <= 1:
            return self.retrieve_docs(rerank_query, use_reranker=use_reranker)
        k = self.docs_retriever.search_kwargs.get("k", 30)
        children = self._fuse_multi_query_children(
            [self._search_children(self.docs_retriever, query, k, False, "docs") for query in unique_queries],
            rrf_k=rrf_k,
            candidate_limit=candidate_limit,
        )
        ranked = self._rank_children(rerank_query, children, use_reranker=use_reranker)
        if not use_reranker:
            for child in ranked:
                child.metadata["rerank_score"] = child.metadata.get("fusion_score", 0.0)
        docs = self._fetch_parents(self.docs_retriever, ranked, "docs")
        docs.sort(key=lambda document: document.metadata.get("rerank_score", 0), reverse=True)
        return docs

    def retrieve_tickets_multi_query(
        self,
        queries: List[str],
        *,
        rerank_query: str,
        final_limit: int | None = None,
        child_k: int | None = None,
        use_reranker: bool = True,
        rrf_k: int = 60,
        candidate_limit: int = 60,
    ) -> List[Document]:
        """RAG-Fusion over support tickets, followed by one shared rerank pass."""
        unique_queries = list(dict.fromkeys(query.strip() for query in queries if query and query.strip()))
        if len(unique_queries) <= 1:
            return self.retrieve_tickets(
                rerank_query,
                final_limit=final_limit,
                child_k=child_k,
                use_reranker=use_reranker,
            )
        k = child_k or self.tickets_retriever.search_kwargs.get("k", 30)
        children = self._fuse_multi_query_children(
            [self._search_children(self.tickets_retriever, query, k, False, "tickets") for query in unique_queries],
            rrf_k=rrf_k,
            candidate_limit=candidate_limit,
        )
        ranked = self._rank_children(rerank_query, children, limit=len(children), use_reranker=use_reranker)
        if not use_reranker:
            for child in ranked:
                child.metadata["rerank_score"] = child.metadata.get("fusion_score", 0.0)
        top_children = self._unique_parent_children(ranked, final_limit or self.top_k_final)
        tickets = self._fetch_parents(self.tickets_retriever, top_children, "tickets")
        tickets.sort(key=lambda document: document.metadata.get("rerank_score", 0), reverse=True)
        return tickets
    def retrieve_docs(self, query: str, *, use_reranker: bool = True) -> List[Document]:
        """Ищет только по коллекции документации без фильтра по оборудованию."""
        k = self.docs_retriever.search_kwargs.get("k", 30)
        children = self._search_children(
            self.docs_retriever,
            query,
            k,
            apply_filter=False,
            db_label="docs",
        )
        top_children = self._rank_children(query, children, use_reranker=use_reranker)
        docs = self._fetch_parents(self.docs_retriever, top_children, "docs")
        docs.sort(key=lambda x: x.metadata.get("rerank_score", 0), reverse=True)
        return docs
    def retrieve_tickets(
        self,
        query: str,
        *,
        final_limit: int | None = None,
        child_k: int | None = None,
        use_reranker: bool = True,
    ) -> List[Document]:
        """Ищет только по коллекции обращений технической поддержки."""
        k = child_k or self.tickets_retriever.search_kwargs.get("k", 30)
        children = self._search_children(
            self.tickets_retriever,
            query,
            k,
            apply_filter=False,
            db_label="tickets",
        )
        result_limit = final_limit or self.top_k_final
        ranked_children = self._rank_children(
            query,
            children,
            limit=len(children),
            use_reranker=use_reranker,
        )
        top_children = self._unique_parent_children(ranked_children, result_limit)
        tickets = self._fetch_parents(self.tickets_retriever, top_children, "tickets")
        tickets.sort(key=lambda x: x.metadata.get("rerank_score", 0), reverse=True)
        return tickets
    def retrieve_tickets_by_module_codes(self, module_codes: List[str], *, limit: int = 30) -> List[Document]:
        """Достаёт обращения с точным совпадением module_code из payload metadata."""
        codes = [str(code).strip() for code in module_codes if str(code).strip()]
        if not codes:
            return []

        vectorstore = self.tickets_retriever.vectorstore
        client = vectorstore.client
        collection_name = vectorstore.collection_name
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.mentioned_module_codes",
                        match=qdrant_models.MatchAny(any=codes),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        child_docs: List[Document] = []
        for point in points:
            payload = point.payload or {}
            metadata = dict(payload.get("metadata") or {})
            metadata["db_source"] = "tickets"
            metadata.setdefault("rerank_score", 0.0)
            child_docs.append(Document(
                page_content=str(payload.get("page_content") or ""),
                metadata=metadata,
            ))

        if not child_docs:
            return []

        parent_ids: List[str] = []
        seen_parent_ids = set()
        direct_docs: List[Document] = []
        for doc in child_docs:
            parent_id = doc.metadata.get("doc_id")
            if parent_id and parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                parent_ids.append(parent_id)
            elif not parent_id:
                direct_docs.append(doc)

        result: List[Document] = []
        if parent_ids:
            parents = self.tickets_retriever.docstore.mget(parent_ids)
            for parent in parents:
                if not parent:
                    continue
                parent.metadata["db_source"] = "tickets"
                parent.metadata.setdefault("rerank_score", 0.0)
                result.append(parent)
        result.extend(direct_docs)

        deduped: List[Document] = []
        seen_tickets = set()
        for doc in result:
            ticket_id = doc.metadata.get("ticket_id") or doc.metadata.get("source_file") or id(doc)
            if ticket_id in seen_tickets:
                continue
            seen_tickets.add(ticket_id)
            deduped.append(doc)
        return deduped

    def retrieve_docs_by_page_titles(self, page_titles: List[str], *, limit: int = 20) -> List[Document]:
        """Достаёт документы по точному metadata.page_title из документационной коллекции."""
        titles = [str(title).strip() for title in page_titles if str(title).strip()]
        if not titles:
            return []

        vectorstore = self.docs_retriever.vectorstore
        client = vectorstore.client
        collection_name = vectorstore.collection_name
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.page_title",
                        match=qdrant_models.MatchAny(any=titles),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        parent_ids: List[str] = []
        seen_parent_ids = set()
        direct_docs: List[Document] = []
        for point in points:
            payload = point.payload or {}
            metadata = dict(payload.get("metadata") or {})
            metadata["db_source"] = "docs"
            metadata.setdefault("rerank_score", 1.0)
            parent_id = metadata.get("doc_id")
            if parent_id and parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                parent_ids.append(parent_id)
            elif not parent_id:
                direct_docs.append(Document(
                    page_content=str(payload.get("page_content") or ""),
                    metadata=metadata,
                ))

        result: List[Document] = []
        if parent_ids:
            parents = self.docs_retriever.docstore.mget(parent_ids)
            for parent in parents:
                if not parent:
                    continue
                parent.metadata["db_source"] = "docs"
                parent.metadata.setdefault("rerank_score", 1.0)
                result.append(parent)
        result.extend(direct_docs)

        deduped: List[Document] = []
        seen_docs = set()
        for doc in result:
            key = (
                doc.metadata.get("source_file"),
                doc.metadata.get("page_title"),
                doc.metadata.get("breadcrumb_raw"),
            )
            if key in seen_docs:
                continue
            seen_docs.add(key)
            deduped.append(doc)
        return deduped

    # ------------------------------------------------------------------
    # Основной метод
    # ------------------------------------------------------------------

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Выполняет совместимый с LangChain смешанный поиск по docs и tickets."""

        k = self.docs_retriever.search_kwargs.get("k", 30)

        # === 1. ПОИСК В ОБЕИХ БД ===
        docs_children    = self._search_children(self.docs_retriever,    query, k, apply_filter=False, db_label="docs")
        tickets_children = self._search_children(self.tickets_retriever, query, k, apply_filter=False, db_label="tickets")

        all_children = docs_children + tickets_children

        if not all_children:
            log.warning("DualRetriever: обе БД вернули 0 чанков")
            return []

        # === 2. РЕРАНКИНГ ОБЪЕДИНЁННОГО ПУЛА (опционально) ===
        if self.reranker_model is not None:
            with TimeProfiler(f"CrossEncoder Reranking (пул {len(all_children)} чанков)"):
                pairs  = [[query, d.page_content] for d in all_children]
                scores = self.reranker_model.predict(pairs)

                scored: List[Document] = []
                for doc, score in zip(all_children, scores):
                    effective_score = float(score)
                    if doc.metadata.get("db_source") == "tickets":
                        effective_score *= self.tickets_weight

                    if effective_score >= self.rerank_threshold:
                        doc.metadata["rerank_score"] = effective_score
                        scored.append(doc)

                scored.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
                top_children = scored[: self.top_k_final]
        else:
            # Реранкер отключён — берём top_k_final по векторному скору
            for doc in all_children:
                doc.metadata.setdefault("rerank_score", 0.0)
            top_children = all_children[: self.top_k_final]

        log.info(
            f"DualRetriever: топ-{len(top_children)} чанков "
            f"(docs={sum(1 for d in top_children if d.metadata.get('db_source')=='docs')}, "
            f"tickets={sum(1 for d in top_children if d.metadata.get('db_source')=='tickets')})"
        )

        # === 3. ИЗВЛЕЧЕНИЕ РОДИТЕЛЕЙ ===
        with TimeProfiler("Извлечение родителей (обе БД)"):
            docs_parents    = self._fetch_parents(self.docs_retriever,    top_children, "docs")
            tickets_parents = self._fetch_parents(self.tickets_retriever, top_children, "tickets")
            final_docs = docs_parents + tickets_parents

        final_docs.sort(key=lambda x: x.metadata.get("rerank_score", 0), reverse=True)

        # === 4. LITM ПЕРЕСТАНОВКА ===
        if self.use_litm and len(final_docs) > 2:
            with TimeProfiler("LiTM сортировка"):
                reordered: List[Document] = []
                pool = final_docs.copy()
                while pool:
                    reordered.append(pool.pop(0))
                    if pool:
                        reordered.append(pool.pop(-1))
                final_docs = reordered

        return final_docs


# ==========================================
# 🏭 ФАБРИКА
# ==========================================

def build_dual_retriever(
    dense_embeddings: HuggingFaceEmbeddings,
    reranker_model: Any,
    top_k_retrieval: int | None = None,
    top_k_final: int | None = None,
    rerank_threshold: float | None = None,
    use_litm: bool | None = None,
) -> DualRetriever:
    """Создает dual-retriever, который умеет отдельно искать по документации и тикетам."""

    """Строит DualRetriever из двух бэкендов, описанных в config.yaml.

    Args:
        dense_embeddings: уже загруженная HuggingFaceEmbeddings модель
        reranker_model:   уже загруженный CrossEncoder

    Returns:
        DualRetriever, готовый к использованию
    """
    from src.config import settings

    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    def _make_parent_retriever(backend_name: str) -> ParentDocumentRetriever:
        """Собирает ParentDocumentRetriever для одного бэкенда из config."""
        backend = settings.db_backends.get(backend_name)
        if not backend:
            raise ValueError(
                f"Бэкенд '{backend_name}' не найден в config.yaml. "
                f"Доступные: {list(settings.db_backends.keys())}"
            )

        url  = backend.get("url")
        path = backend.get("path")
        if url:
            client = QdrantClient(url=url)
            log.info(f"  [{backend_name}] Qdrant @ {url}")
        else:
            client = QdrantClient(path=str(path))
            log.info(f"  [{backend_name}] Qdrant path={path}")

        qdrant = QdrantVectorStore(
            client=client,
            collection_name=backend["collection"],
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
        )

        parent_store_cfg = backend.get("parent_store", {})
        store_path = (
            parent_store_cfg.get("path", "vector_dbs/parent_docstore.db")
            if isinstance(parent_store_cfg, dict)
            else str(parent_store_cfg)
        )

        byte_store = SQLStore(
            namespace="reglab_parents", 
            db_url=f"sqlite:///{store_path}",
        )
        store = EncoderBackedStore(
            store=byte_store,
            key_encoder=lambda k: k,
            value_serializer=pickle.dumps,
            value_deserializer=pickle.loads,
        )

        child_splitter = MarkdownTextSplitter(
            chunk_size=settings.child_chunk_size,
            chunk_overlap=settings.child_chunk_overlap,
        )

        return ParentDocumentRetriever(
            vectorstore=qdrant,
            docstore=store,
            child_splitter=child_splitter,
            search_kwargs={"k": top_k_retrieval or settings.top_k_retrieval},
        )

    docs_backend_name    = settings.active_db_name
    tickets_backend_name = settings.second_db_name

    log.info(f"🔀 DualRetriever: docs={docs_backend_name}, tickets={tickets_backend_name}")

    docs_retriever    = _make_parent_retriever(docs_backend_name)
    tickets_retriever = _make_parent_retriever(tickets_backend_name)

    return DualRetriever(
        docs_retriever=docs_retriever,
        tickets_retriever=tickets_retriever,
        reranker_model=reranker_model,
        top_k_final=top_k_final or settings.top_k_final,
        rerank_threshold=rerank_threshold if rerank_threshold is not None else settings.rerank_threshold,
        use_litm=use_litm if use_litm is not None else settings.use_litm,
    )
