"""
reindex_with_prefix.py
======================
Переиндексация child-векторов с query/document prefix для Qwen3-Embedding.

Что делает:
  - Читает готовые parent-документы из существующей SQLite (parents.db)
  - Создаёт НОВУЮ коллекцию Qdrant (старая не трогается — остаётся резервом)
  - Нарезает parents на child chunks тем же сплиттером что и при ингесте
  - Векторизует child chunks с document prefix (Qwen3 instruction-tuned)
  - Заливает в новую коллекцию

Что НЕ делает:
  - Не запускает LLM (никакого Ollama/Groq)
  - Не парсит исходные docx/html
  - Не трогает parents.db

Время работы: ~20-40 мин на GPU 4 ГБ для базы ~60 МБ
"""

import sys
import os
import pickle
import gc
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

# Импорт после sys.path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import EncoderBackedStore
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.storage import SQLStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from src.config import settings
from src.logger import log

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ─────────────────────────────────────────────
# НАСТРОЙКИ — поменяй при необходимости
# ─────────────────────────────────────────────

# Имя НОВОЙ коллекции (старая сохраняется как резерв)
NEW_COLLECTION = settings.collection_name + "_v2_prefix"

# Prefix для документов при векторизации (Qwen3 instruction format)
DOCUMENT_PREFIX = "Represent this document for retrieval: "

# Батч для векторизации (подбери под свой GPU)
# 4 ГБ VRAM: начни с 64, если OOM — снизь до 32
EMBED_BATCH_SIZE = 32

# Батч для загрузки в Qdrant
QDRANT_BATCH_SIZE = 256

# ─────────────────────────────────────────────


def load_all_parents(db_path: str) -> list[tuple[str, Document]]:
    """
    Читает все parent-документы из SQLite.
    Возвращает список (parent_id, Document).
    """
    log.info(f"Подключение к SQLite: {db_path}")

    byte_store = SQLStore(
        namespace="reglab_parents",
        db_url=f"sqlite:///{db_path}"
    )
    store = EncoderBackedStore(
        store=byte_store,
        key_encoder=lambda k: k,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads,
    )

    log.info("Получение всех ключей из хранилища...")
    # SQLStore.yield_keys() возвращает ключи из namespace
    keys = list(byte_store.yield_keys())
    log.info(f"Найдено parent-документов: {len(keys)}")

    if not keys:
        log.error("Хранилище пустое! Проверь путь к parents.db и namespace.")
        sys.exit(1)

    log.info("Загрузка документов из хранилища (mget)...")
    docs = store.mget(keys)

    result = []
    skipped = 0
    for key, doc in zip(keys, docs):
        if doc is None:
            skipped += 1
            continue
        if not isinstance(doc, Document):
            skipped += 1
            log.warning(f"Пропущен ключ {key}: ожидался Document, получен {type(doc)}")
            continue
        result.append((key, doc))

    log.info(f"Загружено: {len(result)} документов, пропущено: {skipped}")
    return result


def build_child_chunks(
    parents: list[tuple[str, Document]],
    child_splitter: MarkdownTextSplitter,
) -> list[Document]:
    """
    Нарезает parents на child chunks.
    Проставляет metadata["doc_id"] = parent_id (как это делает ParentDocumentRetriever).
    Добавляет document prefix в начало page_content для правильной векторизации Qwen3.
    """
    log.info(f"Нарезка {len(parents)} parents на child chunks...")
    all_children = []

    for parent_id, parent_doc in tqdm(parents, desc="Нарезка"):
        children = child_splitter.split_documents([parent_doc])

        # Вопросы и ключевые слова из метаданных parent-документа.
        # Добавляем их в каждый child-чанк, только если их ещё нет в тексте
        # (старые parents могли быть проиндексированы без этого блока).
        questions = parent_doc.metadata.get("generated_questions", "")
        keywords  = parent_doc.metadata.get("keywords", "")

        for child in children:
            # Копируем метаданные parent → child
            child.metadata.update(parent_doc.metadata)
            # doc_id — ключ для извлечения parent при поиске
            child.metadata["doc_id"] = parent_id

            # Дополняем текст вопросами/ключевыми словами если их ещё нет
            suffix = ""
            if questions and "[ПОТЕНЦИАЛЬНЫЕ ВОПРОСЫ:" not in child.page_content:
                suffix += f"\n\n[ПОТЕНЦИАЛЬНЫЕ ВОПРОСЫ: {questions}]"
            if keywords and "[КЛЮЧЕВЫЕ СЛОВА:" not in child.page_content:
                suffix += f"\n[КЛЮЧЕВЫЕ СЛОВА: {keywords}]"
            if suffix:
                child.page_content = child.page_content + suffix

            # Добавляем document prefix для Qwen3 (всегда последним)
            child.page_content = DOCUMENT_PREFIX + child.page_content

        all_children.extend(children)

    log.info(f"Итого child chunks: {len(all_children)}")
    return all_children


def create_new_collection(
    client: QdrantClient,
    collection_name: str,
    dense_embeddings: HuggingFaceEmbeddings,
) -> QdrantVectorStore:
    """
    Создаёт новую Qdrant коллекцию с hybrid-индексом.
    Если коллекция уже существует — останавливается (чтобы не затереть случайно).
    """
    if client.collection_exists(collection_name):
        log.error(
            f"Коллекция '{collection_name}' уже существует!\n"
            f"Переименуй NEW_COLLECTION в скрипте или удали её вручную:\n"
            f"  curl -X DELETE http://localhost:6333/collections/{collection_name}"
        )
        sys.exit(1)

    log.info(f"Создание новой коллекции: {collection_name}")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    qdrant = QdrantVectorStore.from_texts(
        texts=["_init_"],
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        collection_name=collection_name,
        url=settings.db_url,
        retrieval_mode=RetrievalMode.HYBRID,
    )

    # Удаляем init-документ
    results = client.scroll(
        collection_name=collection_name,
        limit=10,
        with_payload=True,
    )
    init_ids = [
        p.id for p in results[0]
        if p.payload.get("page_content") == "_init_"
    ]
    if init_ids:
        client.delete(
            collection_name=collection_name,
            points_selector=qdrant_models.PointIdsList(points=init_ids),
        )

    return qdrant


def index_children_batched(
    qdrant: QdrantVectorStore,
    children: list[Document],
    batch_size: int,
) -> None:
    """
    Заливает child chunks в Qdrant батчами.
    Векторизация происходит внутри qdrant.add_documents через embeddings модель.
    """
    total = len(children)
    log.info(f"Индексация {total} child chunks (батч={batch_size})...")

    for i in tqdm(range(0, total, batch_size), desc="Индексация в Qdrant"):
        batch = children[i : i + batch_size]
        try:
            qdrant.add_documents(batch)
        except Exception as e:
            log.error(f"Ошибка при индексации батча {i}–{i+batch_size}: {e}")
            raise

        # Освобождаем память GPU каждые 10 батчей
        if (i // batch_size) % 10 == 0:
            gc.collect()
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()

    log.info("Индексация завершена.")


def verify_collection(client: QdrantClient, collection_name: str, expected_children: int) -> None:
    """Проверяет что коллекция создана и содержит ожидаемое количество точек."""
    info = client.get_collection(collection_name)
    actual = info.points_count
    log.info(f"Проверка коллекции '{collection_name}': {actual} точек (ожидалось ~{expected_children})")

    if actual < expected_children * 0.95:
        log.warning(
            f"Точек меньше ожидаемого на >5%. "
            f"Возможно часть батчей не загрузилась. "
            f"Проверь логи выше на наличие ошибок."
        )
    else:
        log.info("✅ Коллекция выглядит корректно.")


def print_next_steps(new_collection: str, old_collection: str) -> None:
    log.info("\n" + "="*60)
    log.info("ПЕРЕИНДЕКСАЦИЯ ЗАВЕРШЕНА")
    log.info("="*60)
    log.info(f"Новая коллекция (с prefix): {new_collection}")
    log.info(f"Резервная коллекция:        {old_collection}")
    log.info("")
    log.info("Чтобы переключить app на новую коллекцию,")
    log.info("измени в config.yaml:")
    log.info(f"  qdrant_v2_docker:")
    log.info(f"    collection: \"{new_collection}\"")
    log.info("")
    log.info("Также добавь query prefix в app_qdrant.py (см. комментарий ниже).")
    log.info("="*60)


def main():
    """Запускает вторую версию переиндексации с расширенной подготовкой child-чанков."""

    log.info("🚀 ЗАПУСК ПЕРЕИНДЕКСАЦИИ С QUERY/DOCUMENT PREFIX")
    log.info(f"Новая коллекция: {NEW_COLLECTION}")
    log.info(f"Старая коллекция (резерв): {settings.collection_name}")
    log.info(f"Document prefix: '{DOCUMENT_PREFIX}'")

    # 1. Загрузка embeddings модели с document prefix
    log.info(f"Загрузка эмбеддинг-модели: {settings.embedding_model_name}")
    dense_embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": settings.device},
        encode_kwargs={
            "normalize_embeddings": True,
            "prompt": DOCUMENT_PREFIX,  # ← ключевое изменение
        },
    )

    # 2. Чтение parents из SQLite
    parents = load_all_parents(settings.parent_store_path)

    # 3. Нарезка на child chunks
    child_splitter = MarkdownTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
    )
    children = build_child_chunks(parents, child_splitter)

    # 4. Создание новой коллекции
    client = QdrantClient(url=settings.db_url)
    qdrant = create_new_collection(client, NEW_COLLECTION, dense_embeddings)

    # 5. Индексация
    index_children_batched(qdrant, children, batch_size=QDRANT_BATCH_SIZE)

    # 6. Проверка
    verify_collection(client, NEW_COLLECTION, len(children))

    # 7. Инструкция
    print_next_steps(NEW_COLLECTION, settings.collection_name)


if __name__ == "__main__":
    main()
