import sys
import os
import json
import hashlib
import uuid
import pickle
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_classic.storage import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import MarkdownTextSplitter

from src.config import settings
from src.logger import log
from src.document_processing.parsers_qdrant import process_docx_file, process_html_file

def get_file_hash(filepath: Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def load_hash_registry() -> Dict[str, dict]:
    if os.path.exists(settings.hash_registry):
        try:
            with open(settings.hash_registry, 'r', encoding='utf-8') as f: return json.load(f)
        except Exception: pass
    return {}

def save_hash_registry(registry: Dict[str, dict]):
    if settings.hash_registry:
        os.makedirs(os.path.dirname(settings.hash_registry), exist_ok=True)
        with open(settings.hash_registry, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=4)

def main():
    log.info("🚀 ЗАПУСК QDRANT ВЕКТОРИЗАЦИИ (PARENT-CHILD + HYBRID SEARCH)")

    log.info(f"Загрузка модели: {settings.embedding_model_name}")
    dense_embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={'device': settings.device},
        encode_kwargs={'normalize_embeddings': True}
    )

    log.info("Загрузка модели FastEmbed Sparse (BM25)...")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    os.makedirs(settings.db_path, exist_ok=True)
    os.makedirs(settings.parent_store_path, exist_ok=True)
    
    log.info(f"Подключение к Qdrant ({settings.db_path})...")
    
    # 1. Открываем временного клиента только чтобы проверить наличие базы
    temp_client = QdrantClient(path=settings.db_path)
    collection_exists = temp_client.collection_exists(settings.collection_name)
    
    # 2. ЗАКРЫВАЕМ клиента, чтобы он снял блокировку (lock) с файлов Windows!
    temp_client.close()
    
    # 3. Теперь LangChain может безопасно создавать свои подключения
    if collection_exists:
        log.info("✅ Подключено к существующей коллекции Qdrant.")
        qdrant = QdrantVectorStore.from_existing_collection(
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            collection_name=settings.collection_name,
            path=settings.db_path,
            retrieval_mode=RetrievalMode.HYBRID
        )
    else:
        log.warning(f"⚠️ Коллекция '{settings.collection_name}' не найдена. Создаем новую...")
        qdrant = QdrantVectorStore.from_texts(
            texts=["init_document"],
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            collection_name=settings.collection_name,
            path=settings.db_path,  # Передаем именно path, а не client
            retrieval_mode=RetrievalMode.HYBRID
        )

    # ---------------------------------------------------------
    # 4. НАДЕЖНОЕ ХРАНИЛИЩЕ РОДИТЕЛЕЙ (IN-MEMORY + PICKLE BACKUP)
    # ---------------------------------------------------------
    store = InMemoryStore()
    parent_store_file = os.path.join(settings.parent_store_path, "parents_store.pkl")

    if os.path.exists(parent_store_file):
        with open(parent_store_file, 'rb') as f:
            store.store = pickle.load(f)
            log.info(f"💾 Загружено {len(store.store)} родительских документов из кэша")

    # Умный нарезчик Детей (не режет таблицы благодаря MarkdownTextSplitter)
    child_splitter = MarkdownTextSplitter(
        chunk_size=settings.child_chunk_size, 
        chunk_overlap=settings.child_chunk_overlap
    )
    
    retriever = ParentDocumentRetriever(
        vectorstore=qdrant,
        docstore=store,
        child_splitter=child_splitter,
    )

    hash_registry = load_hash_registry()
    total_processed = 0

    for source_type, folder_path in settings.source_dirs.items():
        p = Path(folder_path)
        if not p.exists():
            log.warning(f"⚠️ Папка не найдена, пропускаем: {folder_path} ({source_type})")
            continue

        current_base_url = settings.docs_base_urls.get(source_type, "")
        log.info(f"📂 Сканирование {source_type}")

        if source_type.endswith('_docx'):
            files = list(p.rglob("*.docx"))
            processor = lambda f: process_docx_file(f, source_type, settings.images_out_dir)
        else:
            files = list(p.rglob("*.html")) + list(p.rglob("*.htm"))
            processor = lambda f: process_html_file(f, source_type, current_base_url)

        files_to_process = []
        for f in files:
            if f.name.startswith('~$') or f.name.endswith('_print.htm'): continue
            
            current_hash = get_file_hash(f)
            file_record = hash_registry.get(f.name, {})
            
            if file_record.get('hash') != current_hash:
                files_to_process.append((f, current_hash, file_record.get('parent_ids', [])))

        if not files_to_process: continue

        log.info(f"🔄 Требуется обработать: {len(files_to_process)} файлов")

        for file_path, new_hash, old_parent_ids in tqdm(files_to_process, desc=f"Ингест {source_type}"):
            # Очистка старых версий файла
            if old_parent_ids:
                store.mdelete(old_parent_ids)
                try: qdrant.delete(where={"doc_id": {"$in": old_parent_ids}})
                except Exception: pass

            parent_docs = processor(file_path)

            if parent_docs:
                parent_ids = [str(uuid.uuid4()) for _ in parent_docs]
                retriever.add_documents(parent_docs, ids=parent_ids)
                total_processed += 1

                hash_registry[file_path.name] = {"hash": new_hash, "parent_ids": parent_ids}
                save_hash_registry(hash_registry)

    # ---------------------------------------------------------
    # 5. СОХРАНЯЕМ РОДИТЕЛЬСКИЕ ДОКУМЕНТЫ НА ДИСК
    # ---------------------------------------------------------
    with open(parent_store_file, 'wb') as f:
        pickle.dump(store.store, f)
    log.info("💾 Родительские документы успешно сохранены на диск.")

    log.info(f"🎉 ИНГЕСТ ЗАВЕРШЕН! Успешно обработано файлов: {total_processed}")

if __name__ == "__main__":
    main()