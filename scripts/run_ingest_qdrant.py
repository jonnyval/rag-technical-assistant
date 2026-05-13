import sys
import os
import json
import hashlib
import uuid
import gc  # Сборщик мусора
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import MarkdownTextSplitter
from langchain_community.storage import SQLStore
import pickle
from langchain_classic.storage import EncoderBackedStore

from src.config import settings
from src.logger import log
from src.document_processing.parsers_qdrant_with_llm import process_docx_file, process_html_file
# ИСПРАВЛЕНО: импортируем из существующего файла (не _v2)
from src.document_processing.parsers_tickets_with_llm import process_ticket_file

# Попытка импортировать torch для очистки памяти GPU (если есть)
try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

def get_file_hash(filepath: Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def load_hash_registry() -> Dict[str, dict]:
    # ИСПРАВЛЕНО: settings.hash_registry вместо несуществующего свойства
    if os.path.exists(settings.hash_registry):
        try:
            with open(settings.hash_registry, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_hash_registry(registry: Dict[str, dict]):
    if settings.hash_registry:
        os.makedirs(os.path.dirname(settings.hash_registry), exist_ok=True)
        with open(settings.hash_registry, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=4)

def main():
    log.info("🚀 ЗАПУСК QDRANT ВЕКТОРИЗАЦИИ (DOCKER + SQLITE + BATCHING)")

    log.info(f"Загрузка модели: {settings.embedding_model_name}")
    dense_embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={'device': settings.device},
        encode_kwargs={'normalize_embeddings': True}
    )

    log.info("Загрузка модели FastEmbed Sparse (BM25)...")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    os.makedirs(os.path.dirname(settings.parent_store_path), exist_ok=True)
    
    log.info(f"Подключение к Qdrant Server ({settings.db_url})...")
    
    client = QdrantClient(url=settings.db_url)
    collection_exists = client.collection_exists(settings.collection_name)
    
    if collection_exists:
        log.info("✅ Подключено к существующей коллекции Qdrant.")
        qdrant = QdrantVectorStore(
            client=client,
            collection_name=settings.collection_name,
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID
        )
    else:
        log.warning(f"⚠️ Коллекция '{settings.collection_name}' не найдена. Создаем новую...")
        qdrant = QdrantVectorStore.from_texts(
            texts=["init_document"],
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            collection_name=settings.collection_name,
            url=settings.db_url,
            retrieval_mode=RetrievalMode.HYBRID
        )

    log.info(f"Подключение к хранилищу родителей (SQLite): {settings.parent_store_path}")
    
    byte_store = SQLStore(
        namespace="reglab_parents",
        db_url=f"sqlite:///{settings.parent_store_path}"
    )
    byte_store.create_schema()
    
    store = EncoderBackedStore(
        store=byte_store,
        key_encoder=lambda k: k,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads
    )
    
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

    # ИСПРАВЛЕНО: settings.source_dirs (псевдоним source_directories)
    for source_type, folder_path in settings.source_dirs.items():
        p = Path(folder_path)
        if not p.exists():
            log.warning(f"⚠️ Папка не найдена: {folder_path}")
            continue

        # ИСПРАВЛЕНО: settings.docs_base_urls (новое свойство)
        current_base_url = settings.docs_base_urls.get(source_type, "")
        log.info(f"📂 Сканирование {source_type}")

        if source_type.endswith('_docx'):
            files = list(p.rglob("*.docx"))
            # ИСПРАВЛЕНО: settings.images_out_dir (новое свойство)
            processor = lambda f, st=source_type: process_docx_file(f, st, settings.images_out_dir)
        elif source_type == 'support_tickets':
            # ИСПРАВЛЕНО: ключ в config.yaml теперь тоже называется "support_tickets"
            files = list(p.rglob("*.json"))
            processor = lambda f, st=source_type, url=current_base_url: process_ticket_file(f, st, url)
        else:
            files = list(p.rglob("*.html")) + list(p.rglob("*.htm"))
            processor = lambda f, st=source_type, url=current_base_url: process_html_file(f, st, url)

        files_to_process = []
        for f in files:
            if f.name.startswith('~$') or f.name.endswith('_print.htm'):
                continue
            current_hash = get_file_hash(f)
            file_record = hash_registry.get(f.name, {})
            if file_record.get('hash') != current_hash:
                files_to_process.append((f, current_hash, file_record.get('parent_ids', [])))

        if not files_to_process:
            continue
        log.info(f"🔄 Требуется обработать: {len(files_to_process)} файлов")

        BATCH_SIZE = 5 
        batch_docs = []
        batch_ids = []
        pending_hashes = {}

        for file_path, new_hash, old_parent_ids in tqdm(files_to_process, desc=f"Ингест {source_type}"):
            if old_parent_ids:
                store.mdelete(old_parent_ids)
                try: 
                    qdrant.delete(where={"doc_id": {"$in": old_parent_ids}})
                except Exception as e: 
                    log.error(f"Ошибка удаления чанков {file_path.name}: {e}")

            try:
                parent_docs = processor(file_path)
            except Exception as e:
                log.error(f"❌ Ошибка парсинга файла {file_path.name}: {e}")
                continue

            if parent_docs:
                parent_ids = [str(uuid.uuid4()) for _ in parent_docs]
                batch_docs.extend(parent_docs)
                batch_ids.extend(parent_ids)
                pending_hashes[file_path.name] = {"hash": new_hash, "parent_ids": parent_ids}

            if len(batch_docs) >= BATCH_SIZE:
                retriever.add_documents(batch_docs, ids=batch_ids)
                hash_registry.update(pending_hashes)
                save_hash_registry(hash_registry)
                total_processed += len(pending_hashes)
                batch_docs, batch_ids, pending_hashes = [], [], {}
                gc.collect()
                if has_torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if batch_docs:
            retriever.add_documents(batch_docs, ids=batch_ids)
            hash_registry.update(pending_hashes)
            save_hash_registry(hash_registry)
            total_processed += len(pending_hashes)
            batch_docs, batch_ids, pending_hashes = [], [], {}
            gc.collect()
            if has_torch and torch.cuda.is_available():
                torch.cuda.empty_cache()

    log.info(f"🎉 ИНГЕСТ ЗАВЕРШЕН! Успешно обработано файлов: {total_processed}")

if __name__ == "__main__":
    main()
