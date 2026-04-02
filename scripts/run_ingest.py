import sys
import os
import json
import hashlib
import gc
import torch
from pathlib import Path
from typing import Dict
from tqdm import tqdm

# Добавляем корень проекта для импортов
sys.path.append(str(Path(__file__).resolve().parent.parent))

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Импортируем наши модули
from src.config import settings
from src.logger import log
from src.document_processing.parsers import process_docx_file, process_html_file

# ============================================================================
# УМНОЕ ОБНОВЛЕНИЕ (MD5 HASHING)
# ============================================================================
def get_file_hash(filepath: Path) -> str:
    """Вычисляет MD5 хэш файла."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def load_hash_registry() -> Dict[str, str]:
    """Загружает историю хэшей из файла (кэш)."""
    if os.path.exists(settings.hash_registry):
        try:
            with open(settings.hash_registry, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Ошибка чтения кэша хэшей: {e}. Начинаем с чистого листа.")
    return {}

def save_hash_registry(registry: Dict[str, str]):
    """Сохраняет актуальные хэши в файл."""
    os.makedirs(os.path.dirname(settings.hash_registry), exist_ok=True)
    with open(settings.hash_registry, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=4)

# ============================================================================
# ГЛАВНЫЙ ПРОЦЕСС ЗАГРУЗКИ
# ============================================================================
def main():
    log.info("🚀 ЗАПУСК УМНОЙ ВЕКТОРИЗАЦИИ БАЗЫ ЗНАНИЙ")

    # 1. Инициализация моделей (с настройками из config.yaml)
    log.info(f"Загрузка модели эмбеддингов: {settings.embedding_model_name} на {settings.device}")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={'device': settings.device},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. Подключение к БД
    log.info(f"Подключение к базе: {settings.db_path}")
    vectorstore = Chroma(
        persist_directory=settings.db_path,
        embedding_function=embeddings,
        collection_name=settings.collection_name
    )

    hash_registry = load_hash_registry()
    chunk_buffer = []
    WRITE_BATCH_SIZE = 100  # Сколько чанков копим перед записью в БД
    total_processed = 0

    # 3. Обход директорий из config.yaml
    for source_type, folder_path in settings.source_dirs.items():
        p = Path(folder_path)
        if not p.exists():
            log.warning(f"⚠️ Папка не найдена: {folder_path}")
            continue

        # Получаем базовый URL для HTML (если есть)
        current_base_url = settings.docs_base_urls.get(source_type, "")
        log.info(f"📂 Сканирование {source_type} (URL: {current_base_url or 'Локально'})")

        # Определяем функцию-парсер в зависимости от расширения
        if source_type.endswith('_docx'):
            files = list(p.rglob("*.docx"))
            processor = lambda f: process_docx_file(
                file_path=f,
                source_type=source_type,
                images_out_dir=settings.images_out_dir,
                chunk_size=settings.max_chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
        else:
            files = list(p.rglob("*.html")) + list(p.rglob("*.htm"))
            processor = lambda f: process_html_file(
                file_path=f,
                source_type=source_type,
                base_url=current_base_url,
                chunk_size=settings.max_chunk_size,
                chunk_overlap=settings.chunk_overlap
            )

        # 4. Фильтрация файлов (Умное обновление)
        files_to_process = []
        for f in files:
            if f.name.startswith('~$'): continue # Пропускаем временные файлы Word
            
            current_hash = get_file_hash(f)
            # Если файла нет в реестре или его хэш изменился — берем в работу
            if hash_registry.get(f.name) != current_hash:
                files_to_process.append((f, current_hash))

        if not files_to_process:
            log.info(f"✅ В папке {source_type} нет новых или измененных файлов.")
            continue

        log.info(f"🔄 Требуется обработать файлов: {len(files_to_process)}")

        # 5. Парсинг и векторизация
        for file_path, new_hash in tqdm(files_to_process, desc=f"Обработка {source_type}"):
            # Сначала удаляем старые чанки этого файла (если это обновление документа)
            try:
                vectorstore.delete(where={"source_file": file_path.name})
            except ValueError:
                pass # Файла еще не было в базе

            # Парсим файл (тут вызывается код из parsers.py)
            new_docs = processor(file_path)

            if new_docs:
                chunk_buffer.extend(new_docs)
                total_processed += 1

            # Пакетная запись (чтобы не перегружать оперативную память)
            if len(chunk_buffer) >= WRITE_BATCH_SIZE:
                vectorstore.add_documents(chunk_buffer)
                chunk_buffer = []
                # Очистка видеопамяти, если используем видеокарту
                if settings.device == 'cuda':
                    torch.cuda.empty_cache()
                    gc.collect()

            # Обновляем кэш хэшей только после успешного парсинга
            hash_registry[file_path.name] = new_hash

    # Дописываем остатки из буфера
    if chunk_buffer:
        vectorstore.add_documents(chunk_buffer)

    # Сохраняем обновленный реестр хэшей на диск
    save_hash_registry(hash_registry)

    log.info(f"🎉 ИНГЕСТ ЗАВЕРШЕН! Успешно обработано файлов: {total_processed}")

if __name__ == "__main__":
    main()