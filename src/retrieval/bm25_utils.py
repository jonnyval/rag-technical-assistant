import os
import pickle
import nltk
from typing import List, Dict, Tuple, Any

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

# Глобальная переменная для лемматизатора, чтобы не инициализировать его заново для каждого слова
_morph = None

def _get_morph():
    global _morph
    if _morph is None:
        import pymorphy3
        _morph = pymorphy3.MorphAnalyzer()
    return _morph

def preprocess_text_for_bm25(text: str) -> List[str]:
    """Токенизация и лемматизация текста (приведение к начальной форме)."""
    tokens = nltk.word_tokenize(text.lower(), language="russian")
    morph = _get_morph()
    return [morph.parse(t)[0].normal_form for t in tokens if t.isalnum()]

def build_and_save_bm25_index(chroma_collection: Any, save_path: str) -> None:
    """Собирает все документы из Chroma, строит BM25 и сохраняет на диск."""
    if BM25Okapi is None:
        print("⚠️ Библиотека rank_bm25 не установлена. Пропуск сборки лексического индекса.")
        return
        
    print("🔨 Построение индекса BM25 (это может занять время)...")
    
    # Достаем абсолютно все документы из ChromaDB
    docs = chroma_collection.get(include=['documents', 'metadatas'])
    
    if not docs['documents']:
        print("⚠️ Коллекция пуста. BM25 не построен.")
        return

    corpus_tokens = []
    corpus_map = []
    
    for i, doc in enumerate(docs['documents']):
        corpus_tokens.append(preprocess_text_for_bm25(doc))
        corpus_map.append({
            'id': docs['ids'][i],
            'document': doc,
            'metadata': docs['metadatas'][i] if docs['metadatas'] else {}
        })
        
    # Тренируем модель
    bm25_model = BM25Okapi(corpus_tokens)
    
    # Сохраняем на диск
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump((bm25_model, corpus_map), f)
        
    print(f"✅ Индекс BM25 построен для {len(corpus_tokens)} документов и сохранен в {save_path}")

def load_bm25_index(load_path: str) -> Tuple[Any, List[Dict]]:
    """Быстро загружает готовый BM25 индекс из файла (поддерживает старые и новые форматы)."""
    if not os.path.exists(load_path):
        print(f"⚠️ Файл кэша BM25 не найден по пути: {load_path}")
        return None, []
        
    try:
        print(f"📦 Загрузка кэша BM25 из {load_path}...")
        with open(load_path, "rb") as f:
            data = pickle.load(f)
            
        # 1. Достаем сырые данные независимо от того, словарь это или кортеж
        if isinstance(data, dict):
            bm25_model = data.get('bm25_model')
            raw_corpus = data.get('corpus_map', [])
        elif isinstance(data, tuple) and len(data) == 2:
            bm25_model, raw_corpus = data[0], data[1]
        else:
            print("❌ Неизвестный формат кэша BM25.")
            return None, []

        # 2. НОРМАЛИЗАЦИЯ: Конвертируем старые Tuple в новые Dict на лету
        normalized_corpus = []
        for item in raw_corpus:
            if isinstance(item, dict):
                normalized_corpus.append(item)
            elif isinstance(item, tuple):
                # Если в старом кэше формат: (id, текст, метаданные)
                if len(item) >= 3:
                    normalized_corpus.append({'id': item[0], 'document': item[1], 'metadata': item[2]})
                # Если формат: (id, текст)
                elif len(item) == 2:
                    normalized_corpus.append({'id': item[0], 'document': item[1], 'metadata': {}})
                    
        return bm25_model, normalized_corpus
            
    except Exception as e:
        print(f"❌ Ошибка загрузки BM25: {e}")
        return None, []