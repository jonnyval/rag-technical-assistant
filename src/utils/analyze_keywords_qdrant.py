import os
import re
import pickle
import sqlite3
from collections import Counter
from pathlib import Path
import sys

# Добавляем корень проекта
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import settings

def main():
    # Путь берется из config.yaml (active: qdrant_v2_docker)
    db_path = settings.parent_store_path 
    
    if not os.path.exists(db_path):
        print(f"❌ Файл базы не найден: {db_path}")
        return

    print(f"📦 Подключение к SQLite хранилищу: {db_path}")
    
    # Группируем тексты по оборудованию
    corpus = {"R500": "", "R500S": "", "AstraRegul": "", "General": ""}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # ВАЖНО: Используем найденное вами имя таблицы
        cursor.execute("SELECT value FROM langchain_key_value_stores")
        rows = cursor.fetchall()
        
        print(f"🔎 Извлечено документов из документации: {len(rows)}")

        for row in rows:
            # Десериализуем объект Document
            doc = pickle.loads(row[0])
            
            eq_type = doc.metadata.get("equipment_type", "General")
            if eq_type in corpus:
                corpus[eq_type] += " " + doc.page_content.lower()
                
        conn.close()

    except Exception as e:
        print(f"❌ Ошибка при чтении SQLite: {e}")
        return

    # Стоп-слова (технический шум)
    stop_words = set(["в", "на", "с", "и", "по", "для", "к", "о", "от", "до", "или", 
                      "как", "при", "это", "что", "за", "из", "то", "если", "не", "а",
                      "так", "же", "все", "мы", "быть", "раздел", "заголовок", "таблица",
                      "рисунок", "значение", "данных", "модуль", "контроллер", "работа",
                      "система", "устройство", "параметр", "настройка"])

    print("\n" + "="*50)
    print(f"📊 ТОП-30 ТЕРМИНОВ ДОКУМЕНТАЦИИ ({settings.collection_name})")
    print("="*50)

    for eq_type, text in corpus.items():
        if not text: continue
            
        # Вытаскиваем слова длиннее 3 символов
        words = re.findall(r'\b[a-zа-я0-9_]{3,}\b', text)
        filtered_words = [w for w in words if w not in stop_words and not w.isdigit()]
        
        counter = Counter(filtered_words)
        top_words = counter.most_common(30)
        
        print(f"\n🟢 {eq_type.upper()} ({len(words)} слов):")
        words_str = ", ".join([f"{w[0]} ({w[1]})" for w in top_words])
        print(words_str)

if __name__ == "__main__":
    main()