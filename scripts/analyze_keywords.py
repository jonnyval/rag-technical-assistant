import os
import re
import pickle
from collections import Counter
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import settings

def main():
    parent_store_file = os.path.join(settings.parent_store_path, "parents_store.pkl")
    
    print(f"📦 Загрузка хранилища: {parent_store_file}")
    with open(parent_store_file, 'rb') as f:
        store = pickle.load(f)

    # Группируем тексты по оборудованию
    corpus = {"R500": "", "R500S": "", "AstraRegul": "", "General": ""}
    
    for doc_id, doc in store.items():
        eq_type = doc.metadata.get("equipment_type", "General")
        if eq_type in corpus:
            corpus[eq_type] += " " + doc.page_content.lower()

    # Стоп-слова (предлоги, союзы и общие слова, которые нам не нужны в RegEx)
    stop_words = set(["в", "на", "с", "и", "по", "для", "к", "о", "от", "до", "или", 
                      "как", "при", "это", "что", "за", "из", "то", "если", "не", "а",
                      "так", "же", "все", "мы", "быть", "раздел", "заголовок", "таблица",
                      "рисунок", "значение", "данных", "модуль", "контроллер"])

    print("\n" + "="*50)
    print("📊 ТОП-30 КЛЮЧЕВЫХ СЛОВ ДЛЯ КАЖДОГО КЛАССА")
    print("="*50)

    for eq_type, text in corpus.items():
        if not text:
            continue
            
        # Вытаскиваем только слова (кириллица, латиница, цифры) длиннее 3 символов
        words = re.findall(r'\b[a-zа-я0-9_]{3,}\b', text)
        filtered_words = [w for w in words if w not in stop_words and not w.isdigit()]
        
        counter = Counter(filtered_words)
        top_words = counter.most_common(30)
        
        print(f"\n🟢 {eq_type.upper()} ({len(words)} слов):")
        words_str = ", ".join([f"{w[0]} ({w[1]})" for w in top_words])
        print(words_str)

if __name__ == "__main__":
    main()