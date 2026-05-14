import sqlite3
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import settings

def main():
    """Печатает таблицы SQLite-хранилища parent-документов для быстрой проверки."""

    db_path = settings.parent_store_path
    print(f"Путь к БД: {db_path}")

    if not Path(db_path).exists():
        print("Файл базы данных не найден.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()

    print("Найденные таблицы:", [t[0] for t in tables])

if __name__ == "__main__":
    main()
