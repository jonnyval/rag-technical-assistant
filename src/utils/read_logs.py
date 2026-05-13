import json
from pathlib import Path

# 1. Указываем путь к файлу с логами (согласно структуре проекта)
log_file_path = Path("data/logs/query_audit.jsonl")

def read_audit_logs(file_path):
    """Функция для чтения и красивого вывода логов формата JSONL."""
    
    # Проверяем, существует ли файл по указанному пути
    if not file_path.exists():
        print(f"Файл {file_path} не найден. Убедись, что система уже записала логи.")
        return

    print(f"Чтение логов из файла: {file_path}\n" + "="*50)

    # 2. Открываем файл для чтения (с кодировкой utf-8 для поддержки русского языка)
    with open(file_path, "r", encoding="utf-8") as file:
        
        # 3. Читаем файл построчно, автоматически считая номера строк
        for line_number, line in enumerate(file, start=1):
            line = line.strip()  # Убираем невидимые пробелы и переносы по краям
            
            if not line:
                continue  # Пропускаем пустые строки, если они случайно появились
            
            try:
                # 4. Преобразуем строку формата JSON в словарь Python
                record = json.loads(line)
                
                # 5. Извлекаем нужные данные с помощью безопасного метода .get()
                timestamp = record.get("ts", "Неизвестное время")
                query = record.get("query", "Нет вопроса")
                elapsed_time = record.get("elapsed_sec", 0.0)
                answer = record.get("final_answer", "Нет ответа")
                n_docs = record.get("n_docs_retrieved", 0)
                
                # 6. Выводим информацию на экран
                print(f"Запись #{line_number} | Время: {timestamp}")
                print(f"Вопрос пользователя: {query}")
                print(f"Найдено документов в БД: {n_docs}")
                print(f"Время генерации: {elapsed_time} сек.")
                print(f"Итоговый ответ: {answer[:150]}...") # Показываем только начало ответа
                print("-" * 50)
                
            except json.JSONDecodeError as e:
                # Если строка повреждена, скрипт не упадет, а просто сообщит об ошибке
                print(f"Ошибка чтения JSON в строке #{line_number}: {e}")

# Запускаем функцию, если скрипт выполняется напрямую
if __name__ == "__main__":
    read_audit_logs(log_file_path)