import re
from openai import OpenAI
from src.config import settings
from src.logger import log

# Инициализируем клиента Ollama через совместимый с OpenAI API
if settings.enable_smart_metadata:
    try:
        ollama_client = OpenAI(
            base_url=settings.ollama_url,
            api_key='ollama' # Ключ не важен, но требуется библиотекой
        )
    except Exception as e:
        log.error(f"Не удалось инициализировать клиента Ollama: {e}")
        ollama_client = None
else:
    ollama_client = None

def extract_smart_metadata(text: str) -> dict:
    """
    Отправляет текст в локальную модель (DeepSeek-R1) для извлечения 
    потенциальных вопросов и ключевых слов.
    """
    if not settings.enable_smart_metadata or not ollama_client:
        return {}

    # Ограничиваем длину текста для скорости (берем первые 1500 символов чанка)
    text_sample = text[:1500] 

    prompt = f"""
    Проанализируй следующий отрывок технической документации. 
    Твоя задача — извлечь:
    1. 3-5 ключевых терминов или концепций (через запятую).
    2. 1-2 вопроса, на которые пользователь смог бы найти ответ в этом тексте.

    Отвечай СТРОГО по шаблону ниже, без лишних слов.
    КЛЮЧЕВЫЕ СЛОВА: термин1, термин2, термин3
    ВОПРОСЫ: вопрос1 | вопрос2

    Текст документации:
    {text_sample}
    """

    try:
        response = ollama_client.chat.completions.create(
            model=settings.ollama_metadata_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1 # Низкая температура для стабильного формата
        )
        
        raw_output = response.choices[0].message.content
        
        # Регулярное выражение для удаления блока <think>...</think>
        # flags=re.DOTALL позволяет точке (.) захватывать переносы строк
        cleaned_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
        
        # Парсим очищенный ответ
        keywords = ""
        questions = ""
        
        k_match = re.search(r'КЛЮЧЕВЫЕ СЛОВА:\s*(.*)', cleaned_output, re.IGNORECASE)
        if k_match:
            keywords = k_match.group(1).strip()
            
        q_match = re.search(r'ВОПРОСЫ:\s*(.*)', cleaned_output, re.IGNORECASE)
        if q_match:
            questions = q_match.group(1).strip()
            
        return {
            "smart_keywords": keywords,
            "smart_questions": questions
        }

    except Exception as e:
        log.warning(f"Ошибка умного извлечения метаданных Ollama: {e}")
        return {}