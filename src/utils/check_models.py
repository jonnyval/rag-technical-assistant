import os
import requests
import warnings
import google.generativeai as genai
from gigachat import GigaChat
from dotenv import load_dotenv

# Подавляем лишние системные предупреждения (например, об отсутствии SSL-сертификатов Минцифры)
warnings.filterwarnings("ignore")
load_dotenv()

def check_groq_models():
    """Проверяет доступность моделей Groq через первый ключ из GROQ_API_KEYS."""

    print("=== Доступные модели GROQ ===")
    keys_str = os.getenv("GROQ_API_KEYS", "")
    keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    
    if not keys:
        print("❌ Ключи GROQ_API_KEYS не найдены в .env\n")
        return

    api_key = keys[0] 
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        response = requests.get("https://api.groq.com/openai/v1/models", headers=headers)
        if response.status_code == 200:
            models = response.json().get("data", [])
            for model in sorted(models, key=lambda x: x["id"]):
                print(f" - {model['id']}")
        else:
            print(f"❌ Ошибка Groq: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Ошибка соединения: {e}")
    print("\n")


def check_gemini_models():
    """Проверяет доступность моделей Gemini через GOOGLE_API_KEYS."""

    print("=== Доступные модели GEMINI ===")
    keys_str = os.getenv("GOOGLE_API_KEYS", "")
    keys = [k.strip() for k in keys_str.split(",") if k.strip()]

    if not keys:
        print("❌ Ключ GOOGLE_API_KEYS не найден в .env\n")
        return

    api_key = keys[0]
    genai.configure(api_key=api_key)

    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                model_name = m.name.replace("models/", "")
                print(f" - {model_name}")
    except Exception as e:
        print(f"❌ Ошибка Gemini: {e}")
    print("\n")


def check_gigachat_models():
    """Проверяет подключение к GigaChat и выводит доступные модели."""

    print("=== Доступные модели GIGACHAT ===")
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    
    if not credentials:
        print("❌ Токен GIGACHAT_CREDENTIALS не найден в .env\n")
        return

    try:
        # verify_ssl_certs=False обязательно, если вы не устанавливали сертификаты Минцифры
        with GigaChat(credentials=credentials, verify_ssl_certs=False) as giga:
            models = giga.get_models()
            for m in models.data:
                # Безопасное извлечение: ищем .model, если нет - ищем .id, если нет - .name
                model_name = getattr(m, 'model', getattr(m, 'id', getattr(m, 'name', str(m))))
                print(f" - {model_name}")
    except Exception as e:
        print(f"❌ Ошибка GigaChat: {e}")
    print("\n")

if __name__ == "__main__":
    check_groq_models()
    check_gemini_models()
    check_gigachat_models()
