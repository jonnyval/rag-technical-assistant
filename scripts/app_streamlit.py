# scripts/app_qdrant.py

import sys
import os
import warnings
from pathlib import Path

# Добавляем корень проекта для корректных импортов
sys.path.append(str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import streamlit as st
from src.config import settings
from src.engine import RAGEngine
from src.logger import log

# ==========================================
# 🚀 ИНИЦИАЛИЗАЦИЯ ЯДРА (Кэшируется Streamlit)
# ==========================================
@st.cache_resource
def get_engine():
    """Поднимает ядро RAG один раз при старте приложения"""
    try:
        return RAGEngine()
    except Exception as e:
        st.error(f"❌ Критическая ошибка при запуске ядра: {e}")
        log.error(f"RAGEngine init failed: {e}")
        return None

# ==========================================
# 🖥 ИНТЕРФЕЙС STREAMLIT
# ==========================================
def main():
    """Запускает Streamlit-интерфейс для ручного общения с RAG-системой."""

    st.set_page_config(page_title="RegLab AI", layout="wide")
    st.title("🤖 База знаний РегЛаб (Qdrant Production)")
    
    # Инфо-панель на основе настроек из нового config.py
    st.markdown(
        f"**Окружение:** `{settings.environment}` | "
        f"**База данных:** `{settings.active_db_name}` (`{settings.collection_name}`) | "
        f"**LLM:** `{settings.active_llm.upper()}` (`{settings.llm_model_name}`)"
    )
    st.markdown(
        f"**Эмбеддинги:** `{settings.embedding_model_name}` | "
        f"**Реранкер:** `{settings.reranker_model_name}`"
    )
    st.divider()
    
    # Инициализируем движок
    engine = get_engine()
    if not engine:
        st.stop()
    
    # Блок фильтров (безопасная проверка атрибута)
    selected_equipment = []
    if getattr(settings, "show_manual_filter", True): # По умолчанию True, если не задано
        st.markdown("**Фильтр поиска:**")
        col1, _ = st.columns([3, 1])
        with col1:
            selected_equipment = st.multiselect(
                "Принудительный фильтр по оборудованию (оставьте пустым для поиска везде):",
                options=["AstraRegul", "R500", "R500S", "R050", "R150", "R400"],
                default=[],
                help="Выберите один или несколько типов оборудования."
            )
        st.divider()

    # Инициализация истории чата
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Отображение истории
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Обработка нового сообщения
    if prompt := st.chat_input("Введите ваш технический вопрос..."):
        # Добавляем вопрос пользователя в историю
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Анализирую документацию и формирую ответ..."):
                try:
                    # 1. Вызов ядра для обработки запроса
                    # Передаем фильтры напрямую в метод процесса
                    response = engine.process_query(
                        query=prompt, 
                        equipment_filter=selected_equipment if selected_equipment else None
                    )
                    
                    if response is None:
                        st.error("⚠️ Ошибка: Модель не смогла сформировать структурированный ответ.")
                    else:
                        # Основной ответ
                        st.markdown(response.final_answer)

                        # Отрисовка картинок (если функционал поддерживается ядром)
                        if hasattr(response, 'relevant_images') and response.relevant_images:
                            for img_path in response.relevant_images:
                                if os.path.exists(img_path):
                                    st.image(img_path, caption="Иллюстрация из документации")

                        # Экспандер "Мышление" (SGR Audit)
                        with st.expander("🧠 Мышление системы (SGR Audit)"):
                            st.write(f"**Понятый интент:** {response.user_intent}")
                            st.write("**Извлеченные технические факты:**")
                            for f in response.extracted_facts:
                                st.markdown(f"- **{f.source_file}**: {f.fact}")
                            if response.missing_context:
                                st.warning(f"**Чего не хватило:** {response.missing_context}")

                        # Экспандер "Источники" (прямое обращение к ретриверу для прозрачности)
                        with st.expander("📚 Первичные документы (Sources)"):
                            docs = engine._last_docs
                            if not docs:
                                st.write("Документы не найдены.")
                            for i, d in enumerate(docs):
                                fname  = d.metadata.get('source_file', 'Неизвестный файл')
                                score  = d.metadata.get('rerank_score', 0)
                                source = d.metadata.get('db_source', 'unknown')
                                badge  = "📄 Документация" if source == "docs" else "🎫 Тикет"
                                st.markdown(f"{i+1}. {badge} **{fname}** (score: {score:.2f})")
                                with st.container():
                                    st.caption(d.page_content[:300] + "...")

                        # Сохраняем ответ ассистента в историю
                        st.session_state.messages.append({"role": "assistant", "content": response.final_answer})
                
                except Exception as e:
                    st.error(f"Произошла ошибка при обработке: {e}")
                    log.error(f"UI Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
