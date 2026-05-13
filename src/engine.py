import time
import warnings
from typing import List, Optional

warnings.filterwarnings("ignore")

from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_gigachat import GigaChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnablePassthrough

from src.config import settings
from src.retrieval.dual_retriever import build_dual_retriever
from src.logger import log, log_query_audit

FUNCTION_CALLING_MODELS = {
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
}


# ==========================================
# 🧩 SGR СХЕМЫ (Schema-Guided Reasoning)
# ==========================================

class FactExtraction(BaseModel):
    """Один извлечённый факт из контекста."""
    source_file: str = Field(description="Название файла-источника из предоставленного контекста")
    fact: str = Field(description="Конкретный технический факт или шаг, полезный для ответа")


class RAGReasoningSchema(BaseModel):
    """Структурированный ответ RAG системы."""
    user_intent: str = Field(description="Кратко переформулируйте, что именно хочет узнать пользователь.")
    extracted_facts: List[FactExtraction] = Field(description="Массив полезных фактов. Пусто, если ничего не найдено.")
    missing_context: str = Field(description="Чего не хватает в контексте для полного ответа.")
    final_answer: str = Field(description="ОБЯЗАТЕЛЬНОЕ ПОЛЕ. Итоговый ответ. Формат Markdown.")
    relevant_images: List[str] = Field(
        default=[],
        description="Массив путей к изображениям (извлекай пути из разметки ![alt](путь) в контексте)."
    )


# ==========================================
# 🛠 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

class KeyRotationCallbackHandler(BaseCallbackHandler):
    """Логирует ошибки LLM при ротации API ключей."""
    
    def __init__(self, key_index: int = 0):
        self.key_index = key_index

    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        log.warning(f"⚠️ [РОТАЦИЯ] Ошибка на ключе #{self.key_index}: {error}")


def format_docs(docs: List) -> str:
    """Форматирует документы для передачи в LLM.

    Добавляет заголовок с типом источника: для тикетов — [ТИКЕТ ПОДДЕРЖКИ],
    для документации — [equipment | файл | раздел].
    """
    if not docs:
        log.warning("format_docs: Получен пустой список документов")
        return ""

    formatted = []
    for i, d in enumerate(docs):
        try:
            meta      = d.metadata if hasattr(d, 'metadata') else {}
            content   = d.page_content if hasattr(d, 'page_content') else str(d)
            db_source = meta.get('db_source', 'docs')

            if db_source == 'tickets':
                header = f"[ТИКЕТ ПОДДЕРЖКИ | Файл: {meta.get('source_file', 'Unknown')}]"
            else:
                header = (
                    f"[{meta.get('equipment_type', 'Unknown')} "
                    f"| Файл: {meta.get('source_file', 'Unknown')} "
                    f"| Раздел: {meta.get('breadcrumb_raw', 'No section')}]"
                )

            formatted.append(f"{header}\n{content}")
        except Exception as e:
            log.error(f"Ошибка при форматировании документа {i}: {e}", exc_info=True)
            if hasattr(d, 'page_content'):
                formatted.append(d.page_content)

    return "\n\n".join(formatted)


# ==========================================
# 🚀 ЯДРО RAG-СИСТЕМЫ
# ==========================================

class RAGEngine:
    """Оркестратор RAG системы: поиск по двум БД + LLM + SGR цепочка.

    Attributes:
        retriever:   DualRetriever — параллельный поиск по документации и тикетам
        sgr_chain:   LangChain цепочка для генерации структурированного ответа
        _last_docs:  Документы из последнего вызова (для отображения в Sources без
                     повторного поиска)
    """

    def __init__(self):
        try:
            log.info("🚀 Инициализация RAGEngine...")
            self.retriever, self.sgr_chain = self._build_pipeline()
            self._last_docs: List = []
            log.info("✅ RAGEngine успешно инициализирован.")
        except Exception as e:
            log.error(f"❌ Критическая ошибка инициализации RAGEngine: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize RAGEngine: {str(e)}") from e

    def _build_pipeline(self):
        """Строит полный RAG pipeline.

        Returns:
            Tuple[DualRetriever, Runnable]: ретривер и SGR цепочка
        """
        try:
            # === 1. ЭМБЕДДИНГИ ===
            log.info(f"📥 Загрузка эмбеддингов: {settings.embedding_model_name}...")
            dense_embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model_name,
                model_kwargs={'device': settings.device},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'prompt': 'Instruct: Retrieve relevant technical documentation passage to answer the query.\nQuery: ',
                }
            )
            log.debug("✓ Dense embeddings загружены")

            # === 2. РЕРАНКЕР (опционально) ===
            rerank_model = None
            if settings.use_reranker:
                log.info(f"📊 Загрузка реранкера: {settings.reranker_model_name}...")
                rerank_model = CrossEncoder(
                    settings.reranker_model_name,
                    device=settings.device,
                    trust_remote_code=True
                )
                log.debug("✓ Реранкер загружен")
            else:
                log.info("⏭️  Реранкер отключён (use_reranker: false) — используется порядок векторного скора")

            # === 3. DUAL RETRIEVER (документация + тикеты) ===
            log.info("🔀 Построение DualRetriever...")
            retriever = build_dual_retriever(dense_embeddings, rerank_model)
            log.debug("✓ DualRetriever готов")

            # === 4. LLM С РОТАЦИЕЙ КЛЮЧЕЙ ===
            log.info(f"🤖 Инициализация LLM: {settings.active_llm.upper()} ({settings.llm_model_name})...")

            if settings.active_llm == "gemini":
                log.debug(f"  Ключей Gemini: {len(settings.google_api_keys)}")
                
                if not settings.google_api_keys:
                    raise ValueError("❌ GOOGLE_API_KEYS не установлена в .env!")
                
                llms = [
                    ChatGoogleGenerativeAI(
                        model=settings.llm_model_name,
                        google_api_key=k,
                        temperature=0.2,
                        timeout=settings.llm_timeout,
                        callbacks=[KeyRotationCallbackHandler(key_index=i)],
                    )
                    for i, k in enumerate(settings.google_api_keys)
                ]
                
                if len(llms) > 1:
                    llm = llms[0].with_fallbacks(llms[1:])
                    log.info(f"✅ Ротация Gemini ключей включена ({len(llms)} ключей)")
                else:
                    log.warning("⚠️  Только 1 Gemini ключ — ротация отключена")
                    llm = llms[0]

            elif settings.active_llm == "gigachat":
                log.debug("  Подключение к GigaChat...")
                llm = GigaChat(
                    credentials=settings.gigachat_credentials,
                    verify_ssl_certs=False,
                    model=settings.llm_model_name,
                    temperature=0.2
                )

            elif settings.active_llm == "ollama":
                log.debug(f"  Подключение к Ollama: {settings.ollama_url}...")
                llm = ChatOpenAI(
                    base_url=settings.ollama_url,
                    api_key="ollama",
                    model=settings.llm_model_name,
                    temperature=0.2
                )

            else:  # GROQ
                log.debug(f"  Ключей GROQ: {len(settings.groq_api_keys)}")
                
                if not settings.groq_api_keys:
                    raise ValueError("❌ GROQ_API_KEYS не установлена в .env!")
                
                llms = [
                    ChatOpenAI(
                        base_url="https://api.groq.com/openai/v1",
                        api_key=k,
                        model=settings.llm_model_name,
                        temperature=0.2,
                        timeout=settings.llm_timeout,
                        max_retries=settings.llm_max_retries,
                        callbacks=[KeyRotationCallbackHandler(key_index=i)]
                    )
                    for i, k in enumerate(settings.groq_api_keys)
                ]
                
                if len(llms) > 1:
                    llm = llms[0].with_fallbacks(llms[1:])
                    log.info(f"✅ Ротация GROQ ключей включена ({len(llms)} ключей)")
                else:
                    log.warning("⚠️  Только 1 GROQ ключ — ротация отключена")
                    llm = llms[0]

            log.debug("✓ LLM инициализирована")

            # === 5. ПРОМПТ ===
            log.info("📝 Создание prompt template...")
            prompt_template = ChatPromptTemplate.from_template("""
Ты ведущий технический эксперт компании "РегЛаб". 
Твоя задача — проанализировать контекст и ответить на вопрос. 
Ты ДОЛЖЕН вернуть ответ СТРОГО в формате валидного JSON со следующей структурой:

{{
  "user_intent": "Кратко переформулируй, что именно хочет узнать пользователь",
  "extracted_facts": [
    {{"source_file": "имя файла", "fact": "факт из текста"}}
  ],
  "missing_context": "Чего не хватает в контексте (или 'Всего хватает')",
  "final_answer": "Твой подробный итоговый ответ в формате Markdown",
  "relevant_images": ["массив путей к картинкам, если они есть"]
}}

ГЛОССАРИЙ:
- R500: Стандартные ПЛК (РСУ).
- R500S: Контроллеры безопасности (ПСБ, SIL3).
- AstraRegul: Верхний уровень (HMI, Server, Historian).
- ТИКЕТ ПОДДЕРЖКИ: Реальный случай из практики техподдержки. Используй как пример решения аналогичных проблем.

ПРАВИЛА:
1. Если есть варианты ответов — выбери ВСЕ правильные.
2. ДЕЛАЙ ДОПУЩЕНИЯ: Если контекст говорит о Modbus Serial, а вопрос о Modbus TCP, считай логику статусов одинаковой.
3. Поле final_answer ОБЯЗАТЕЛЬНО. Никогда не оставляй его пустым.

Контекст:
{context}

Вопрос: {input}
""")

            # === 6. STRUCTURED OUTPUT ===
            if settings.active_llm == "gigachat":
                structured_llm = llm.with_structured_output(RAGReasoningSchema)
            elif settings.active_llm == "ollama":
                structured_llm = llm.with_structured_output(RAGReasoningSchema, method="json_mode")
            else:
                method = "function_calling" if settings.llm_model_name in FUNCTION_CALLING_MODELS else "json_mode"
                structured_llm = llm.with_structured_output(RAGReasoningSchema, method=method)

            log.debug(f"✓ Structured LLM готова (метод: {method if settings.active_llm not in ('gigachat', 'ollama') else settings.active_llm})")

            # === 7. СБОРКА ЦЕПОЧКИ ===
            log.info("⛓️  Сборка SGR цепочки...")
            sgr_chain = (
                {"context": retriever | format_docs, "input": RunnablePassthrough()}
                | prompt_template
                | structured_llm
            )
            log.debug("✓ SGR цепочка собрана")

            return retriever, sgr_chain

        except Exception as e:
            log.error(f"❌ Ошибка при построении pipeline: {e}", exc_info=True)
            raise RuntimeError(f"Failed to build RAG pipeline: {str(e)}") from e

    def process_query(
        self,
        query: str,
        equipment_filter: Optional[List[str]] = None,
    ) -> RAGReasoningSchema:
        """Обрабатывает вопрос пользователя и возвращает структурированный ответ.

        Сохраняет найденные документы в self._last_docs — используйте их в UI
        для блока Sources вместо повторного вызова retriever.invoke().

        После каждого запроса (успешного или нет) записывает запись аудита
        в data/logs/query_audit.jsonl.

        Args:
            query:            Вопрос пользователя (непустая строка)
            equipment_filter: Фильтр по типу оборудования для docs-БД.
                              None или [] — поиск по всей документации.

        Returns:
            RAGReasoningSchema со структурированным ответом

        Raises:
            ValueError:    Если query пустой или не строка
            RuntimeError:  Если произошла ошибка при генерации ответа
        """
        # === ВАЛИДАЦИЯ ===
        if not query:
            raise ValueError("query не может быть пустым")
        if not isinstance(query, str):
            raise ValueError(f"query должен быть строкой, получен {type(query).__name__}")
        query = query.strip()
        if not query:
            raise ValueError("query содержит только пробелы")

        # === ФИЛЬТР ===
        if not equipment_filter:
            self.retriever.equipment_filter = ["Все"]
            log.debug("Фильтр оборудования: ВСЕ")
        else:
            if not isinstance(equipment_filter, list):
                raise ValueError(f"equipment_filter должен быть списком, получен {type(equipment_filter).__name__}")
            self.retriever.equipment_filter = equipment_filter
            log.debug(f"Фильтр оборудования: {', '.join(equipment_filter)}")

        # === ОБРАБОТКА ===
        start = time.perf_counter()
        self._last_docs = []

        try:
            log.info(f"🔄 Обработка запроса ({len(query)} символов)")
            log.debug(f"   Запрос: {query[:100]}{'...' if len(query) > 100 else ''}")

            # Поиск документов — сохраняем для UI и передаём в цепочку
            self._last_docs = self.retriever.invoke(query)

            # Генерация ответа (retriever внутри sgr_chain вызовется повторно —
            # это нормально, LangChain chain этого требует)
            response = self.sgr_chain.invoke(query)

            elapsed = time.perf_counter() - start
            log.info(
                f"✅ Ответ сгенерирован за {elapsed:.2f}с "
                f"| документов: {len(self._last_docs)} "
                f"| фактов: {len(response.extracted_facts)}"
            )

            log_query_audit(
                query=query,
                equipment_filter=equipment_filter,
                retrieved_docs=self._last_docs,
                response=response,
                elapsed_sec=elapsed,
            )

            return response

        except Exception as e:
            elapsed = time.perf_counter() - start
            log.error(f"❌ Ошибка при генерации ответа: {e}", exc_info=True)

            log_query_audit(
                query=query,
                equipment_filter=equipment_filter,
                retrieved_docs=self._last_docs,
                response=None,
                elapsed_sec=elapsed,
                extra={"error": str(e)},
            )

            raise RuntimeError(f"Failed to process query: {str(e)}") from e