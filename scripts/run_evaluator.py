import sys
import json
import csv
import time
import numpy as np
import re
from pathlib import Path
import warnings
from datetime import datetime

# Добавляем корень проекта для корректных импортов
sys.path.append(str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.config import settings
from src.logger import log

# ==========================================
# 🧩 СХЕМЫ И МАТЕМАТИКА
# ==========================================
class JudgeSchema(BaseModel):
    """Структура оценки ответа: числовой балл и краткий комментарий судьи."""

    """Схема вердикта технического судьи."""
    is_correct: bool = Field(description="Смысл ответа совпадает с эталоном? true/false.")
    reasoning: str = Field(description="Краткое обоснование решения (1-2 предложения).")

def cosine_similarity(vec1: list, vec2: list) -> float:
    """Считает косинусную близость двух embedding-векторов."""

    """Вычисляет косинусное расстояние между двумя векторами."""
    a, b = np.array(vec1), np.array(vec2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a > 0 and norm_b > 0:
        return np.dot(a, b) / (norm_a * norm_b)
    return 0.0

# ==========================================
# 🚀 УМНЫЙ ОЦЕНЩИК (COSINE + LLM JUDGE)
# ==========================================
class UnifiedEvaluator:
    """Оценивает ответы RAG по семантической близости, LLM-судье и точности выбора варианта."""

    def __init__(self, threshold: float = 0.82):
        """Настраивает порог точности и пути к входным/выходным данным оценщика."""

        self.db_name = settings.active_db_name
        self.llm_name = settings.active_llm
        self.threshold = threshold
        
        # Откуда берем результаты
        self.results_dir = Path("result_test_auto") / self.db_name
        
        # === 🌟 ГЕНЕРИРУЕМ УНИКАЛЬНЫЙ ТАЙМСТАМП ===
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Куда сохраняем оцененные файлы и отчеты (добавляем таймстамп в название)
        self.report_path = Path("reports") / f"{self.db_name}_evaluation_{self.timestamp}"
        self.report_path.mkdir(parents=True, exist_ok=True)
        # ==========================================
        
        self.emb_model = None
        self.judge_chain = None

    def _init_models(self):
        """Загружает embedding-модель и LLM-судью, если они доступны в текущей конфигурации."""

        log.info(f"📥 Загрузка модели эмбеддингов: {settings.embedding_model_name}...")
        self.emb_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={'device': settings.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        log.info(f"🤖 Подключение LLM-судьи (Используется: {self.llm_name.upper()})...")
        
        # --- ДИНАМИЧЕСКИЙ ВЫБОР LLM-СУДЬИ (из config.yaml) ---
        if self.llm_name == "gigachat":
            from langchain_gigachat import GigaChat
            base_judge = GigaChat(
                credentials=settings.gigachat_credentials, 
                verify_ssl_certs=False, 
                model="GigaChat-2", 
                temperature=0.0         
            )
            judge_llm = base_judge.with_structured_output(JudgeSchema)
            
        elif self.llm_name == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            base_judge = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                google_api_key=settings.google_api_key, 
                temperature=0.0
            )
            judge_llm = base_judge.with_structured_output(JudgeSchema)

       # === 🌟 ИСПОЛЬЗУЕМ ВАШ КОНФИГ ДЛЯ OLLAMA ===
        elif self.llm_name == "ollama":
            from langchain_ollama import ChatOllama
            
            ollama_config = getattr(settings, "ollama", {})
            model_name = ollama_config.get("metadata_model", "qwen3:8b")
            
            raw_url = ollama_config.get("base_url", "http://localhost:11434")
            base_url = raw_url.replace("/v1", "")
            
            log.info(f"Подключение к локальной Ollama: {model_name} по адресу {base_url}")
            
            base_judge = ChatOllama(
                model=model_name,                
                temperature=0.0,                    
                base_url=base_url,                  
                format="json"                       
            )
            judge_llm = base_judge.with_structured_output(JudgeSchema)
            
        else: # По умолчанию GROQ
            from langchain_openai import ChatOpenAI
            keys = settings.groq_api_keys
            if not keys:
                raise ValueError("Ключи GROQ не найдены в конфигурации!")
                
            base_judge = ChatOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=keys[0],
                model="llama-3.3-70b-versatile",
                temperature=0.0,
                max_retries=2
            )
            judge_llm = base_judge.with_structured_output(JudgeSchema, method="function_calling")

        # Промпт для Судьи
        prompt = ChatPromptTemplate.from_template("""
        Ты беспристрастный технический судья. Твоя задача — сравнить ответ RAG-системы с эталоном.
        
        ПРАВИЛА:
        1. Сравнивай СМЫСЛ, а не точное совпадение слов.
        2. Если в эталоне указан конкретный пункт (например, "а)"), а RAG выдал этот же пункт или его текстовое содержимое — это правильный ответ.
        3. Если RAG-система дала более развернутый ответ, но он включает в себя суть эталона и не противоречит ему — это правильный ответ.
        
        Вопрос пользователя: {question}
        Эталонный ответ: {reference}
        Ответ RAG-системы: {generated}
        """)
        
        self.judge_chain = prompt | judge_llm

    def extract_option(self, text: str):
        """Извлекает выбранный вариант ответа из текста модели."""

        """Пытается извлечь букву варианта ответа (например, 'а', 'б', 'в', 'г')."""
        match = re.search(r'^([а-яa-z])\)', text.strip().lower())
        return match.group(1) if match else None

    def run(self):
        """Выполняет оценку всех результатов автотеста и собирает статистику качества."""

        if not self.results_dir.exists():
            log.error(f"❌ Папка результатов {self.results_dir} не найдена!")
            return

        self._init_models()
        json_files = list(self.results_dir.glob("*.json"))
        
        if not json_files:
            log.warning(f"⚠️ В папке {self.results_dir} нет JSON файлов для проверки.")
            return

        full_results = []
        incorrect_results = [] 
        stats_by_file = {}

        for file_path in json_files:
            log.info(f"▶️ Оценка файла: {file_path.name}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            file_stats = {"total": 0, "correct": 0, "incorrect": 0, "api_errors": 0}

            for i, item in enumerate(data):
                rag_ans = str(item.get("Ответ RAG", "")).strip()
                ref_ans = str(item.get("Правильный ответ", item.get("Эталон", ""))).strip()
                
                if not rag_ans or not ref_ans:
                    continue
                
                # 1. Проверка на ошибки API при генерации RAG
                is_error = any(kw in rag_ans for kw in ["ОШИБКА:", "Error code:", "Rate limit"])
                if is_error:
                    item["Правильность"] = "Ошибка"
                    item["Судья_Инфо"] = "Ошибка генерации API"
                    item["Оценка_Сходства"] = 0.0
                    file_stats["api_errors"] += 1
                    
                    # Сохраняем в общий и в "ошибочный" списки
                    result_item = {**item, "source_json": file_path.name}
                    full_results.append(result_item)
                    incorrect_results.append(result_item)
                    continue

                file_stats["total"] += 1
                
                # 2. Быстрый фильтр вариантов (а, б, в, г)
                opt_rag = self.extract_option(rag_ans)
                opt_ref = self.extract_option(ref_ans)

                if opt_rag and opt_ref and opt_rag != opt_ref:
                    item["Правильность"] = "Нет"
                    item["Оценка_Сходства"] = 0.0
                    item["Судья_Инфо"] = f"Несовпадение вариантов: '{opt_rag}' вместо '{opt_ref}'"
                else:
                    # 3. Векторное сходство
                    v_rag = self.emb_model.embed_query(rag_ans)
                    v_ref = self.emb_model.embed_query(ref_ans)
                    score = cosine_similarity(v_rag, v_ref)
                    item["Оценка_Сходства"] = round(score, 4)
                    
                    if score >= self.threshold:
                        item["Правильность"] = "Да"
                        item["Судья_Инфо"] = f"Успешно по векторам (Сходство: {score:.2f})"
                    else:
                        # 4. Обращение к LLM-судье
                        try:
                            verdict = self.judge_chain.invoke({
                                "question": item.get("Вопрос", "Не указан"),
                                "reference": ref_ans,
                                "generated": rag_ans
                            })
                            item["Правильность"] = "Да" if verdict.is_correct else "Нет"
                            item["Судья_Инфо"] = f"LLM: {verdict.reasoning}"
                            
                            if self.llm_name not in ["gemini", "ollama"]:
                                time.sleep(1)
                                
                        except Exception as e:
                            item["Правильность"] = "Ошибка Оценки"
                            item["Судья_Инфо"] = f"Ошибка LLM-Судьи: {str(e)}"
                            file_stats["api_errors"] += 1

                # Подсчет статистики и распределение по спискам
                result_item = {**item, "source_json": file_path.name}
                
                if item["Правильность"] == "Да":
                    file_stats["correct"] += 1
                elif item["Правильность"] in ["Нет", "Ошибка Оценки", "Ошибка"]:
                    if item["Правильность"] == "Нет":
                        file_stats["incorrect"] += 1
                    incorrect_results.append(result_item) 

                full_results.append(result_item)

            # Сохраняем оцененный файл в новую директорию С ТАЙМСТАМПОМ
            new_filename = f"{file_path.stem}_{self.timestamp}{file_path.suffix}"
            evaluated_file_path = self.report_path / new_filename
            with open(evaluated_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            stats_by_file[new_filename] = file_stats

        self._save_reports(full_results, incorrect_results, stats_by_file)

    def _save_reports(self, full_data, incorrect_data, stats):
        """Сохраняет полный отчет, ошибки и агрегированную статистику оценивания."""

        """Сохраняет полный дамп, ошибки и генерирует CSV-сводку с таймстампами."""
        
        # 1. Сохраняем полный отчет
        json_out = self.report_path / f"full_report_{self.timestamp}.json"
        with open(json_out, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {"db": self.db_name, "llm_judge": self.llm_name, "threshold": self.threshold},
                "results": full_data
            }, f, ensure_ascii=False, indent=4)

        # 2. 🌟 СОХРАНЯЕМ ФАЙЛ С ОШИБКАМИ 🌟
        if incorrect_data:
            incorrect_out = self.report_path / f"incorrect_answers_{self.timestamp}.json"
            with open(incorrect_out, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {"total_errors": len(incorrect_data), "db": self.db_name},
                    "results": incorrect_data
                }, f, ensure_ascii=False, indent=4)
            log.info(f"🚨 Сохранено неверных ответов: {len(incorrect_data)} -> {incorrect_out.name}")

        # 3. CSV Сводка
        csv_out = self.report_path / f"summary_{self.timestamp}.csv"
        with open(csv_out, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(["Файл", "Всего обработано", "Ошибки API", "Верно", "Неверно", "Accuracy (%)"])
            
            print("\n" + "="*90)
            print(f"{'ФАЙЛ':<45} | {'УСПЕХ':<7} | {'API ERR':<7} | {'ВЕРНО':<7} | {'ACCURACY'}")
            print("-" * 90)
            
            total_v, total_e, total_c = 0, 0, 0
            for name, s in stats.items():
                acc = (s['correct'] / s['total'] * 100) if s['total'] > 0 else 0
                writer.writerow([name, s['total'], s['api_errors'], s['correct'], s['incorrect'], f"{acc:.1f}"])
                
                short_name = (name[:42] + '...') if len(name) > 45 else name
                print(f"{short_name:<45} | {s['total']:<7} | {s['api_errors']:<7} | {s['correct']:<7} | {acc:.1f}%")
                
                total_v += s['total']
                total_e += s['api_errors']
                total_c += s['correct']
            
            final_acc = (total_c / total_v * 100) if total_v > 0 else 0
            print("-" * 90)
            print(f"{'ИТОГО':<45} | {total_v:<7} | {total_e:<7} | {total_c:<7} | {final_acc:.1f}%")
            print("="*90)

        log.info(f"✅ Отчеты успешно сохранены в: {self.report_path}")

if __name__ == "__main__":
    evaluator = UnifiedEvaluator(threshold=0.82)
    evaluator.run()
