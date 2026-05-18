"""Оценивает результаты автотеста RAG по эталонным ответам.

Скрипт читает JSON-файлы из `result_test_auto/<active_db>`, сравнивает поле
`Ответ RAG` с `Правильный ответ`, проставляет `Правильность` и сохраняет
отчеты в `reports/<db>_evaluation_<timestamp>`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator

sys.path.append(str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.logger import log


QUESTION_KEYS = ("Вопрос", "question")
REFERENCE_KEYS = ("Правильный ответ", "Эталон", "reference", "expected_answer")
RAG_ANSWER_KEYS = ("Ответ RAG", "rag_answer", "generated")
CORRECTNESS_KEY = "Правильность"
SCORE_KEY = "Оценка_сходства"
JUDGE_INFO_KEY = "Судья_инфо"
ERROR_PREFIXES = ("ОШИБКА:", "ERROR:", "Error code:", "Rate limit", "Failed to")


class JudgeSchema(BaseModel):
    """Структура ответа LLM-судьи."""

    is_correct: bool = Field(default=False, description="Совпадает ли смысл ответа с эталоном")
    reasoning: str = Field(default="", description="Короткое объяснение решения судьи")

    @field_validator("reasoning", mode="before")
    @classmethod
    def normalize_reasoning(cls, value: Any) -> str:
        """Приводит пояснение судьи к строке."""
        return "" if value is None else str(value)


def parse_args() -> argparse.Namespace:
    """Разбирает параметры запуска evaluator."""
    parser = argparse.ArgumentParser(description="Evaluate RAG autotest result JSON files.")
    parser.add_argument(
        "--results-dir",
        default=str(Path("result_test_auto") / settings.active_db_name),
        help="Папка с результатами автотеста.",
    )
    parser.add_argument("--file", default=None, help="Оценить только один JSON-файл из results-dir.")
    parser.add_argument("--threshold", type=float, default=0.82, help="Порог cosine similarity.")
    parser.add_argument("--no-llm", action="store_true", help="Не использовать LLM-судью после embeddings.")
    return parser.parse_args()


def get_first(item: Dict[str, Any], keys: Iterable[str], default: str = "") -> str:
    """Возвращает первое непустое значение по списку возможных ключей."""
    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def normalize_text(text: str) -> str:
    """Нормализует текст для грубого сравнения ответов."""
    text = str(text or "").lower().replace("\t", " ")
    text = text.replace("ё", "е")
    text = re.sub(r"(^|\n)\s*[-•]?\s*[а-яa-z]\)\s*", r"\1", text)
    text = re.sub(r"^[\s\-•]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[«»\"'`.,;:!?()\[\]{}]", "", text)
    return text.strip()


def split_reference_answers(reference: str) -> List[str]:
    """Делит эталон на несколько правильных ответов, если они записаны построчно."""
    parts = []
    for raw_part in str(reference).splitlines():
        part = raw_part.strip(" -\t\r")
        part = re.sub(r"^\s*[-•]?\s*[а-яa-z]\)\s*", "", part, flags=re.IGNORECASE)
        parts.append(part)
    return [part for part in parts if part]


def extract_option_letters(text: str) -> set[str]:
    """Извлекает буквы вариантов ответа вида `а)` или `б)` из начала строк."""
    letters = set()
    for line in str(text or "").splitlines():
        match = re.match(r"^\s*[-•]?\s*([а-яa-z])\)", line.strip().lower())
        if match:
            letters.add(match.group(1).replace("ё", "е"))
    first = re.match(r"^\s*([а-яa-z])\)", str(text or "").strip().lower())
    if first:
        letters.add(first.group(1).replace("ё", "е"))
    return letters


def cosine_similarity(vec1: list, vec2: list) -> float:
    """Считает cosine similarity двух embedding-векторов."""
    a, b = np.array(vec1), np.array(vec2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a > 0 and norm_b > 0:
        return float(np.dot(a, b) / (norm_a * norm_b))
    return 0.0


class UnifiedEvaluator:
    """Оценивает ответы автотеста быстрыми правилами, embeddings и LLM-судьей."""

    def __init__(
        self,
        results_dir: Path,
        *,
        threshold: float = 0.82,
        use_llm_judge: bool = True,
    ):
        """Настраивает пути, порог сходства и режим LLM-судьи."""
        self.db_name = settings.active_db_name
        self.llm_name = settings.active_llm
        self.threshold = threshold
        self.use_llm_judge = use_llm_judge
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_path = Path("reports") / f"{self.db_name}_evaluation_{self.timestamp}"
        self.report_path.mkdir(parents=True, exist_ok=True)
        self.emb_model: Optional[HuggingFaceEmbeddings] = None
        self.judge_chain = None

    def _init_models(self) -> None:
        """Загружает embeddings и, при необходимости, LLM-судью."""
        log.info("Загрузка embedding-модели для evaluator: %s", settings.embedding_model_name)
        self.emb_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": settings.device},
            encode_kwargs={"normalize_embeddings": True},
        )

        if not self.use_llm_judge:
            log.info("LLM-судья отключен параметром --no-llm")
            return

        judge_llm = self._build_judge_llm()
        prompt = ChatPromptTemplate.from_template(
            """Ты беспристрастный технический судья.
Сравни ответ RAG с эталоном по смыслу.
Верни строго валидный JSON по схеме JudgeSchema: {{"is_correct": true/false, "reasoning": "краткая причина"}}.

Правила:
- Если RAG дал тот же вариант ответа или содержит тот же смысл, верни is_correct=true.
- Если эталон содержит несколько правильных пунктов, RAG должен содержать все эти пункты.
- Не требуй дословного совпадения.

Вопрос: {question}
Эталонный ответ: {reference}
Ответ RAG: {generated}
"""
        )
        self.judge_chain = prompt | judge_llm

    def _build_judge_llm(self):
        """Создает structured-output LLM для оценки спорных ответов."""
        log.info("Подключение LLM-судьи: %s", self.llm_name)
        if self.llm_name == "gigachat":
            from langchain_gigachat import GigaChat

            base_judge = GigaChat(
                credentials=settings.gigachat_credentials,
                verify_ssl_certs=False,
                model="GigaChat-2",
                temperature=0.0,
            )
            return base_judge.with_structured_output(JudgeSchema)

        if self.llm_name == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            base_judge = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=settings.google_api_key,
                temperature=0.0,
            )
            return base_judge.with_structured_output(JudgeSchema)

        if self.llm_name == "ollama":
            from langchain_ollama import ChatOllama

            raw_url = getattr(settings, "ollama_url", "http://localhost:11434")
            base_url = raw_url.replace("/v1", "")
            base_judge = ChatOllama(
                model=settings.llm_model_name,
                temperature=0.0,
                base_url=base_url,
                format="json",
            )
            return base_judge.with_structured_output(JudgeSchema)

        from langchain_openai import ChatOpenAI

        keys = settings.groq_api_keys
        if not keys:
            raise ValueError("GROQ API keys не найдены в конфигурации")
        base_judge = ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=keys[0],
            model=settings.llm_model_name,
            temperature=0.0,
            max_retries=1,
        )
        return base_judge.with_structured_output(JudgeSchema, method="json_mode")

    def _quick_rule_verdict(self, reference: str, generated: str) -> Optional[tuple[bool, str]]:
        """Возвращает быстрый вердикт без embeddings и LLM, если совпадение очевидно."""
        ref_norm = normalize_text(reference)
        gen_norm = normalize_text(generated)
        if not ref_norm or not gen_norm:
            return None

        ref_letters = extract_option_letters(reference)
        gen_letters = extract_option_letters(generated)
        if ref_letters and gen_letters:
            return ref_letters == gen_letters, f"Сравнение букв вариантов: RAG={sorted(gen_letters)}, эталон={sorted(ref_letters)}"

        ref_parts = [normalize_text(part) for part in split_reference_answers(reference)]
        ref_parts = [part for part in ref_parts if part]
        if len(ref_parts) > 1:
            missing = [part for part in ref_parts if part not in gen_norm]
            return not missing, "Все эталонные пункты найдены в ответе" if not missing else f"Не найдены пункты: {missing}"

        if ref_norm == gen_norm:
            return True, "Точное совпадение после нормализации"
        if ref_norm in gen_norm:
            return True, "Эталон содержится в развернутом ответе RAG"
        return None

    def _evaluate_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Оценивает одну строку результата и обновляет ее полями оценки."""
        rag_ans = get_first(item, RAG_ANSWER_KEYS)
        ref_ans = get_first(item, REFERENCE_KEYS)
        question = get_first(item, QUESTION_KEYS, "Не указан")

        if not rag_ans or not ref_ans:
            return None

        if any(marker in rag_ans for marker in ERROR_PREFIXES):
            item[CORRECTNESS_KEY] = "Ошибка"
            item[JUDGE_INFO_KEY] = "Ошибка генерации RAG"
            item[SCORE_KEY] = 0.0
            return item

        quick = self._quick_rule_verdict(ref_ans, rag_ans)
        if quick is not None:
            is_correct, reason = quick
            item[CORRECTNESS_KEY] = "Да" if is_correct else "Нет"
            item[JUDGE_INFO_KEY] = reason
            item[SCORE_KEY] = 1.0 if is_correct else 0.0
            return item

        assert self.emb_model is not None
        score = cosine_similarity(
            self.emb_model.embed_query(rag_ans),
            self.emb_model.embed_query(ref_ans),
        )
        item[SCORE_KEY] = round(score, 4)

        if score >= self.threshold:
            item[CORRECTNESS_KEY] = "Да"
            item[JUDGE_INFO_KEY] = f"Высокое embedding-сходство: {score:.2f}"
            return item

        if self.judge_chain is None:
            item[CORRECTNESS_KEY] = "Нет"
            item[JUDGE_INFO_KEY] = f"Embedding-сходство ниже порога: {score:.2f}; LLM-судья отключен"
            return item

        try:
            verdict = self.judge_chain.invoke({
                "question": question,
                "reference": ref_ans,
                "generated": rag_ans,
            })
            item[CORRECTNESS_KEY] = "Да" if verdict.is_correct else "Нет"
            item[JUDGE_INFO_KEY] = f"LLM: {verdict.reasoning}"
            if self.llm_name not in ("gemini", "ollama"):
                time.sleep(1)
        except Exception as exc:
            item[CORRECTNESS_KEY] = "Ошибка оценки"
            item[JUDGE_INFO_KEY] = f"Ошибка LLM-судьи: {exc}"
        return item

    def run(self, only_file: Optional[str] = None) -> None:
        """Запускает оценку всех выбранных JSON-файлов."""
        if not self.results_dir.exists():
            log.error("Папка результатов не найдена: %s", self.results_dir)
            return

        self._init_models()
        json_files = sorted(self.results_dir.glob("*.json"))
        json_files = [
            path for path in json_files
            if path.name != "summary.json" and not path.name.startswith(("full_report_", "incorrect_answers_"))
        ]
        if only_file:
            json_files = [path for path in json_files if path.name == only_file]

        if not json_files:
            log.warning("Нет JSON-файлов для оценки в %s", self.results_dir)
            return

        full_results: List[Dict[str, Any]] = []
        incorrect_results: List[Dict[str, Any]] = []
        stats_by_file: Dict[str, Dict[str, int]] = {}

        for file_path in json_files:
            log.info("Оценка файла: %s", file_path.name)
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                log.warning("Пропускаю %s: верхний уровень не список", file_path.name)
                continue

            file_stats = {"total": 0, "correct": 0, "incorrect": 0, "errors": 0, "skipped": 0}
            for item in data:
                evaluated = self._evaluate_item(item)
                if evaluated is None:
                    file_stats["skipped"] += 1
                    continue

                file_stats["total"] += 1
                status = evaluated.get(CORRECTNESS_KEY, "")
                if status == "Да":
                    file_stats["correct"] += 1
                elif status == "Нет":
                    file_stats["incorrect"] += 1
                    incorrect_results.append({**evaluated, "source_json": file_path.name})
                else:
                    file_stats["errors"] += 1
                    incorrect_results.append({**evaluated, "source_json": file_path.name})
                full_results.append({**evaluated, "source_json": file_path.name})

            evaluated_path = self.report_path / f"{file_path.stem}_{self.timestamp}{file_path.suffix}"
            with evaluated_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            stats_by_file[evaluated_path.name] = file_stats

        self._save_reports(full_results, incorrect_results, stats_by_file)

    def _save_reports(
        self,
        full_data: List[Dict[str, Any]],
        incorrect_data: List[Dict[str, Any]],
        stats: Dict[str, Dict[str, int]],
    ) -> None:
        """Сохраняет полный отчет, ошибки и CSV-сводку."""
        full_report = self.report_path / f"full_report_{self.timestamp}.json"
        full_report.write_text(
            json.dumps({
                "metadata": {
                    "db": self.db_name,
                    "llm_judge": self.llm_name if self.use_llm_judge else "disabled",
                    "threshold": self.threshold,
                },
                "results": full_data,
            }, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )

        if incorrect_data:
            incorrect_report = self.report_path / f"incorrect_answers_{self.timestamp}.json"
            incorrect_report.write_text(
                json.dumps({
                    "metadata": {"total_errors": len(incorrect_data), "db": self.db_name},
                    "results": incorrect_data,
                }, ensure_ascii=False, indent=4),
                encoding="utf-8",
            )

        csv_report = self.report_path / f"summary_{self.timestamp}.csv"
        with csv_report.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow(["Файл", "Всего", "Пропущено", "Ошибки", "Верно", "Неверно", "Accuracy (%)"])
            total, correct = 0, 0
            for name, stat in stats.items():
                accuracy = (stat["correct"] / stat["total"] * 100) if stat["total"] else 0.0
                writer.writerow([
                    name,
                    stat["total"],
                    stat["skipped"],
                    stat["errors"],
                    stat["correct"],
                    stat["incorrect"],
                    f"{accuracy:.1f}",
                ])
                total += stat["total"]
                correct += stat["correct"]

        final_accuracy = (correct / total * 100) if total else 0.0
        log.info("Оценка завершена: всего=%s, верно=%s, accuracy=%.1f%%", total, correct, final_accuracy)
        log.info("Отчеты сохранены в: %s", self.report_path)


def main() -> None:
    """Точка входа CLI."""
    args = parse_args()
    evaluator = UnifiedEvaluator(
        Path(args.results_dir),
        threshold=args.threshold,
        use_llm_judge=not args.no_llm,
    )
    evaluator.run(only_file=args.file)


if __name__ == "__main__":
    main()
