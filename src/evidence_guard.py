import re
from typing import Any, List

from src.context_formatting import SourceReference
from src.module_detection import unique_texts


def strict_evidence_context(query: str) -> str:
    query_norm = query.lower().replace("ё", "е")
    definition_patterns = [
        "что означает",
        "что значит",
        "что это",
        "расшифр",
        "обозначает",
        "зачем буква",
        "что за буква",
        "суффикс",
        "маркировк",
        "в названии",
        "в обозначении",
    ]
    if any(pattern in query_norm for pattern in definition_patterns):
        return (
            "Включен строгий режим определения/расшифровки. "
            "Отвечай значением только при наличии прямого определения в тексте источника. "
            "Если прямого определения нет, прямо напиши, что в найденных источниках оно не найдено, "
            "и не делай вывод по косвенным признакам."
        )
    return "Обычный режим: все технические утверждения всё равно должны иметь прямую опору в помеченных фрагментах источников."


def is_definition_query(query: str) -> bool:
    query_norm = query.lower().replace("ё", "е")
    return any(
        pattern in query_norm
        for pattern in [
            "что означает",
            "что значит",
            "что это",
            "расшифр",
            "обозначает",
            "зачем буква",
            "что за буква",
            "суффикс",
            "маркировк",
            "в названии",
            "в обозначении",
        ]
    )


def has_direct_definition_evidence(query: str, docs_context: str) -> bool:
    if not is_definition_query(query):
        return True

    query_norm = query.lower().replace("ё", "е")
    context_norm = docs_context.lower().replace("ё", "е")
    target_terms: List[str] = []

    letter_match = re.search(r"\b(?:буква|суффикс|маркировк\w*|обозначени\w*)\s+([a-zа-я0-9])\b", query_norm)
    if letter_match:
        target_terms.append(letter_match.group(1))
    quoted_match = re.search(r"[\"'«„]([^\"'»“]{1,20})[\"'»“]", query_norm)
    if quoted_match:
        target_terms.append(quoted_match.group(1).strip())
    if " w " in f" {query_norm} ":
        target_terms.append("w")
    for token in re.findall(r"\b[a-zа-я]{1,4}\b", query_norm, flags=re.IGNORECASE):
        if token not in {
            "что",
            "это",
            "как",
            "для",
            "при",
            "или",
            "его",
            "её",
            "она",
            "оно",
            "мод",
            "cu",
        }:
            target_terms.append(token)

    target_terms = unique_texts(target_terms)
    if not target_terms:
        return False

    definition_verbs = [
        "означает",
        "обозначает",
        "расшифровывается",
        "расшифровка",
        "указывает на",
        "предназначен для",
        "является обозначением",
    ]
    for term in target_terms:
        escaped = re.escape(term)
        verbs = "|".join(map(re.escape, definition_verbs))
        patterns = [
            rf"\b{escaped}\b\s*(?:[-–—:]\s*)?(?:{verbs})",
            rf"(?:{verbs})\s+(?:букв[ауы]\s+)?\b{escaped}\b",
            rf"\({escaped}\)\s*(?:[-–—:]\s*)?(?:{verbs})",
        ]
        if any(re.search(pattern, context_norm) for pattern in patterns):
            return True
    return False


def apply_definition_guard(
    response: Any,
    query: str,
    docs_context: str,
    doc_sources: List[SourceReference],
) -> Any:
    if not is_definition_query(query) or has_direct_definition_evidence(query, docs_context):
        return response

    sources_text = ""
    if doc_sources:
        source_ids = ", ".join(f"[{source.source_id}]" for source in doc_sources[:3])
        sources_text = f" В просмотренных фрагментах {source_ids} встречаются связанные упоминания, но они не дают расшифровку."

    safe_answer = (
        "В найденных источниках прямого определения не найдено."
        + sources_text
        + " Поэтому нельзя утверждать, что буква или суффикс в маркировке что-то конкретно означает. "
        "Любая расшифровка без отдельного источника должна считаться гипотезой и не использоваться как подтвержденный факт."
    )
    response.docs_answer = safe_answer
    response.draft_private_comment = (
        "**Что известно из обращения**\n"
        "Пользователь просит расшифровать обозначение в названии модуля.\n\n"
        "**Что подтверждено источниками**\n"
        f"{safe_answer}\n\n"
        "**Гипотезы и ограничения**\n"
        "Гипотезы о значении обозначения намеренно не приводятся, потому что в найденных фрагментах нет прямого определения.\n\n"
        "**Что проверить дальше**\n"
        "Нужен источник, где явно описаны правила маркировки или расшифровка суффиксов модулей CU."
    )
    response.evidence_notes = []
    response.recommended_questions = [
        "Запросить документ или раздел, где явно описаны правила маркировки модулей CU.",
        "Уточнить полное обозначение модуля с артикулом.",
    ]
    response.internal_notes = [
        "Не использовать косвенные совпадения в документации как расшифровку суффикса.",
    ]
    response.confidence = "low"
    return response
