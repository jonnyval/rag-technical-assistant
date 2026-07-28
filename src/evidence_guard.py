import re
from typing import Any, List

from src.context_formatting import SourceReference
from src.logger import log
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


def check_and_format_equipment_mismatch_warning(query: str, doc_sources: List[SourceReference]) -> str:
    """Глобальная проверка несовпадения запрошенной серии оборудования и найденных источников.

    Универсально работает для любых серий и моделей (R500, R500S, R400, R200, AstraRegul и т.д.).
    """
    if not query or not doc_sources:
        return ""

    query_series = set(re.findall(r"\b(R\d{3}S?|REGUL\s*\d*|AstraRegul|ST\d{3}|CU\d{3})\b", query, flags=re.IGNORECASE))
    query_series_compact = {s.upper().replace(" ", "") for s in query_series}

    if not query_series_compact:
        return ""

    mismatches = []
    for source in doc_sources:
        source_text = f"{source.title} {source.source_file} {source.equipment_type} {source.breadcrumb}"
        source_series_found = set(re.findall(
            r"\b(R\d{3}S?|REGUL\s*\d*|AstraRegul|ST\d{3}|CU\d{3})\b",
            source_text,
            flags=re.IGNORECASE,
        ))
        source_series_compact = {series.upper().replace(" ", "") for series in source_series_found}
        # Exact series comparison is important: R500 and R500S are different models.
        matched = bool(query_series_compact & source_series_compact)
        if not matched:
            source_series_str = ", ".join(sorted(source_series_found)) if source_series_found else source.title
            mismatches.append(f"[{source.source_id}] (на самом деле относится к '{source_series_str}')")

    if not mismatches:
        return ""

    user_series_str = ", ".join(sorted(query_series_compact))
    log.warning(
        "EvidenceGuard: equipment mismatch. Requested: %s | Mismatched sources: %s",
        user_series_str,
        ", ".join(mismatches),
    )
    warning = (
        "\n⚠️ ПРОВЕРКА СООТВЕТСТВИЯ ОБОРУДОВАНИЯ:\n"
        f"- Запрошенная серия: {user_series_str}\n"
        f"- Источники с другой серией: {', '.join(mismatches)}\n"
        "Не переноси процедуру, параметры или выводы из этих источников на запрошенную серию "
        "без явного подтверждения совместимости в самом источнике.\n"
    )
    if len(mismatches) == len(doc_sources):
        warning += (
            "Прямой инструкции для запрошенного оборудования в найденных источниках нет; "
            "укажи это в ответе одной короткой фразой.\n"
        )
    return warning


def apply_entity_citation_guard(answer: str, query: str, doc_sources: List[SourceReference]) -> str:
    """Глобальный постобработчик ответа против подмены названий серий под метки источников."""
    if not answer or not doc_sources or not query:
        return answer

    query_series = set(re.findall(r"\b(R\d{3}S?)\b", query, flags=re.IGNORECASE))
    query_series_upper = {s.upper() for s in query_series}

    if not query_series_upper:
        return answer

    sources_text = " ".join([f"{s.title} {s.source_file} {s.breadcrumb}" for s in doc_sources]).upper()
    has_direct_source_match = any(series in sources_text for series in query_series_upper)

    if not has_direct_source_match:
        for series in query_series_upper:
            pattern = re.compile(rf"({re.escape(series)}[^\n.!?]*?\[D\d+\]|\[D\d+\][^\n.!?]*?{re.escape(series)})", re.IGNORECASE)
            if pattern.search(answer):
                log.warning("EvidenceGuard: перехвачена ложная привязка серии %s к метке источника", series)
                disclaimer = (
                    f"> ⚠️ **Обратите внимание:** В найденной документации прямого руководства для **{series}** не обнаружено. "
                    f"Приведенный ниже порядок действий описан в источниках для смежных моделей.\n\n"
                )
                if not answer.startswith("> ⚠️"):
                    answer = disclaimer + answer
                break

    return answer



def _source_ids_used_in_response(response: Any) -> set[str]:
    """Return only source labels explicitly referenced by the generated response."""
    cited: set[str] = set()
    for field_name in ("docs_answer", "draft_private_comment", "final_answer"):
        text = str(getattr(response, field_name, "") or "")
        cited.update(re.findall(r"\[([DT]\d+)\]", text, flags=re.IGNORECASE))
    for note in getattr(response, "evidence_notes", None) or []:
        cited.update(str(item).upper() for item in (getattr(note, "source_ids", None) or []))
    return {source_id.upper() for source_id in cited}


def _remove_unknown_source_labels(response: Any, allowed_ids: set[str]) -> None:
    """Remove fabricated [D..]/[T..] labels rather than presenting them as evidence."""
    def replace_label(match: re.Match) -> str:
        source_id = match.group(1).upper()
        if source_id in allowed_ids:
            return match.group(0)
        log.warning("EvidenceGuard: removed unavailable source label [%s]", source_id)
        return ""

    for field_name in ("docs_answer", "draft_private_comment", "final_answer"):
        if not hasattr(response, field_name):
            continue
        text = str(getattr(response, field_name, "") or "")
        setattr(response, field_name, re.sub(r"\[([DT]\d+)\]", replace_label, text, flags=re.IGNORECASE))


def apply_response_provenance(
    response: Any,
    doc_sources: List[SourceReference],
    ticket_sources: List[SourceReference],
) -> tuple[Any, List[SourceReference], List[SourceReference]]:
    """Validate LLM-provided references and expose only sources actually cited.

    Prompt instructions are advisory. This guard ensures source IDs and ticket IDs in
    structured output originate from the retrieval result before API/UI rendering.
    """
    allowed_docs = {source.source_id.upper(): source for source in doc_sources}
    allowed_tickets = {source.source_id.upper(): source for source in ticket_sources}
    allowed_ids = set(allowed_docs) | set(allowed_tickets)
    _remove_unknown_source_labels(response, allowed_ids)

    valid_notes = []
    for note in getattr(response, "evidence_notes", None) or []:
        source_ids = [str(source_id).upper() for source_id in (getattr(note, "source_ids", None) or [])]
        if not source_ids or any(source_id not in allowed_ids for source_id in source_ids):
            log.warning("EvidenceGuard: dropped evidence note with unavailable source IDs: %s", source_ids)
            continue
        note.source_ids = source_ids
        valid_notes.append(note)
    if hasattr(response, "evidence_notes"):
        response.evidence_notes = valid_notes

    allowed_ticket_keys = {
        str(value).strip()
        for source in ticket_sources
        for value in (source.title, source.source_file)
        if str(value).strip()
    }
    valid_tickets = []
    for ticket in getattr(response, "similar_tickets", None) or []:
        ticket_keys = {str(getattr(ticket, "ticket_id", "")).strip(), str(getattr(ticket, "source_file", "")).strip()}
        if ticket_keys & allowed_ticket_keys:
            valid_tickets.append(ticket)
        else:
            log.warning("EvidenceGuard: dropped ticket absent from retrieval: %s", ticket_keys)
    if hasattr(response, "similar_tickets"):
        response.similar_tickets = valid_tickets

    cited_ids = _source_ids_used_in_response(response)
    for ticket in valid_tickets:
        for source in ticket_sources:
            if str(getattr(ticket, "ticket_id", "")).strip() in {source.title, source.source_file}:
                cited_ids.add(source.source_id.upper())

    used_docs = [source for source in doc_sources if source.source_id.upper() in cited_ids]
    used_tickets = [source for source in ticket_sources if source.source_id.upper() in cited_ids]
    return response, used_docs, used_tickets
