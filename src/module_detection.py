import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.logger import log


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_TAXONOMY_PATH = PROJECT_ROOT / "data" / "module_alias_taxonomy.json"
_MODULE_RULES_CACHE: Optional[List[Dict[str, Any]]] = None


def list_text(value: Any) -> List[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def unique_texts(values: List[Any]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        for item in list_text(value):
            key = item.lower()
            if key not in seen:
                seen.add(key)
                result.append(item)
    return result


def normalize_compact_text(value: str) -> str:
    value = str(value or "").lower().replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я]+", "", value)


def _generated_module_aliases(product_family: str, module_code: str) -> List[str]:
    aliases: List[str] = []
    family_short = ""
    family_match = re.search(r"\b(R\d{3}S?)\b", product_family, flags=re.IGNORECASE)
    if family_match:
        family_short = family_match.group(1).upper()

    parts = re.findall(r"[A-Za-z]+|\d+", module_code)
    if len(parts) < 3:
        return aliases

    prefix = parts[0].upper()
    middle = parts[1]
    suffix = parts[2]
    aliases.extend([f"{prefix}{middle}{suffix}", f"{prefix}{middle} {suffix}", f"{prefix} {middle}{suffix}"])
    if family_short:
        aliases.extend([
            f"{family_short} {prefix}{middle}{suffix}",
            f"{family_short} {prefix}{middle} {suffix}",
            f"{family_short} {prefix} {middle}{suffix}",
        ])
    if len(middle) == 2:
        aliases.extend([f"{prefix}0{middle}{suffix}", f"{prefix}0{middle} {suffix}"])
        if family_short:
            aliases.extend([f"{family_short} {prefix}0{middle}{suffix}", f"{family_short} {prefix}0{middle} {suffix}"])
    if middle == "00":
        aliases.extend([f"{prefix} {suffix}", f"{prefix}{suffix}", f"{prefix}-{suffix}", f"{prefix}_{suffix}"])
        if family_short:
            aliases.extend([f"{family_short} {prefix} {suffix}", f"{family_short} {prefix}{suffix}", f"{family_short}-{prefix}-{suffix}"])
    return aliases


def _load_module_rules() -> List[Dict[str, Any]]:
    global _MODULE_RULES_CACHE
    if _MODULE_RULES_CACHE is not None:
        return _MODULE_RULES_CACHE
    if not MODULE_TAXONOMY_PATH.exists():
        log.warning("Module taxonomy not found: %s", MODULE_TAXONOMY_PATH)
        _MODULE_RULES_CACHE = []
        return _MODULE_RULES_CACHE

    data = json.loads(MODULE_TAXONOMY_PATH.read_text(encoding="utf-8-sig"))
    rules: List[Dict[str, Any]] = []
    for item in data.get("entities", []):
        if not isinstance(item, dict):
            continue
        canonical = str(item.get("canonical") or "").strip()
        module_code = str(item.get("module_code") or "").strip()
        product_family = str(item.get("product_family") or "").strip()
        aliases = unique_texts([
            canonical,
            module_code,
            item.get("article_numbers", []),
            item.get("aliases", []),
            item.get("weak_aliases", []),
            _generated_module_aliases(product_family, module_code),
        ])
        alias_keys = sorted(
            {normalize_compact_text(alias) for alias in aliases if len(normalize_compact_text(alias)) >= 4},
            key=len,
            reverse=True,
        )
        if canonical and alias_keys:
            rules.append({
                "canonical": canonical,
                "product_family": product_family,
                "module_code": module_code,
                "function": str(item.get("function") or "").strip(),
                "russian_name": str(item.get("russian_name") or "").strip(),
                "confidence": str(item.get("confidence") or "").strip(),
                "alias_keys": alias_keys,
            })
    _MODULE_RULES_CACHE = rules
    return rules


def detect_modules_in_query(query: str) -> List[Dict[str, Any]]:
    normalized = normalize_compact_text(query)
    if not normalized:
        return []
    matched: List[Dict[str, Any]] = []
    occupied: List[tuple[int, int]] = []
    for module in _load_module_rules():
        match_span: Optional[tuple[int, int]] = None
        for alias_key in module["alias_keys"]:
            start = normalized.find(alias_key)
            if start < 0:
                continue
            end = start + len(alias_key)
            if any(not (end <= old_start or start >= old_end) for old_start, old_end in occupied):
                continue
            match_span = (start, end)
            break
        if match_span is None:
            continue
        occupied.append(match_span)
        matched.append(module)
    return matched


def format_detected_modules(modules: List[Dict[str, Any]]) -> str:
    if not modules:
        return "Модули в вопросе не определены по таксономии алиасов."
    lines = []
    for index, module in enumerate(modules, start=1):
        details = [
            f"Каноническое имя: {module.get('canonical')}",
            f"Код: {module.get('module_code')}",
            f"Семейство: {module.get('product_family')}",
        ]
        if module.get("russian_name"):
            details.append(f"Название: {module['russian_name']}")
        if module.get("function"):
            details.append(f"Функция: {module['function']}")
        lines.append(f"[M{index}] " + " | ".join(detail for detail in details if detail and not detail.endswith(": ")))
    return "\n".join(lines)


def build_module_enriched_query(query: str, modules: List[Dict[str, Any]]) -> str:
    if not modules:
        return query
    module_terms: List[str] = []
    for module in modules:
        module_terms.extend([module.get("canonical"), module.get("module_code"), module.get("product_family")])
    return query.rstrip() + "\n\nНормализованные модули запроса: " + ", ".join(unique_texts(module_terms))


def module_doc_page_title_hints(modules: List[Dict[str, Any]]) -> List[str]:
    titles: List[str] = []
    for module in modules:
        family = str(module.get("product_family") or "")
        code = str(module.get("module_code") or "")
        if family == "Regul R500" and code in {"CU 00 151", "CU 00 161", "CU 00 171", "CU 00 181"}:
            titles.extend([
                "Модули центрального процессора CU 00 151 / CU 00 161 / CU 00 171 / CU 00 181 (III тип)",
                "Модули центрального процессора CU 00\xa0151 / CU 00\xa0161 / CU 00 171 / CU 00 181 (III тип)",
            ])
        elif family == "Regul R500" and code in {"CU 00 021", "CU 00 031"}:
            titles.extend([
                "Модули центрального процессора CU 00 021 / CU 00 031 (II тип)",
                "Модули центрального процессора CU 00\xa0021\xa0/\xa0CU 00\xa0031 (II тип)",
            ])
    return unique_texts(titles)


def merge_documents(primary_docs: List[Any], secondary_docs: List[Any]) -> List[Any]:
    merged: List[Any] = []
    seen = set()
    for doc in [*primary_docs, *secondary_docs]:
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        key = (
            meta.get("source_file"),
            meta.get("page_title"),
            meta.get("breadcrumb_raw"),
            meta.get("ticket_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
    return merged


def ensure_module_block(text: str, modules: List[Dict[str, Any]]) -> str:
    if not modules or "Определенные модули:" in (text or ""):
        return text
    return (
        "Определенные модули:\n"
        + format_detected_modules(modules)
        + "\n\n"
        + (text or "").lstrip()
    )


def _ticket_module_match_score(doc: Any, modules: List[Dict[str, Any]]) -> int:
    if not modules:
        return 0
    meta = doc.metadata if hasattr(doc, "metadata") else {}
    content = doc.page_content if hasattr(doc, "page_content") else str(doc)
    meta_modules = normalize_compact_text(" ".join(list_text(meta.get("mentioned_modules"))))
    meta_codes = normalize_compact_text(" ".join(list_text(meta.get("mentioned_module_codes"))))
    text_norm = normalize_compact_text(content)

    score = 0
    for module in modules:
        canonical = normalize_compact_text(str(module.get("canonical") or ""))
        code = normalize_compact_text(str(module.get("module_code") or ""))
        if canonical and canonical in meta_modules:
            score += 100
        if code and code in meta_codes:
            score += 100
        if canonical and canonical in text_norm:
            score += 5
        if code and code in text_norm:
            score += 5
    return score


def _ticket_query_intent_score(doc: Any, query: str) -> int:
    text_parts = []
    meta = doc.metadata if hasattr(doc, "metadata") else {}
    text_parts.extend(list_text(meta.get("page_title")))
    text_parts.extend(list_text(meta.get("llm_symptoms")))
    text_parts.extend(list_text(meta.get("llm_solution")))
    text_parts.append(doc.page_content if hasattr(doc, "page_content") else str(doc))
    text_norm = normalize_compact_text(" ".join(text_parts))
    query_norm = normalize_compact_text(query)

    score = 0
    for raw_token in re.findall(r"[A-Za-zА-Яа-я]+\d[\wА-Яа-я]*|\d+[A-Za-zА-Яа-я][\wА-Яа-я]*", query):
        token = normalize_compact_text(raw_token)
        if len(token) >= 5 and token in text_norm:
            score += 40
    if any(term in query_norm for term in ["проблем", "ошиб", "неработ", "сбой", "исключ"]):
        for term in ["ошиб", "проблем", "неисправ", "несовмест", "прошив", "подключ", "исключ", "авари", "сбой"]:
            if term in text_norm:
                score += 10
        for weak_term in ["заказ", "постав", "снят", "производств", "недоступ"]:
            if weak_term in text_norm:
                score -= 12
    return score


def rank_tickets_for_modules(
    tickets: List[Any],
    modules: List[Dict[str, Any]],
    *,
    query: str = "",
    limit: int = 2,
) -> List[Any]:
    if not modules:
        return tickets[:limit]
    ranked = sorted(
        tickets,
        key=lambda doc: (
            min(_ticket_module_match_score(doc, modules), 200),
            _ticket_query_intent_score(doc, query),
            _ticket_module_match_score(doc, modules),
            float((doc.metadata if hasattr(doc, "metadata") else {}).get("rerank_score", 0) or 0),
        ),
        reverse=True,
    )
    return ranked[:limit]
