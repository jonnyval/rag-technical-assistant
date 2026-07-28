import argparse
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TICKETS_DIR = PROJECT_ROOT / "data" / "source_docs" / "docs_json"
DEFAULT_TAXONOMY = PROJECT_ROOT / "data" / "module_alias_taxonomy.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "module_alias_candidates.json"

TICKET_ID_RE = re.compile(r"\bRL-\d+\b", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_()\-]+")
EMAIL_RE = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-zА-Яа-я]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?7|8)(?:[\s\-().]*\d){10}(?!\d)")
FIO_RE = re.compile(
    r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+"
    r"[А-ЯЁ][а-яё]+\s+"
    r"[А-ЯЁ][а-яё]+(?:вич|вна|ич|на)?\b"
)
DESCRIPTOR_TOKENS = {
    "модуль",
    "блок",
    "плата",
    "контроллер",
    "процессор",
    "цп",
    "cpu",
    "центральный",
    "аналоговый",
    "дискретный",
    "коммуникационный",
    "оконечный",
    "вход",
    "выход",
    "шасси",
    "питания",
    "питание",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine module alias candidates from raw support ticket JSON files."
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=DEFAULT_TAXONOMY,
        help="Path to module alias taxonomy JSON.",
    )
    parser.add_argument(
        "--tickets-dir",
        type=Path,
        nargs="+",
        default=[DEFAULT_TICKETS_DIR],
        help="One or more directories with raw ticket JSON or llm_cache JSON files. Searched recursively.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON report with alias candidates.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Keep candidates seen at least this many times.",
    )
    parser.add_argument(
        "--window-chars",
        type=int,
        default=220,
        help="How many characters around a module mention to store as evidence.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Max evidence examples per candidate.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Limit number of ticket files for a quick test run.",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Include candidates that are already present in aliases/weak_aliases.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def strip_html(text: Any) -> str:
    if not isinstance(text, str) or not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = HTML_TAG_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def normalize_key(text: str) -> str:
    text = (text or "").lower().replace("ё", "е")
    text = re.sub(r"[\s_\-()]+", "", text)
    return text


def normalize_phrase(text: str) -> str:
    text = strip_html(text)
    text = text.strip(" \t\r\n\"'“”")
    text = re.sub(
        r"^\(\s*(модуль|процессор|контроллер|шасси|цп|cpu|блок|плата|вход|выход)",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(модуль|процессор|контроллер|шасси|цп|cpu|блок|плата|вход|выход)[?:.]\s*",
        r"\1 ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(модуль|процессор|контроллер|шасси|цп|cpu|блок|плата|вход|выход)\(\s*",
        r"\1 ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*([()])\s*", r"\1", text)
    text = re.sub(r"\s*[-_]\s*", "-", text)
    text = SPACE_RE.sub(" ", text).strip(" ,.;:|")
    return normalize_alias_display(text)


def normalize_alias_display(text: str) -> str:
    if not text:
        return ""

    first_word = text.split(" ", 1)[0]
    lowerable = {
        "Аналоговый",
        "Дискретный",
        "Коммуникационный",
        "Оконечный",
        "Модуль",
        "Контроллер",
        "Процессор",
        "Шасси",
        "Плата",
        "Блок",
        "Вход",
        "Выход",
        "ВЫХОД",
    }
    if first_word in lowerable:
        text = first_word.lower() + text[len(first_word):]

    return text


def is_bad_candidate(candidate: str) -> bool:
    lowered = candidate.lower()
    if candidate.startswith(("(", "[", "{")):
        return True
    if re.match(r"^питани[ея]\W*", lowered):
        return True
    if re.search(r"\b(модуль|процессор|контроллер|шасси)[?:.]", lowered):
        return True
    if re.search(r"\b(модуль|процессор|контроллер|шасси)\s+:", lowered):
        return True
    return False


def scrub_pii(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    return FIO_RE.sub("[PERSON]", text)


def load_taxonomy(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or not isinstance(data.get("entities"), list):
        raise ValueError("taxonomy must be a JSON object with an entities list")
    return data


def ticket_id_from_file(path: Path, ticket_data: dict[str, Any]) -> str:
    for value in (ticket_data.get("code"), path.name):
        match = TICKET_ID_RE.search(str(value or ""))
        if match:
            return match.group(0).upper()
    return path.stem.upper()


def result_blocks(raw_data: Any) -> list[dict[str, Any]]:
    if isinstance(raw_data, dict):
        result = raw_data.get("result", raw_data)
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
        return [result] if isinstance(result, dict) else []
    if isinstance(raw_data, list):
        return [item for item in raw_data if isinstance(item, dict)]
    return []


def result_block(raw_data: Any) -> dict[str, Any]:
    blocks = result_blocks(raw_data)
    return blocks[0] if blocks else {}


def ticket_text(ticket_data: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key in (
        "page_content",
        "name",
        "text",
        "description",
        "cf_artikul",
        "cf_opisanie_otkaza",
        "cf_prichina_otkaza",
        "cf_kommentarij_k_o",
        "cf_tip_oborud_reg_name",
    ):
        value = ticket_data.get(key)
        if value:
            chunks.append(strip_html(str(value)))

    metadata = ticket_data.get("metadata")
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if key in {"source_file", "source_type", "format", "ticket_url", "doc_level"}:
                continue
            if isinstance(value, (str, int, float)):
                chunks.append(strip_html(str(value)))
            elif isinstance(value, list):
                chunks.append(strip_html(" ".join(str(item) for item in value if item)))

    comments = ticket_data.get("comments_list", [])
    if isinstance(comments, list):
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            chunks.append(strip_html(str(comment.get("text") or comment.get("html") or "")))

    return "\n".join(chunk for chunk in chunks if chunk)


def iter_ticket_files(tickets_dir: Path, limit: int | None = None) -> list[Path]:
    files = sorted(
        path for path in tickets_dir.rglob("*.json")
        if not path.name.startswith("_")
    )
    return files[:limit] if limit else files


def iter_ticket_files_from_dirs(tickets_dirs: list[Path], limit: int | None = None) -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for tickets_dir in tickets_dirs:
        for path in iter_ticket_files(tickets_dir):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)
            if limit and len(files) >= limit:
                return files
    return files


def code_variants(value: str) -> set[str]:
    value = normalize_phrase(value)
    if not value:
        return set()

    compact = re.sub(r"[\s_\-()]+", "", value)
    dash = re.sub(r"[\s_]+", "-", value)
    underscore = re.sub(r"[\s\-]+", "_", value)
    variants = {value, compact, dash, underscore}

    if "(W)" in value:
        variants.add(value.replace("(W)", "W"))
        variants.add(value.replace("(W)", " W"))
        variants.add(value.replace("(W)", ""))

    return {variant.strip() for variant in variants if variant.strip()}


def known_alias_keys(entity: dict[str, Any]) -> set[str]:
    aliases = []
    aliases.extend(entity.get("aliases") or [])
    aliases.extend(entity.get("weak_aliases") or [])
    aliases.extend(entity.get("article_numbers") or [])
    aliases.append(entity.get("canonical") or "")
    aliases.append(entity.get("module_code") or "")
    return {normalize_key(str(alias)) for alias in aliases if str(alias).strip()}


def search_terms(entity: dict[str, Any]) -> list[str]:
    terms: set[str] = set()
    for value in [
        entity.get("canonical"),
        entity.get("module_code"),
        *(entity.get("aliases") or []),
        *(entity.get("article_numbers") or []),
    ]:
        if not value:
            continue
        terms.update(code_variants(str(value)))
    return sorted(terms, key=len, reverse=True)


def make_term_regex(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    escaped = escaped.replace(r"\ ", r"[\s\-_]*")
    escaped = escaped.replace(r"\-", r"[\s\-_]*")
    escaped = escaped.replace(r"_", r"[\s\-_]*")
    return re.compile(r"(?<![A-Za-zА-Яа-яЁё0-9])" + escaped + r"(?![A-Za-zА-Яа-яЁё0-9])", re.IGNORECASE)


def context_window(text: str, start: int, end: int, window_chars: int) -> str:
    left = max(0, start - window_chars)
    right = min(len(text), end + window_chars)
    return scrub_pii(SPACE_RE.sub(" ", text[left:right]).strip())


def descriptor_candidates(text: str, start: int, end: int) -> set[str]:
    mention = normalize_phrase(text[start:end])
    candidates = {mention}

    offset = max(0, start - 120)
    token_spans = list(TOKEN_RE.finditer(text[offset:min(len(text), end + 80)]))
    absolute_spans = [(m.start() + offset, m.end() + offset) for m in token_spans]
    mention_indexes = [
        index for index, (token_start, token_end) in enumerate(absolute_spans)
        if token_end > start and token_start < end
    ]
    if not mention_indexes:
        return candidates

    first_mention = min(mention_indexes)
    left_indexes: list[int] = []
    for index in range(first_mention - 1, max(-1, first_mention - 5), -1):
        token = text[absolute_spans[index][0]:absolute_spans[index][1]]
        if _norm_token(token) not in DESCRIPTOR_TOKENS:
            break
        left_indexes.append(index)

    if left_indexes:
        left_indexes.reverse()
        phrase_start = absolute_spans[left_indexes[0]][0]
        candidates.add(normalize_phrase(text[phrase_start:end]))

    return candidates


def _norm_token(token: str) -> str:
    return token.lower().replace("ё", "е").strip(".,;:!?()[]{}")


def classify_candidate(candidate: str, entity: dict[str, Any]) -> tuple[str, str]:
    module_code = str(entity.get("module_code") or "")
    suffixes = re.findall(r"\d+", module_code)
    unique_suffix = suffixes[-1] if suffixes else ""
    candidate_key = normalize_key(candidate)
    module_key = normalize_key(module_code)

    if module_key and module_key in candidate_key:
        return "aliases", "high"
    if unique_suffix and unique_suffix in candidate:
        return "aliases", "medium"
    return "weak_aliases", "low"


def add_candidate(
    bucket: dict[str, dict[str, Any]],
    *,
    candidate: str,
    canonical: str,
    ticket_id: str,
    example: str,
    suggested_bucket: str,
    confidence: str,
    max_examples: int,
) -> None:
    candidate = normalize_phrase(candidate)
    if len(candidate) < 5:
        return
    if confidence == "low":
        return
    if is_bad_candidate(candidate):
        return

    candidate_key = normalize_key(candidate)

    item = bucket.setdefault(
        candidate_key,
        {
            "alias": candidate,
            "target_field": suggested_bucket,
            "suggested_bucket": suggested_bucket,
            "confidence": confidence,
            "count": 0,
            "ticket_ids": [],
            "examples": [],
        },
    )
    item["count"] += 1
    if ticket_id not in item["ticket_ids"]:
        item["ticket_ids"].append(ticket_id)
    if len(item["examples"]) < max_examples:
        item["examples"].append({"ticket_id": ticket_id, "text": example})


def mine_candidates(
    taxonomy: dict[str, Any],
    ticket_files: list[Path],
    *,
    window_chars: int,
    max_examples: int,
    include_existing: bool,
) -> dict[str, Any]:
    entities = taxonomy["entities"]
    compiled_terms: list[tuple[dict[str, Any], set[str], set[str], list[tuple[str, re.Pattern[str]]]]] = []
    for entity in entities:
        terms = search_terms(entity)
        if not terms:
            continue
        compiled_terms.append(
            (
                entity,
                known_alias_keys(entity),
                {normalize_key(term) for term in terms if normalize_key(term)},
                [(term, make_term_regex(term)) for term in terms],
            )
        )

    candidates_by_entity: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    scanned = 0
    tickets_with_matches: set[str] = set()

    for path in ticket_files:
        try:
            raw_data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        ticket_blocks = result_blocks(raw_data)
        text = "\n".join(ticket_text(block) for block in ticket_blocks)
        if not text:
            continue
        text_key = normalize_key(text)

        ticket_id = ticket_id_from_file(path, ticket_blocks[0] if ticket_blocks else {})
        scanned += 1

        for entity, known_keys, term_keys, terms in compiled_terms:
            if not any(term_key in text_key for term_key in term_keys):
                continue

            canonical = str(entity.get("canonical") or "")
            entity_bucket = candidates_by_entity[canonical]
            entity_matched = False

            for _, pattern in terms:
                for match in pattern.finditer(text):
                    entity_matched = True
                    example = context_window(text, match.start(), match.end(), window_chars)

                    for candidate in descriptor_candidates(text, match.start(), match.end()):
                        if not include_existing and normalize_key(candidate) in known_keys:
                            continue
                        suggested_bucket, confidence = classify_candidate(candidate, entity)
                        add_candidate(
                            entity_bucket,
                            candidate=candidate,
                            canonical=canonical,
                            ticket_id=ticket_id,
                            example=example,
                            suggested_bucket=suggested_bucket,
                            confidence=confidence,
                            max_examples=max_examples,
                        )

            if entity_matched:
                tickets_with_matches.add(ticket_id)

    return {
        "version": 1,
        "source": "ticket_json_or_llm_cache",
        "taxonomy_source": str(taxonomy.get("source") or ""),
        "stats": {
            "entities": len(entities),
            "ticket_files_scanned": scanned,
            "tickets_with_matches": len(tickets_with_matches),
        },
        "entities": [
            {
                "canonical": canonical,
                "candidates": sorted(entity_candidates.values(), key=lambda item: (-item["count"], item["alias"])),
            }
            for canonical, entity_candidates in sorted(candidates_by_entity.items())
            if entity_candidates
        ],
    }


def filter_report(report: dict[str, Any], min_count: int) -> dict[str, Any]:
    filtered_entities = []
    total_candidates = 0
    for entity in report["entities"]:
        candidates = [
            candidate for candidate in entity["candidates"]
            if int(candidate.get("count") or 0) >= min_count
        ]
        if not candidates:
            continue
        total_candidates += len(candidates)
        filtered_entities.append({**entity, "candidates": candidates})

    report = {**report, "entities": filtered_entities}
    report["stats"] = {
        **report["stats"],
        "entities_with_candidates": len(filtered_entities),
        "candidates": total_candidates,
        "min_count": min_count,
    }
    return report


def main() -> int:
    args = parse_args()
    taxonomy_path = resolve_path(args.taxonomy)
    tickets_dirs = [resolve_path(path) for path in args.tickets_dir]
    output_path = resolve_path(args.output)

    if args.min_count <= 0:
        raise SystemExit("--min-count must be greater than 0")
    if not taxonomy_path.exists():
        raise SystemExit(f"Taxonomy file not found: {taxonomy_path}")
    missing_dirs = [path for path in tickets_dirs if not path.exists()]
    if missing_dirs:
        raise SystemExit(f"Tickets directory not found: {missing_dirs[0]}")

    taxonomy = load_taxonomy(taxonomy_path)
    ticket_files = iter_ticket_files_from_dirs(tickets_dirs, args.limit_files)
    report = mine_candidates(
        taxonomy,
        ticket_files,
        window_chars=args.window_chars,
        max_examples=args.max_examples,
        include_existing=args.include_existing,
    )
    report = filter_report(report, args.min_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = report["stats"]
    print(f"Ticket files scanned: {stats['ticket_files_scanned']}")
    print(f"Tickets with module matches: {stats['tickets_with_matches']}")
    print(f"Entities with candidates: {stats['entities_with_candidates']}")
    print(f"Candidates: {stats['candidates']}")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
