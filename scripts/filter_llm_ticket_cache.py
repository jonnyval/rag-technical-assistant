import argparse
import csv
import html
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "llm_cache_tickets"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "llm_cache_tickets_filtered"

ALLOWED_STATUSES = {
    "closed",
    "solved",
    "resolved",
    "done",
    "\u0437\u0430\u043a\u0440\u044b\u0442\u043e",
    "\u0440\u0435\u0448\u0435\u043d\u043e",
    "\u0437\u0430\u0432\u0435\u0440\u0448\u0435\u043d\u043e",
}

CSS_NOISE_RE = re.compile(
    r"#outlook|ExternalClass|ReadMsgBody|content-block|backgroundTable|"
    r"webkit-text-size|apple-data-detectors|mso-|border-collapse|"
    r"font-family\s*:\s*arial|line-height\s*:\s*initial",
    re.IGNORECASE,
)
CSS_DECL_RE = re.compile(r"[a-z-]{3,}\s*:\s*[^;{}]{0,120};", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
TICKET_ID_RE = re.compile(r"([A-Z]+-\d+)", re.IGNORECASE)
EQUIPMENT_FAILURE_CATEGORY = "\u041e\u0442\u043a\u0430\u0437\u041e\u0431\u043e\u0440\u041f\u0440\u043e\u0441\u043e\u0444\u0442"
RECLAMATION_MARKERS = (
    "\u0440\u0435\u043a\u043b\u0430\u043c\u0430\u0446",
    "\u0440\u0435\u043a\u043b\u0430\u043c\u0430\u0446\u0438\u043e\u043d",
    "\u0430\u043a\u0442 \u0440\u0435\u043a\u043b\u0430\u043c",
)
RECLAMATION_TITLE_PREFIXES = (
    "\u043f\u043e\u0441\u0442\u0443\u043f\u0438\u043b",
    "\u043f\u043e\u0441\u0442\u0443\u043f\u0438\u043b\u0438",
    "\u043d\u0435 \u043f\u043e\u0441\u0442\u0443\u043f\u0438\u043b",
    "\u0440\u0435\u043c\u043e\u043d\u0442",
    "\u0433\u0430\u0440\u0430\u043d\u0442\u0438\u0439\u043d",
    "\u0434\u0435\u0444\u0435\u043a\u0442\u043d",
    "\u0430\u043a\u0442 ",
)
SERVICE_TEST_MARKERS = (
    "\u0442\u0435\u0441\u0442\u043e\u0432\u043e\u0435 \u043e\u0431\u0440\u0430\u0449\u0435\u043d\u0438\u0435",
    "\u0442\u0435\u0441\u0442\u043e\u0432\u0430\u044f \u0437\u0430\u044f\u0432\u043a\u0430",
    "\u0442\u0435\u0441\u0442\u043e\u0432\u044b\u0439 \u0437\u0430\u043f\u0440\u043e\u0441",
    "\u0442\u0435\u0441\u0442\u043e\u0432\u043e\u0435",
    "\u0442\u0435\u0441\u0442!",
    "test",
    "\u043d\u0430\u0437\u043d\u0430\u0447\u0438\u0442\u044c \u043d\u0430 \u0433\u0440\u0443\u043f\u043f\u0443 \u043e\u0442\u0432\u0435\u0442\u0441\u0442\u0432\u0435\u043d\u043d\u044b\u0445",
)
RETARGET_MARKERS = (
    "\u0440\u0435\u0442\u0430\u0440\u0433\u0435\u0442",
    "\u0440\u0435\u0442\u0430\u0440\u0433\u0435\u0442\u0438\u043d\u0433",
)
LICENSE_ADMIN_PRODUCTS = (
    "splitopc",
    "drvmanager",
    "drvmngr",
    "drv mngr",
    "drv manager",
    "drv-manager",
    "drvmgr",
)
LICENSE_ADMIN_MARKERS = (
    "\u043b\u0438\u0446\u0435\u043d\u0437",
    "\u043a\u043b\u044e\u0447",
    "\u0430\u043a\u0442\u0438\u0432\u0430\u0446",
    "\u0441\u0435\u0440\u0438\u0439\u043d",
    "\u0441\u0435\u0440\u0432\u0435\u0440 \u043b\u0438\u0446\u0435\u043d\u0437",
    "license",
    "licence",
    "activation",
    "hasp",
    "guardant",
)
TRAINING_MARKERS = (
    "\u043e\u0431\u0443\u0447\u0435\u043d",
    "\u043e\u0431\u0443\u0447\u0430\u044e\u0449\u0438\u0439 \u043a\u0443\u0440\u0441",
    "\u043e\u0441\u043d\u043e\u0432\u043d\u043e\u0439 \u043a\u0443\u0440\u0441",
    "\u0432\u0432\u043e\u0434\u043d\u044b\u0439 \u043a\u0443\u0440\u0441",
    "\u043a\u0443\u0440\u0441\u044b ",
    "\u0432\u0435\u0431\u0438\u043d\u0430\u0440",
    "\u0441\u0435\u043c\u0438\u043d\u0430\u0440",
)
COMMERCIAL_MARKERS = (
    "\u043a\u043e\u043c\u043c\u0435\u0440\u0447\u0435\u0441\u043a\u043e\u0435 \u043f\u0440\u0435\u0434\u043b\u043e\u0436\u0435\u043d\u0438\u0435",
    "\u0441\u0442\u043e\u0438\u043c\u043e\u0441\u0442\u044c \u043e\u0431\u043e\u0440\u0443\u0434\u043e\u0432\u0430\u043d\u0438\u044f",
    "\u0441\u0442\u043e\u0438\u043c\u043e\u0441\u0442\u044c \u0437\u0430\u043c\u0435\u043d\u044b",
    "\u0441\u0442\u043e\u0438\u043c\u043e\u0441\u0442\u044c \u043f\u043b\u043a",
    "\u043f\u043e\u0441\u0442\u0430\u0432\u043a\u0430 \u0437\u0438\u043f",
    "\u043f\u0440\u0435\u0434\u043b\u043e\u0436\u0435\u043d\u0438\u0435 \u043e \u0441\u043e\u0442\u0440\u0443\u0434\u043d\u0438\u0447\u0435\u0441\u0442\u0432\u0435",
    "\u0434\u043e\u0433\u043e\u0432\u043e\u0440",
    "\u043e\u043f\u043b\u0430\u0442\u0430",
    "\u0441\u0447\u0435\u0442 \u043d\u0430 \u043e\u043f\u043b\u0430\u0442\u0443",
    "\u0441\u0447\u0451\u0442 \u043d\u0430 \u043e\u043f\u043b\u0430\u0442\u0443",
)
REPAIR_LOGISTICS_MARKERS = (
    "\u043f\u043e\u0441\u043b\u0435 \u0440\u0435\u043c\u043e\u043d\u0442\u0430",
    "\u0430\u043a\u0442\u044b \u043f\u043e\u0441\u043b\u0435 \u0440\u0435\u043c\u043e\u043d\u0442\u0430",
    "\u0430\u043a\u0442 \u043f\u043e\u0441\u043b\u0435 \u0440\u0435\u043c\u043e\u043d\u0442\u0430",
    "\u0438\u0437 \u0440\u0435\u043c\u043e\u043d\u0442\u0430",
    "\u0432\u0435\u0440\u043d\u0443\u0432\u0448\u0438\u043c\u0438\u0441\u044f \u0438\u0437\u0434\u0435\u043b\u0438\u044f\u043c\u0438",
    "\u0440\u0435\u043c\u043e\u043d\u0442 \u043a\u043e\u043d\u0442\u0440\u043e\u043b\u043b\u0435\u0440\u0430",
    "\u0440\u0435\u043c\u043e\u043d\u0442 r500",
    "\u0440\u0435\u043c\u043e\u043d\u0442 r600",
    "\u0433\u0430\u0440\u0430\u043d\u0442\u0438\u0439\u043d\u044b\u0439 \u0440\u0435\u043c\u043e\u043d\u0442",
    "\u043f\u043e\u0441\u0442\u0443\u043f\u0438\u043b r500",
    "\u043f\u043e\u0441\u0442\u0443\u043f\u0438\u043b regul",
    "\u043f\u043e\u0441\u0442\u0443\u043f\u0438\u043b \u043c\u043e\u0434\u0443\u043b\u044c",
    "\u043f\u043e\u0441\u0442\u0443\u043f\u0438\u043b \u043a\u043e\u043d\u0442\u0440\u043e\u043b\u043b\u0435\u0440",
)

ADMIN_NOISE_RE = re.compile(
    r"Story Point|IMP-\d+|tasks-only|Email .*message|new Email|"
    r"assigned|notification|None\s*->|"
    r"support service|please reply to this email",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\."
    r"(?:ru|com|net|org|su|kz|by|info|biz|pro|рф)",
    re.IGNORECASE,
)
EMAIL_LIKE_RE = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]+@[^\s\\/]+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?7|8)(?:[\s\-().]*\d){10}(?!\d)")
FIO_RE = re.compile(
    r"\b[\u0410-\u042f\u0401][\u0430-\u044f\u0451]+(?:-[\u0410-\u042f\u0401][\u0430-\u044f\u0451]+)?"
    r"\s+[\u0410-\u042f\u0401][\u0430-\u044f\u0451]+"
    r"\s+[\u0410-\u042f\u0401][\u0430-\u044f\u0451]+(?:\u0432\u0438\u0447|\u0432\u043d\u0430|\u0438\u0447|\u043d\u0430)?\b"
)
INITIALS_NAME_RE = re.compile(
    r"\b[\u0410-\u042f\u0401][\u0430-\u044f\u0451]+(?:-[\u0410-\u042f\u0401][\u0430-\u044f\u0451]+)?"
    r"\s+[A-Z\u0410-\u042f\u0401]\.\s*[A-Z\u0410-\u042f\u0401]\."
)
CONTACT_LINE_RE = re.compile(
    r"(?im)^\s*(?:contact|contacts?|\u043a\u043e\u043d\u0442\u0430\u043a\u0442|"
    r"\u0442\u0435\u043b\.?|\u0442\u0435\u043b\u0435\u0444\u043e\u043d|e-?mail|email)\s*[:;].*$"
)
SIGNATURE_BEFORE_CONTACT_RE = re.compile(
    r"(?is)\u0441\s+\u0443\u0432\u0430\u0436\u0435\u043d\u0438\u0435\u043c,?.{0,300}?"
    r"(?=(?:\u043c\u043e\u0431\.?|\u0442\u0435\u043b\.?|\u0442\u0435\u043b\u0435\u0444\u043e\u043d|"
    r"e-?mail|email|web)\s*:)"
)
INLINE_NAME_FIELD_RE = re.compile(
    r"(?i)(\b\u0438\u043c\u044f\s*:\s*)[^:\n]{2,120}?"
    r"(?=(?:\u043a\u043e\u043c\u043f\u0430\u043d\u0438\u044f|\u0442\u0435\u043c\u0430|e-?mail|email|"
    r"\u0442\u0435\u043b\u0435\u0444\u043e\u043d|\u0432\u043e\u043f\u0440\u043e\u0441)\s*:)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter noisy LLM ticket cache before vectorization."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="CSV report path. Default: <output-dir>/_filter_report.csv",
    )
    parser.add_argument(
        "--keep-cards",
        action="store_true",
        help="Keep cleaned card documents too. Default keeps only fact documents.",
    )
    parser.add_argument(
        "--allow-fallback-facts",
        action="store_true",
        help="Keep fact docs without llm_symptoms/llm_solution if they look meaningful.",
    )
    parser.add_argument(
        "--allow-open",
        action="store_true",
        help="Keep open tickets too. Default keeps only closed/resolved tickets.",
    )
    parser.add_argument(
        "--keep-reclamations",
        action="store_true",
        help="Keep reclamation/repair logistics tickets. Default excludes them.",
    )
    parser.add_argument(
        "--keep-retarget",
        action="store_true",
        help="Keep retargeting tickets. Default excludes them.",
    )
    parser.add_argument(
        "--keep-license-admin",
        action="store_true",
        help="Keep SplitOPC/DrvManager licensing tickets. Default excludes them.",
    )
    parser.add_argument(
        "--keep-training",
        action="store_true",
        help="Keep training/course/certificate tickets. Default excludes them.",
    )
    parser.add_argument(
        "--keep-commercial",
        action="store_true",
        help="Keep explicit commercial/procurement tickets. Default excludes them.",
    )
    parser.add_argument(
        "--keep-repair-logistics",
        action="store_true",
        help="Keep repair logistics tickets. Default excludes them.",
    )
    parser.add_argument(
        "--min-fact-chars",
        type=int,
        default=120,
        help="Minimum cleaned fact text length.",
    )
    parser.add_argument(
        "--replace-output",
        action="store_true",
        help="Delete existing output directory before writing.",
    )
    return parser.parse_args()


def load_ticket(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("top-level JSON is not a list")
    return [item for item in data if isinstance(item, dict)]


def metadata_of(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def get_ticket_id(path: Path, docs: list[dict[str, Any]]) -> str:
    for item in docs:
        ticket_id = metadata_of(item).get("ticket_id")
        if ticket_id:
            return str(ticket_id).upper()
    match = TICKET_ID_RE.search(path.stem)
    return match.group(1).upper() if match else path.stem.upper()


def list_has_text(value: Any) -> bool:
    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    return bool(str(value or "").strip())


def clean_block(block: str) -> str:
    block = html.unescape(block or "")
    block = HTML_TAG_RE.sub(" ", block)
    lines = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if CSS_NOISE_RE.search(line):
            continue
        if len(CSS_DECL_RE.findall(line)) >= 3:
            continue
        if ADMIN_NOISE_RE.search(line) and len(line) < 600:
            continue
        line = WHITESPACE_RE.sub(" ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def clean_content(text: str) -> str:
    parts = re.split(r"\n\s*---\s*\n", text or "")
    cleaned = [clean_block(part) for part in parts]
    cleaned = [part for part in cleaned if part]
    return "\n---\n".join(cleaned).strip()


def scrub_pii_text(text: str) -> str:
    text = CONTACT_LINE_RE.sub("[CONTACT REMOVED]", text or "")
    text = SIGNATURE_BEFORE_CONTACT_RE.sub("[SIGNATURE REMOVED]", text)
    text = INLINE_NAME_FIELD_RE.sub(r"\1[PERSON]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = EMAIL_LIKE_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = INITIALS_NAME_RE.sub("[PERSON]", text)
    return FIO_RE.sub("[PERSON]", text)


def scrub_pii_value(value: Any) -> Any:
    if isinstance(value, str):
        return scrub_pii_text(value)
    if isinstance(value, list):
        return [scrub_pii_value(item) for item in value]
    if isinstance(value, tuple):
        return [scrub_pii_value(item) for item in value]
    if isinstance(value, dict):
        return {key: scrub_pii_value(item) for key, item in value.items()}
    return value


def scrub_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    scrubbed: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in {"ticket_id", "ticket_url"}:
            scrubbed[key] = value
        else:
            scrubbed[key] = scrub_pii_value(value)
    return scrubbed


def is_allowed_status(status: str, allow_open: bool) -> bool:
    if allow_open:
        return True
    status_norm = str(status or "").strip().lower()
    return status_norm in ALLOWED_STATUSES


def fact_has_llm_value(item: dict[str, Any]) -> bool:
    meta = metadata_of(item)
    return list_has_text(meta.get("llm_symptoms")) or list_has_text(meta.get("llm_solution"))


def is_reclamation_ticket(docs: list[dict[str, Any]]) -> bool:
    if not docs:
        return False

    first_meta = metadata_of(docs[0])
    title = str(first_meta.get("page_title") or "").strip().lower()
    category = str(first_meta.get("category") or "").strip()
    meta_text = " ".join(
        str(metadata_of(item).get(key) or "")
        for item in docs
        for key in ("page_title", "category", "equipment_type")
    ).lower()
    content_text = " ".join(str(item.get("page_content") or "") for item in docs).lower()
    full_text = f"{meta_text} {content_text}"

    if any(marker in full_text for marker in RECLAMATION_MARKERS):
        return True
    if category == EQUIPMENT_FAILURE_CATEGORY and title.startswith(RECLAMATION_TITLE_PREFIXES):
        return True
    return False


def is_service_test_ticket(docs: list[dict[str, Any]]) -> bool:
    if not docs:
        return False

    first_meta = metadata_of(docs[0])
    title = str(first_meta.get("page_title") or "").strip().lower()
    title_norm = title.replace("\u0451", "\u0435")
    if title_norm in {"test", "\u0442\u0435\u0441\u0442", "\u0442\u0435\u0441\u0442\u043e\u0432\u043e\u0435"}:
        return True
    if any(marker in title_norm for marker in SERVICE_TEST_MARKERS):
        return True

    text = " ".join(str(item.get("page_content") or "") for item in docs).lower().replace("\u0451", "\u0435")
    return any(marker in text for marker in SERVICE_TEST_MARKERS)


def ticket_search_text(docs: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for item in docs:
        meta = metadata_of(item)
        chunks.extend(
            str(meta.get(key) or "")
            for key in (
                "page_title",
                "category",
                "equipment_type",
                "source_file",
                "llm_symptoms",
                "llm_solution",
            )
        )
        chunks.append(str(item.get("page_content") or ""))
    return " ".join(chunks).lower().replace("\u0451", "\u0435")


def ticket_metadata_search_text(docs: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for item in docs:
        meta = metadata_of(item)
        chunks.extend(
            str(meta.get(key) or "")
            for key in ("page_title", "category", "equipment_type")
        )
    return " ".join(chunks).lower().replace("\u0451", "\u0435")


def is_retarget_ticket(docs: list[dict[str, Any]]) -> bool:
    text = ticket_search_text(docs)
    return any(marker in text for marker in RETARGET_MARKERS)


def is_license_admin_ticket(docs: list[dict[str, Any]]) -> bool:
    text = ticket_search_text(docs)
    has_product = any(product in text for product in LICENSE_ADMIN_PRODUCTS)
    has_license_marker = any(marker in text for marker in LICENSE_ADMIN_MARKERS)
    return has_product and has_license_marker


def is_training_ticket(docs: list[dict[str, Any]]) -> bool:
    text = ticket_metadata_search_text(docs)
    return any(marker in text for marker in TRAINING_MARKERS)


def is_commercial_ticket(docs: list[dict[str, Any]]) -> bool:
    text = ticket_metadata_search_text(docs)
    return any(marker in text for marker in COMMERCIAL_MARKERS)


def is_repair_logistics_ticket(docs: list[dict[str, Any]]) -> bool:
    text = ticket_metadata_search_text(docs)
    return any(marker in text for marker in REPAIR_LOGISTICS_MARKERS)


def filter_docs(
    docs: list[dict[str, Any]],
    *,
    keep_cards: bool,
    allow_fallback_facts: bool,
    min_fact_chars: int,
) -> tuple[list[dict[str, Any]], list[str], int, int]:
    kept: list[dict[str, Any]] = []
    reasons: list[str] = []
    chars_before = 0
    chars_after = 0

    for item in docs:
        content = str(item.get("page_content") or "")
        chars_before += len(content)
        meta = metadata_of(item)
        level = str(meta.get("doc_level") or "").lower()

        if level == "card" and not keep_cards:
            reasons.append("card_doc_skipped")
            continue
        if level not in {"fact", "card"}:
            reasons.append("unknown_doc_level")
            continue

        cleaned_content = scrub_pii_text(clean_content(content))
        if level == "fact":
            has_llm = fact_has_llm_value(item)
            if not has_llm and not allow_fallback_facts:
                reasons.append("fact_without_llm_summary")
                continue
            if len(cleaned_content) < min_fact_chars:
                reasons.append("fact_too_short")
                continue
        elif len(cleaned_content) < 160:
            reasons.append("card_too_short")
            continue

        new_item = dict(item)
        new_item["page_content"] = cleaned_content
        new_item["metadata"] = scrub_metadata(meta)
        kept.append(new_item)
        chars_after += len(cleaned_content)

    return kept, sorted(set(reasons)), chars_before, chars_after


def decision_for_file(
    path: Path,
    docs: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
    ticket_id = get_ticket_id(path, docs)
    first_meta = metadata_of(docs[0]) if docs else {}
    status = str(first_meta.get("status") or "")

    row_base = {
        "file": str(path.relative_to(args.input_dir)),
        "ticket_id": ticket_id,
        "status": status,
        "title": str(first_meta.get("page_title") or ""),
        "docs_before": len(docs),
    }

    if not docs:
        return "drop", "empty_json", [], {**row_base, "docs_after": 0}
    if not is_allowed_status(status, args.allow_open):
        return "drop", "status_not_resolved", [], {**row_base, "docs_after": 0}
    if not args.keep_retarget and is_retarget_ticket(docs):
        return "drop", "retarget_ticket", [], {**row_base, "docs_after": 0}
    if not args.keep_license_admin and is_license_admin_ticket(docs):
        return "drop", "splitopc_drvmanager_license", [], {**row_base, "docs_after": 0}
    if not args.keep_training and is_training_ticket(docs):
        return "drop", "training_ticket", [], {**row_base, "docs_after": 0}
    if not args.keep_commercial and is_commercial_ticket(docs):
        return "drop", "commercial_ticket", [], {**row_base, "docs_after": 0}
    if not args.keep_repair_logistics and is_repair_logistics_ticket(docs):
        return "drop", "repair_logistics_ticket", [], {**row_base, "docs_after": 0}
    if not args.keep_reclamations and is_reclamation_ticket(docs):
        return "drop", "reclamation_ticket", [], {**row_base, "docs_after": 0}
    if is_service_test_ticket(docs):
        return "drop", "service_test_ticket", [], {**row_base, "docs_after": 0}

    kept_docs, reasons, chars_before, chars_after = filter_docs(
        docs,
        keep_cards=args.keep_cards,
        allow_fallback_facts=args.allow_fallback_facts,
        min_fact_chars=args.min_fact_chars,
    )
    row_base.update({
        "docs_after": len(kept_docs),
        "chars_before": chars_before,
        "chars_after": chars_after,
    })

    has_fact = any(
        str(metadata_of(item).get("doc_level") or "").lower() == "fact"
        for item in kept_docs
    )
    if not has_fact:
        reason = ";".join(reasons or ["no_fact_doc"])
        return "drop", reason, [], row_base

    return "keep", ";".join(reasons or ["ok"]), kept_docs, row_base


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "decision",
        "reason",
        "file",
        "ticket_id",
        "status",
        "title",
        "docs_before",
        "docs_after",
        "chars_before",
        "chars_after",
        "output_file",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_path = (args.report or (output_dir / "_filter_report.csv")).resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if output_dir == input_dir:
        raise SystemExit("Output directory must differ from input directory.")
    if output_dir.exists() and not args.replace_output:
        existing_json = list(output_dir.glob("*.json"))
        if existing_json:
            raise SystemExit(
                f"Output directory already contains {len(existing_json)} JSON files. "
                "Use --replace-output or choose another --output-dir."
            )
    if args.replace_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        path for path in input_dir.rglob("*.json")
        if not path.name.startswith("_")
    )
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_ticket_ids: set[str] = set()

    for path in files:
        try:
            docs = load_ticket(path)
            ticket_id = get_ticket_id(path, docs)
            if ticket_id in seen_ticket_ids:
                row = {
                    "decision": "drop",
                    "reason": "duplicate_ticket_id",
                    "file": str(path.relative_to(input_dir)),
                    "ticket_id": ticket_id,
                    "docs_before": len(docs),
                    "docs_after": 0,
                }
                rows.append(row)
                counts["drop"] += 1
                continue

            decision, reason, kept_docs, row_base = decision_for_file(path, docs, args)
            output_file = ""
            if decision == "keep":
                seen_ticket_ids.add(ticket_id)
                output_file = f"{ticket_id}_llm.json"
                write_json(output_dir / output_file, kept_docs)

            row = {
                **row_base,
                "decision": decision,
                "reason": reason,
                "output_file": output_file,
            }
            rows.append(row)
            counts[decision] += 1
        except Exception as exc:
            rows.append({
                "decision": "error",
                "reason": str(exc),
                "file": str(path.relative_to(input_dir)),
                "docs_before": 0,
                "docs_after": 0,
            })
            counts["error"] += 1

    write_report(report_path, rows)
    print(f"Input files: {len(files)}")
    print(f"Kept: {counts['keep']}")
    print(f"Dropped: {counts['drop']}")
    print(f"Errors: {counts['error']}")
    print(f"Output: {output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
