import argparse
import gc
import json
import os
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import EncoderBackedStore
from langchain_community.storage import SQLStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import MarkdownTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import settings  # noqa: E402


DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "llm_cache_tickets_filtered"
RAW_CACHE_DIR = PROJECT_ROOT / "data" / "llm_cache_tickets"
DEFAULT_TAG_TAXONOMY = PROJECT_ROOT / "data" / "ticket_tag_taxonomy.json"
DEFAULT_MODULE_TAXONOMY = PROJECT_ROOT / "data" / "module_alias_taxonomy.json"
MONTHLY_CACHE_RE = re.compile(r"^llm_cache_tickets_\d{4}-\d{2}-\d{2}$")
SPARSE_VECTOR_NAME = "langchain-sparse"
PARENT_NAMESPACE = "reglab_parents"
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
EQUIPMENT_CODE_GROUPS = {
    "r01": "regul_r050_r100_r200_r400_r500_r600",
    "r02": "regul_r500s",
    "r03": "astraregul_platform",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vectorize prebuilt ticket LLM cache into the configured ticket backend."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory with cached ticket JSON files. Defaults to filtered cache if it exists.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Backend name from config.yaml. Defaults to storage.vector_db.second_db.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="How many parent documents to add per batch.",
    )
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=12000,
        help="Trim very long parent docs before indexing. Use 0 to disable.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete the target Qdrant collection and SQLite parent store before indexing.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to a non-empty target. Without this or --recreate, non-empty targets abort.",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="With --append, do not skip cache files whose ticket_id is already present in Qdrant.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and validate the cache, but do not load embedding models or write to Qdrant.",
    )
    parser.add_argument(
        "--tag-taxonomy",
        type=Path,
        default=DEFAULT_TAG_TAXONOMY,
        help="Closed quality-tag taxonomy JSON for deterministic metadata enrichment.",
    )
    parser.add_argument(
        "--module-taxonomy",
        type=Path,
        default=DEFAULT_MODULE_TAXONOMY,
        help="Module alias taxonomy JSON for deterministic module metadata enrichment.",
    )
    parser.add_argument(
        "--no-enrich-metadata",
        action="store_true",
        help="Disable deterministic quality tag and module enrichment.",
    )
    return parser.parse_args()


def resolve_cache_dir(cache_dir: Path | None) -> Path:
    if cache_dir is not None:
        return cache_dir if cache_dir.is_absolute() else PROJECT_ROOT / cache_dir
    monthly_cache_dirs = [
        path for path in (PROJECT_ROOT / "data").iterdir()
        if path.is_dir() and MONTHLY_CACHE_RE.match(path.name)
    ]
    if monthly_cache_dirs:
        return max(monthly_cache_dirs, key=lambda path: path.stat().st_mtime)
    if DEFAULT_CACHE_DIR.exists():
        return DEFAULT_CACHE_DIR
    return RAW_CACHE_DIR


def resolve_project_path(path_value: str | os.PathLike[str]) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_backend(name: str) -> dict[str, Any]:
    backend = settings.db_backends.get(name)
    if not backend:
        available = ", ".join(settings.db_backends.keys())
        raise SystemExit(f"Backend '{name}' not found in config.yaml. Available: {available}")
    if backend.get("type") != "qdrant":
        raise SystemExit(f"Backend '{name}' is not a qdrant backend.")
    return backend


def make_qdrant_client(backend: dict[str, Any]) -> QdrantClient:
    if backend.get("url"):
        return QdrantClient(url=backend["url"])
    if backend.get("path"):
        path = resolve_project_path(backend["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(path))
    raise SystemExit("Qdrant backend must define either url or path.")


def parent_store_path(backend: dict[str, Any]) -> Path:
    parent_store = backend.get("parent_store", {})
    if isinstance(parent_store, dict):
        path_value = parent_store.get("path", "vector_dbs/parent_docstore.db")
    else:
        path_value = parent_store or "vector_dbs/parent_docstore.db"
    return resolve_project_path(str(path_value))


def recreate_storage(client: QdrantClient, collection_name: str, store_path: Path) -> None:
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"Deleted Qdrant collection: {collection_name}")
    if store_path.exists():
        if store_path.is_dir():
            shutil.rmtree(store_path)
        else:
            store_path.unlink()
        print(f"Deleted parent store: {store_path}")


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    dense_embeddings: HuggingFaceEmbeddings,
) -> None:
    if client.collection_exists(collection_name):
        return

    probe_vector = dense_embeddings.embed_query("dimension probe")
    vector_size = len(probe_vector)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(),
        },
        on_disk_payload=True,
    )
    print(f"Created Qdrant collection: {collection_name} (dense size={vector_size})")


def collection_points_count(client: QdrantClient, collection_name: str) -> int:
    if not client.collection_exists(collection_name):
        return 0
    info = client.get_collection(collection_name)
    return int(info.points_count or 0)


def build_retriever(
    backend: dict[str, Any],
    client: QdrantClient,
    collection_name: str,
    store_path: Path,
    dense_embeddings: HuggingFaceEmbeddings,
) -> ParentDocumentRetriever:
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    qdrant = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
    )

    store_path.parent.mkdir(parents=True, exist_ok=True)
    byte_store = SQLStore(
        namespace=PARENT_NAMESPACE,
        db_url=f"sqlite:///{store_path}",
    )
    byte_store.create_schema()

    docstore = EncoderBackedStore(
        store=byte_store,
        key_encoder=lambda key: key,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads,
    )

    child_splitter = MarkdownTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_chunk_overlap,
    )
    return ParentDocumentRetriever(
        vectorstore=qdrant,
        docstore=docstore,
        child_splitter=child_splitter,
        search_kwargs={"k": settings.top_k_retrieval},
    )


def trim_content(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half].rstrip()
        + "\n\n...[LONG TICKET TEXT TRIMMED FOR INDEXING]...\n\n"
        + text[-half:].lstrip()
    )


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


def list_text(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def unique_texts(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in list_text(value):
            key = item.lower()
            if key not in seen:
                seen.add(key)
                result.append(item)
    return result


def normalize_match_text(value: str) -> str:
    value = str(value or "").lower().replace("ё", "е")
    value = re.sub(r"[_\-.]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_compact_text(value: str) -> str:
    value = str(value or "").lower().replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я]+", "", value)


def contains_term(text_norm: str, term: str) -> bool:
    term_norm = normalize_match_text(term)
    if not term_norm:
        return False
    pattern = r"(?<![0-9a-zа-я])" + re.escape(term_norm) + r"(?![0-9a-zа-я])"
    return re.search(pattern, text_norm, re.IGNORECASE) is not None


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return data if isinstance(data, dict) else {}


def quality_tag_rules(taxonomy_path: Path) -> list[dict[str, Any]]:
    taxonomy = load_json_object(taxonomy_path)
    aliases_by_tag = taxonomy.get("aliases", {})
    rules: list[dict[str, Any]] = []
    for group in taxonomy.get("groups", []):
        if not isinstance(group, dict):
            continue
        products = [str(product) for product in group.get("products", []) if str(product).strip()]
        for tag in group.get("tags", []):
            tag_text = str(tag).strip()
            if not tag_text:
                continue
            aliases = [tag_text, *[str(alias) for alias in aliases_by_tag.get(tag_text, [])]]
            rules.append(
                {
                    "name": tag_text,
                    "group_id": str(group.get("id") or ""),
                    "group_title": str(group.get("title") or group.get("id") or ""),
                    "products": products,
                    "aliases": list(dict.fromkeys(alias for alias in aliases if alias.strip())),
                }
            )
    return rules


def load_module_rules(taxonomy_path: Path) -> list[dict[str, Any]]:
    taxonomy = load_json_object(taxonomy_path)
    modules: list[dict[str, Any]] = []
    for item in taxonomy.get("entities", []):
        if not isinstance(item, dict):
            continue
        canonical = str(item.get("canonical") or "").strip()
        module_code = str(item.get("module_code") or "").strip()
        product_family = str(item.get("product_family") or "").strip()
        aliases = unique_texts(
            [
                canonical,
                module_code,
                item.get("article_numbers", []),
                item.get("aliases", []),
                item.get("weak_aliases", []),
                generated_module_aliases(product_family, module_code),
            ]
        )
        alias_keys = sorted(
            {normalize_compact_text(alias) for alias in aliases if len(normalize_compact_text(alias)) >= 4},
            key=len,
            reverse=True,
        )
        if canonical and alias_keys:
            modules.append(
                {
                    "canonical": canonical,
                    "product_family": product_family,
                    "module_code": module_code,
                    "function": str(item.get("function") or "").strip(),
                    "russian_name": str(item.get("russian_name") or "").strip(),
                    "confidence": str(item.get("confidence") or "").strip(),
                    "alias_keys": alias_keys,
                }
            )
    return modules


def generated_module_aliases(product_family: str, module_code: str) -> list[str]:
    aliases: list[str] = []
    family_short = ""
    family_match = re.search(r"\b(R\d{3}S?)\b", product_family, flags=re.IGNORECASE)
    if family_match:
        family_short = family_match.group(1).upper()

    parts = re.findall(r"[A-Za-z]+|\d+", module_code)
    if len(parts) >= 3:
        prefix = parts[0].upper()
        middle = parts[1]
        suffix = parts[2]
        aliases.extend([f"{prefix}{middle}{suffix}", f"{prefix}{middle} {suffix}", f"{prefix} {middle}{suffix}"])
        if family_short:
            aliases.extend(
                [
                    f"{family_short} {prefix}{middle}{suffix}",
                    f"{family_short} {prefix}{middle} {suffix}",
                    f"{family_short} {prefix} {middle}{suffix}",
                ]
            )
        if len(middle) == 2:
            aliases.extend([f"{prefix}0{middle}{suffix}", f"{prefix}0{middle} {suffix}"])
            if family_short:
                aliases.extend([f"{family_short} {prefix}0{middle}{suffix}", f"{family_short} {prefix}0{middle} {suffix}"])
        if middle == "00":
            aliases.extend([f"{prefix} {suffix}", f"{prefix}{suffix}", f"{prefix}-{suffix}", f"{prefix}_{suffix}"])
            if family_short:
                aliases.extend(
                    [f"{family_short} {prefix} {suffix}", f"{family_short} {prefix}{suffix}", f"{family_short}-{prefix}-{suffix}"]
                )
    return aliases


def file_search_text(items: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in items:
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
        parts.extend(
            unique_texts(
                [
                    metadata.get("page_title"),
                    metadata.get("equipment_type"),
                    metadata.get("category"),
                    metadata.get("cf_tip_oborud_reg_name"),
                    metadata.get("cf_kategoriya_or_name"),
                    metadata.get("quality_tags"),
                    metadata.get("llm_quality_tags"),
                    metadata.get("llm_symptoms"),
                    metadata.get("llm_solution"),
                    item.get("page_content"),
                ]
            )
        )
    return "\n".join(parts)


def existing_quality_tags(items: list[dict[str, Any]]) -> list[str]:
    tags: list[Any] = []
    for item in items:
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
        tags.extend([metadata.get("quality_tags"), metadata.get("llm_quality_tags")])
    return unique_texts(tags)


def deterministic_quality_tags(
    text: str,
    items: list[dict[str, Any]],
    rules: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    text_norm = normalize_match_text(text)
    metadata_values: list[str] = []
    equipment_codes: set[str] = set()
    for item in items:
        metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
        metadata_values.extend(unique_texts([metadata.get("equipment_type"), metadata.get("cf_tip_oborud_reg_name")]))
        code = str(metadata.get("cf_tip_oborud_reg") or "").lower()
        if code:
            equipment_codes.add(code)

    matched_group_ids: list[str] = []
    matched_group_titles: list[str] = []
    matched_tags: list[str] = []
    for rule in rules:
        group_id = rule["group_id"]
        product_match = any(EQUIPMENT_CODE_GROUPS.get(code) == group_id for code in equipment_codes)
        if not product_match:
            product_match = any(contains_term(text_norm, product) for product in rule["products"])
        if not product_match:
            product_match = any(contains_term(normalize_match_text(" ".join(metadata_values)), product) for product in rule["products"])
        if not product_match:
            continue

        if group_id and group_id not in matched_group_ids:
            matched_group_ids.append(group_id)
        if rule["group_title"] and rule["group_title"] not in matched_group_titles:
            matched_group_titles.append(rule["group_title"])
        if any(contains_term(text_norm, alias) for alias in rule["aliases"]):
            matched_tags.append(rule["name"])

    return unique_texts([existing_quality_tags(items), matched_tags]), matched_group_ids, matched_group_titles


def deterministic_modules(text: str, module_rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = normalize_compact_text(text)
    matched: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for module in module_rules:
        match_span: tuple[int, int] | None = None
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


def build_file_enrichment(
    items: list[dict[str, Any]],
    tag_rules: list[dict[str, Any]],
    module_rules: list[dict[str, Any]],
) -> dict[str, Any]:
    text = file_search_text(items)
    quality_tags, group_ids, group_titles = deterministic_quality_tags(text, items, tag_rules)
    modules = deterministic_modules(text, module_rules)
    return {
        "quality_tags": quality_tags,
        "quality_tag_count": len(quality_tags),
        "ticket_product_groups": group_ids,
        "ticket_product_group_titles": group_titles,
        "mentioned_modules": [module["canonical"] for module in modules],
        "mentioned_module_codes": [module["module_code"] for module in modules if module["module_code"]],
        "mentioned_module_families": unique_texts(module["product_family"] for module in modules if module["product_family"]),
        "mentioned_module_functions": unique_texts(module["function"] for module in modules if module["function"]),
    }


def append_enrichment_to_content(content: str, enrichment: dict[str, Any]) -> str:
    lines: list[str] = []
    if enrichment.get("quality_tags") and "ТЕГИ КАЧЕСТВА:" not in content:
        lines.append("ТЕГИ КАЧЕСТВА: " + ", ".join(enrichment["quality_tags"]))
    if enrichment.get("mentioned_modules") and "НАЙДЕННЫЕ МОДУЛИ:" not in content:
        lines.append("НАЙДЕННЫЕ МОДУЛИ: " + ", ".join(enrichment["mentioned_modules"]))
    if not lines:
        return content
    return content.rstrip() + "\n\n" + "\n".join(lines)


def load_documents_from_file(
    path: Path,
    max_doc_chars: int,
    *,
    tag_rules: list[dict[str, Any]] | None = None,
    module_rules: list[dict[str, Any]] | None = None,
    enrich_metadata: bool = True,
) -> list[Document]:
    with path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    if isinstance(raw_data, dict):
        items = [raw_data]
    elif isinstance(raw_data, list):
        items = raw_data
    else:
        raise ValueError("expected JSON object or list")

    enrichment: dict[str, Any] = {}
    if enrich_metadata:
        enrichment = build_file_enrichment(items, tag_rules or [], module_rules or [])

    docs: list[Document] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = item.get("page_content")
        if not isinstance(content, str) or not content.strip():
            continue
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata = scrub_metadata(dict(metadata))
        if enrichment:
            metadata.update(enrichment)
        metadata.setdefault("source_cache_file", path.name)
        scrubbed_content = scrub_pii_text(content.strip())
        if enrichment:
            scrubbed_content = append_enrichment_to_content(scrubbed_content, enrichment)
        docs.append(
            Document(
                page_content=trim_content(scrubbed_content, max_doc_chars),
                metadata=metadata,
            )
        )
    return docs


def iter_cache_files(cache_dir: Path) -> list[Path]:
    return sorted(
        path for path in cache_dir.rglob("*.json")
        if not path.name.startswith("_")
    )


def ticket_id_from_cache_file(path: Path) -> str:
    match = re.search(r"(RL-\d+)", path.stem, re.IGNORECASE)
    return match.group(1).upper() if match else path.stem.upper()


def extract_ticket_id_from_payload(payload: dict[str, Any]) -> str | None:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get("ticket_id"):
        return str(metadata["ticket_id"]).upper()
    if payload.get("ticket_id"):
        return str(payload["ticket_id"]).upper()
    return None


def existing_ticket_ids(client: QdrantClient, collection_name: str) -> set[str]:
    if not client.collection_exists(collection_name):
        return set()

    ticket_ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            ticket_id = extract_ticket_id_from_payload(payload)
            if ticket_id:
                ticket_ids.add(ticket_id)
        if offset is None:
            break
    return ticket_ids


def filter_existing_cache_files(files: list[Path], known_ticket_ids: set[str]) -> tuple[list[Path], int]:
    if not known_ticket_ids:
        return files, 0
    filtered = [path for path in files if ticket_id_from_cache_file(path) not in known_ticket_ids]
    return filtered, len(files) - len(filtered)


def chunks(items: list[Path], size: int) -> Iterable[list[Path]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]


def read_batch(
    files: list[Path],
    max_doc_chars: int,
    *,
    tag_rules: list[dict[str, Any]] | None = None,
    module_rules: list[dict[str, Any]] | None = None,
    enrich_metadata: bool = True,
) -> tuple[list[Document], int]:
    docs: list[Document] = []
    errors = 0
    for path in files:
        try:
            docs.extend(
                load_documents_from_file(
                    path,
                    max_doc_chars,
                    tag_rules=tag_rules,
                    module_rules=module_rules,
                    enrich_metadata=enrich_metadata,
                )
            )
        except Exception as exc:
            errors += 1
            print(f"Skipping {path.name}: {exc}")
    return docs, errors


def main() -> None:
    args = parse_args()
    cache_dir = resolve_cache_dir(args.cache_dir).resolve()
    backend_name = args.db or settings.second_db_name
    backend = get_backend(backend_name)
    collection_name = backend["collection"]
    store_path = parent_store_path(backend)

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0.")
    if not cache_dir.exists():
        raise SystemExit(f"Cache directory does not exist: {cache_dir}")

    files = iter_cache_files(cache_dir)
    if not files:
        raise SystemExit(f"No JSON files found in: {cache_dir}")

    print("=== Ticket Cache Vectorization ===")
    print(f"Cache dir: {cache_dir}")
    print(f"Backend: {backend_name}")
    print(f"Collection: {collection_name}")
    print(f"Parent store: {store_path}")
    print(f"Files: {len(files)}")
    print(f"Batch size: {args.batch_size}")

    tag_rules: list[dict[str, Any]] = []
    module_rules: list[dict[str, Any]] = []
    enrich_metadata = not args.no_enrich_metadata
    if enrich_metadata:
        tag_taxonomy_path = resolve_project_path(args.tag_taxonomy)
        module_taxonomy_path = resolve_project_path(args.module_taxonomy)
        tag_rules = quality_tag_rules(tag_taxonomy_path)
        module_rules = load_module_rules(module_taxonomy_path)
        print(f"Tag taxonomy: {tag_taxonomy_path} ({len(tag_rules)} rules)")
        print(f"Module taxonomy: {module_taxonomy_path} ({len(module_rules)} modules)")
    else:
        print("Deterministic metadata enrichment: disabled")

    if args.dry_run:
        total_docs = 0
        total_errors = 0
        for batch_files in tqdm(list(chunks(files, args.batch_size)), desc="Dry run"):
            docs, errors = read_batch(
                batch_files,
                args.max_doc_chars,
                tag_rules=tag_rules,
                module_rules=module_rules,
                enrich_metadata=enrich_metadata,
            )
            total_docs += len(docs)
            total_errors += errors
        print(f"Dry run complete. Documents: {total_docs}, file errors: {total_errors}")
        return

    client = make_qdrant_client(backend)
    if args.recreate:
        recreate_storage(client, collection_name, store_path)
    elif args.append and not args.allow_duplicates:
        existing_ids = existing_ticket_ids(client, collection_name)
        files, skipped_existing_tickets = filter_existing_cache_files(files, existing_ids)
        print(f"Existing ticket_ids in Qdrant: {len(existing_ids)}")
        print(f"Skipped cache files already indexed: {skipped_existing_tickets}")
        print(f"Files left for append: {len(files)}")
        if not files:
            print("Nothing to append.")
            return

    dense_embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": settings.device},
        encode_kwargs={
            "normalize_embeddings": True,
            "prompt": "Instruct: Retrieve relevant technical support ticket to answer the query.\nQuery: ",
        },
    )

    ensure_collection(client, collection_name, dense_embeddings)
    existing_points = collection_points_count(client, collection_name)
    if existing_points and not args.append and not args.recreate:
        raise SystemExit(
            f"Target collection already has {existing_points} points. "
            "Use --recreate to rebuild or --append to add more."
        )

    retriever = build_retriever(
        backend=backend,
        client=client,
        collection_name=collection_name,
        store_path=store_path,
        dense_embeddings=dense_embeddings,
    )

    total_docs = 0
    total_errors = 0
    batch_files_iter = list(chunks(files, args.batch_size))
    for batch_files in tqdm(batch_files_iter, desc="Indexing batches"):
        docs, errors = read_batch(
            batch_files,
            args.max_doc_chars,
            tag_rules=tag_rules,
            module_rules=module_rules,
            enrich_metadata=enrich_metadata,
        )
        total_errors += errors
        if not docs:
            continue
        retriever.add_documents(docs)
        total_docs += len(docs)
        del docs
        gc.collect()

    final_points = collection_points_count(client, collection_name)
    print("\nDone.")
    print(f"Indexed parent documents: {total_docs}")
    print(f"File errors: {total_errors}")
    print(f"Qdrant points in collection: {final_points}")
    print(f"Collection: {collection_name}")
    print(f"Parent store: {store_path}")


if __name__ == "__main__":
    main()
