from typing import Any, List
from urllib.parse import urljoin

from pydantic import BaseModel

from src.config import settings
from src.logger import log
from src.module_detection import list_text


class SourceReference(BaseModel):
    """Ссылка на источник, найденный ретривером."""

    source_id: str = ""
    title: str
    source_file: str
    url: str = ""
    source_type: str = ""
    page_title: str = ""
    release_version: str = ""
    equipment_type: str = ""
    library_name: str = ""
    breadcrumb: str = ""


def trim_text(text: str, max_chars: int) -> str:
    """Обрезает длинный контекст по границе строки, чтобы не переполнять лимит LLM."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit("\n", 1)[0].strip()
    if not cut:
        cut = text[:max_chars].strip()
    return f"{cut}\n\n[...контекст обрезан из-за лимита размера запроса...]"


def is_ticket_document(doc: Any) -> bool:
    """Проверяет, что найденный документ действительно является обращением, а не документацией."""
    meta = doc.metadata if hasattr(doc, "metadata") else {}
    return bool(meta.get("ticket_id")) or meta.get("format") == "ticket" or meta.get("source_type") == "support_tickets"


def resolve_source_url(meta: dict, *, ticket_only: bool = False) -> str:
    """Строит URL источника из metadata или из корня источника и имени файла."""
    if ticket_only:
        return str(meta.get("ticket_url") or "")

    direct_url = meta.get("source_url") or meta.get("url")
    if direct_url:
        return str(direct_url)

    source_type = str(meta.get("source_type") or "")
    source_file = str(meta.get("source_file") or "")
    base_url = (
        meta.get("source_root")
        or meta.get("source_base_url")
        or meta.get("base_url")
        or meta.get("root_url")
        or settings.docs_base_urls.get(source_type, "")
    )
    if base_url and source_file:
        return urljoin(str(base_url).rstrip("/") + "/", source_file)
    return ""


def _source_key_from_meta(meta: dict, *, ticket_only: bool = False) -> tuple:
    if ticket_only:
        return (
            meta.get("ticket_id") or meta.get("source_file") or "",
            resolve_source_url(meta, ticket_only=True),
        )
    return (
        meta.get("source_file") or "",
        resolve_source_url(meta),
        meta.get("page_title") or "",
        meta.get("breadcrumb_raw") or "",
    )


def _source_id_for_doc(meta: dict, sources: List[SourceReference], *, ticket_only: bool = False) -> str:
    key = _source_key_from_meta(meta, ticket_only=ticket_only)
    for source in sources:
        if ticket_only:
            source_key = (source.title or source.source_file, source.url)
        else:
            source_key = (source.source_file, source.url, source.page_title, source.breadcrumb)
        if source_key == key:
            return source.source_id
    return ""


def format_docs(
    docs: List,
    *,
    sources: List[SourceReference] | None = None,
    max_total_chars: int = 12000,
    max_doc_chars: int = 3000,
) -> str:
    """Форматирует документы для передачи в LLM."""
    if not docs:
        log.warning("format_docs: Получен пустой список документов")
        return ""

    formatted = []
    for i, doc in enumerate(docs):
        try:
            meta = doc.metadata if hasattr(doc, "metadata") else {}
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            content = trim_text(content, max_doc_chars)
            db_source = meta.get("db_source", "docs")
            source_id = _source_id_for_doc(meta, sources or [], ticket_only=db_source == "tickets")
            source_label = f"Источник: [{source_id}] | " if source_id else ""

            if db_source == "tickets":
                header = f"[{source_label}ТИКЕТ ПОДДЕРЖКИ | Файл: {meta.get('source_file', 'Unknown')}]"
            else:
                source_url = resolve_source_url(meta)
                header = (
                    f"[{source_label}{meta.get('equipment_type', 'Unknown')} "
                    f"| Library: {meta.get('library_name', 'Unknown')} "
                    f"| Release: {meta.get('release_version', 'Unknown')} "
                    f"| Page: {meta.get('page_title', 'Unknown')} "
                    f"| Файл: {meta.get('source_file', 'Unknown')} "
                    f"| Раздел: {meta.get('breadcrumb_raw', 'No section')} "
                    f"| URL: {source_url}]"
                )

            formatted.append(f"{header}\n{content}")
        except Exception as error:
            log.error("Ошибка при форматировании документа %s: %s", i, error, exc_info=True)
            if hasattr(doc, "page_content"):
                formatted.append(trim_text(doc.page_content, max_doc_chars))

    return trim_text("\n\n".join(formatted), max_total_chars)


def format_ticket_docs(
    docs: List,
    *,
    sources: List[SourceReference] | None = None,
    max_total_chars: int = 6000,
    max_doc_chars: int = 2000,
) -> str:
    """Formats support tickets with stable IDs for the LLM."""
    if not docs:
        return ""

    formatted = []
    for doc in docs:
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        if not is_ticket_document(doc):
            continue
        ticket_id = meta.get("ticket_id") or meta.get("source_file", "Unknown")
        source_file = meta.get("source_file", "Unknown")
        ticket_url = meta.get("ticket_url", "")
        modules = ", ".join(list_text(meta.get("mentioned_modules")))
        tags = ", ".join(list_text(meta.get("quality_tags")))
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        content = trim_text(content, max_doc_chars)
        source_id = _source_id_for_doc(meta, sources or [], ticket_only=True)
        source_label = f"source: [{source_id}] | " if source_id else ""
        header = f"[TICKET | {source_label}id: {ticket_id} | file: {source_file} | url: {ticket_url}]"
        meta_lines = []
        if modules:
            meta_lines.append(f"НАЙДЕННЫЕ МОДУЛИ В МЕТАДАННЫХ: {modules}")
        if tags:
            meta_lines.append(f"ТЕГИ КАЧЕСТВА В МЕТАДАННЫХ: {tags}")
        if meta_lines:
            content = "\n".join(meta_lines) + "\n" + content
        formatted.append(f"{header}\n{content}")

    return trim_text("\n\n".join(formatted), max_total_chars)


def source_references(docs: List, *, ticket_only: bool = False, prefix: str = "S") -> List[SourceReference]:
    """Собирает стабильный список источников из metadata найденных документов."""
    result: List[SourceReference] = []
    seen = set()
    for doc in docs:
        if ticket_only and not is_ticket_document(doc):
            continue
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        url = resolve_source_url(meta, ticket_only=ticket_only)
        source_file = meta.get("source_file") or meta.get("ticket_id") or "Unknown"
        title = meta.get("page_title") or meta.get("ticket_id") or meta.get("breadcrumb_raw") or source_file
        key = _source_key_from_meta(meta, ticket_only=ticket_only)
        if key in seen:
            continue
        seen.add(key)
        result.append(SourceReference(
            source_id=f"{prefix}{len(result) + 1}",
            title=str(title),
            source_file=str(source_file),
            url=str(url),
            source_type=str(meta.get("source_type", "")),
            page_title=str(meta.get("page_title", "")),
            release_version=str(meta.get("release_version", "")),
            equipment_type=str(meta.get("equipment_type", "")),
            library_name=str(meta.get("library_name", "")),
            breadcrumb=str(meta.get("breadcrumb_raw", "")),
        ))
    return result


def format_source_references(sources: List[SourceReference]) -> str:
    """Форматирует источники для промпта так, чтобы LLM ссылалась на реальные документы."""
    if not sources:
        return "Источники не найдены."
    lines = []
    for source in sources:
        location = source.url or source.source_file
        details = []
        if source.equipment_type:
            details.append(f"equipment={source.equipment_type}")
        if source.library_name:
            details.append(f"library={source.library_name}")
        if source.release_version:
            details.append(f"release={source.release_version}")
        if source.breadcrumb:
            details.append(f"section={source.breadcrumb}")
        detail_suffix = " | " + " | ".join(details) if details else ""
        lines.append(
            f"[{source.source_id}] {source.title} | file={source.source_file} | url={location}{detail_suffix}"
        )
    return "\n".join(lines)
