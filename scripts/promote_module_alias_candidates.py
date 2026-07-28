import argparse
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TAXONOMY = PROJECT_ROOT / "data" / "module_alias_taxonomy.json"
DEFAULT_CANDIDATES = PROJECT_ROOT / "data" / "module_alias_candidates.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "module_alias_taxonomy.json"
DEFAULT_REVIEW = PROJECT_ROOT / "data" / "module_alias_candidates_review.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote mined module alias candidates into a module alias taxonomy."
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=DEFAULT_TAXONOMY,
        help="Base module alias taxonomy JSON.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_CANDIDATES,
        help="Alias candidate report produced by mine_ticket_module_aliases.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Merged taxonomy output path.",
    )
    parser.add_argument(
        "--review-output",
        type=Path,
        default=DEFAULT_REVIEW,
        help="Review report for candidates that were not auto-promoted.",
    )
    parser.add_argument(
        "--auto-min-count",
        type=int,
        default=5,
        help="Promote candidates with at least this count.",
    )
    parser.add_argument(
        "--confidence",
        nargs="+",
        default=["high"],
        help="Candidate confidence values allowed for auto-promotion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing output files.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def normalize_key(text: str) -> str:
    text = (text or "").lower().replace("ё", "е")
    text = re.sub(r"[\s_\-()]+", "", text)
    return text


def entity_alias_keys(entity: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in ("canonical", "module_code"):
        value = entity.get(field)
        if value:
            keys.add(normalize_key(str(value)))
    for field in ("article_numbers", "aliases", "weak_aliases"):
        values = entity.get(field) or []
        if isinstance(values, list):
            keys.update(normalize_key(str(value)) for value in values if str(value).strip())
    return keys


def ensure_list(entity: dict[str, Any], field: str) -> list[Any]:
    value = entity.get(field)
    if isinstance(value, list):
        return value
    value = []
    entity[field] = value
    return value


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "alias": candidate.get("alias"),
        "target_field": candidate.get("target_field") or candidate.get("suggested_bucket"),
        "confidence": candidate.get("confidence"),
        "count": candidate.get("count"),
        "ticket_ids": candidate.get("ticket_ids", [])[:10],
        "examples": candidate.get("examples", [])[:3],
    }


def promote_candidates(
    taxonomy: dict[str, Any],
    candidates_report: dict[str, Any],
    *,
    auto_min_count: int,
    allowed_confidence: set[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    merged = deepcopy(taxonomy)
    entities = merged.get("entities")
    if not isinstance(entities, list):
        raise ValueError("taxonomy must contain an entities list")

    by_canonical = {
        str(entity.get("canonical")): entity
        for entity in entities
        if isinstance(entity, dict) and entity.get("canonical")
    }

    promoted_by_entity: dict[str, list[dict[str, Any]]] = {}
    review_by_entity: dict[str, list[dict[str, Any]]] = {}
    missing_entities: list[str] = []

    promoted_total = 0
    skipped_existing = 0
    review_total = 0

    for candidate_entity in candidates_report.get("entities", []):
        canonical = str(candidate_entity.get("canonical") or "")
        entity = by_canonical.get(canonical)
        if not entity:
            missing_entities.append(canonical)
            continue

        known_keys = entity_alias_keys(entity)

        for candidate in candidate_entity.get("candidates", []):
            alias = str(candidate.get("alias") or "").strip()
            target_field = str(candidate.get("target_field") or candidate.get("suggested_bucket") or "aliases")
            count = int(candidate.get("count") or 0)
            confidence = str(candidate.get("confidence") or "")
            alias_key = normalize_key(alias)

            if not alias or not alias_key:
                continue
            if alias_key in known_keys:
                skipped_existing += 1
                continue

            can_promote = (
                target_field in {"aliases", "weak_aliases"}
                and confidence in allowed_confidence
                and count >= auto_min_count
            )

            if can_promote:
                ensure_list(entity, target_field).append(alias)
                known_keys.add(alias_key)
                promoted_by_entity.setdefault(canonical, []).append(compact_candidate(candidate))
                promoted_total += 1
            else:
                review_by_entity.setdefault(canonical, []).append(compact_candidate(candidate))
                review_total += 1

    merged["last_alias_candidate_merge"] = {
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "source": str(candidates_report.get("source") or ""),
        "candidate_stats": candidates_report.get("stats", {}),
        "auto_min_count": auto_min_count,
        "allowed_confidence": sorted(allowed_confidence),
        "promoted_candidates": promoted_total,
        "review_candidates": review_total,
        "skipped_existing_candidates": skipped_existing,
        "missing_entities": missing_entities,
    }

    review_report = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "auto_min_count": auto_min_count,
        "allowed_confidence": sorted(allowed_confidence),
        "summary": {
            "promoted_candidates": promoted_total,
            "review_candidates": review_total,
            "skipped_existing_candidates": skipped_existing,
            "missing_entities": len(missing_entities),
        },
        "missing_entities": missing_entities,
        "promoted": [
            {"canonical": canonical, "candidates": candidates}
            for canonical, candidates in sorted(promoted_by_entity.items())
        ],
        "review": [
            {"canonical": canonical, "candidates": candidates}
            for canonical, candidates in sorted(review_by_entity.items())
        ],
    }

    stats = {
        "entities": len(entities),
        "promoted_candidates": promoted_total,
        "review_candidates": review_total,
        "skipped_existing_candidates": skipped_existing,
        "missing_entities": len(missing_entities),
    }
    return merged, review_report, stats


def main() -> int:
    args = parse_args()
    taxonomy_path = resolve_path(args.taxonomy)
    candidates_path = resolve_path(args.candidates)
    output_path = resolve_path(args.output)
    review_path = resolve_path(args.review_output)

    if args.auto_min_count <= 0:
        raise SystemExit("--auto-min-count must be greater than 0")
    if not taxonomy_path.exists():
        raise SystemExit(f"Taxonomy file not found: {taxonomy_path}")
    if not candidates_path.exists():
        raise SystemExit(f"Candidates file not found: {candidates_path}")

    taxonomy = load_json(taxonomy_path)
    candidates = load_json(candidates_path)
    merged, review_report, stats = promote_candidates(
        taxonomy,
        candidates,
        auto_min_count=args.auto_min_count,
        allowed_confidence={str(item) for item in args.confidence},
    )

    print(f"Entities: {stats['entities']}")
    print(f"Promoted candidates: {stats['promoted_candidates']}")
    print(f"Review candidates: {stats['review_candidates']}")
    print(f"Skipped existing candidates: {stats['skipped_existing_candidates']}")
    print(f"Missing entities: {stats['missing_entities']}")

    if args.dry_run:
        print("Dry run: output files were not written.")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    review_path.write_text(json.dumps(review_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved taxonomy: {output_path}")
    print(f"Saved review: {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
