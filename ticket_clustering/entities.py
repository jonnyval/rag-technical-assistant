from __future__ import annotations

import re
from dataclasses import dataclass

from ticket_clustering.data import SymptomNode


SERIES_RE = re.compile(r"(?<![A-ZА-Я0-9])R\s*(050|100|200|400|500S?|600)(?![A-ZА-Я0-9])", re.IGNORECASE)
MODULE_RE = re.compile(r"(?<![A-Z0-9])([A-Z]{2,5})[ _-]?(\d{2})[ _-]?(\d{3})(?!\d)", re.IGNORECASE)
LIBRARY_RE = re.compile(r"(?<![A-Za-z0-9_])((?:Ps|Sys)[A-Z][A-Za-z0-9_]{2,})(?![A-Za-z0-9_])")


ENTITY_PATTERNS: dict[str, dict[str, tuple[str, ...]]] = {
    "protocols": {
        "opc_ua": (r"\bOPC[\s._-]*UA\b",),
        "opc_da": (r"\bOPC[\s._-]*DA\b",),
        "modbus_tcp": (r"\bMODBUS[\s._-]*TCP\b",),
        "modbus_rtu": (r"\bMODBUS[\s._-]*(?:RTU|SERIAL)\b",),
        "ntp": (r"\bNTP\b",),
        "profinet": (r"\bPROFINET\b",),
        "profibus": (r"\bPROFIBUS\b",),
        "ethercat": (r"\bETHERCAT\b",),
        "regulbus": (r"\bREGUL[\s._-]*BUS\b",),
        "mqtt": (r"\bMQTT\b",),
        "iec_60870_5_104": (r"\b(?:IEC\s*)?60870[\s._/-]*5[\s._/-]*104\b", r"\bIEC[\s._-]*104\b"),
    },
    "components": {
        "astraide": (r"\bASTRA[\s._-]*IDE\b",),
        "astraregul": (r"\bASTRA[\s._-]*REGUL\b",),
        "regul_rts": (r"\bREGUL[\s._-]*RTS\b",),
        "codesys": (r"\bCODESYS\b",),
        "psdiagn": (r"\bPS[\s._-]*DIAGN\b",),
        "virtual_plc": (r"\b(?:ВПЛК|ВИРТУАЛЬН\w*\s+ПЛК)\b",),
        "redundancy": (r"\b(?:REDUNDAN\w*|РЕЗЕРВИР\w*|CPU[\s._-]*[AB])\b",),
        "qnx": (r"\bQNX\b",),
        "astra_linux": (r"\bASTRA[\s._-]*LINUX\b",),
    },
    "issue_types": {
        "connection": (r"\b(?:СВЯЗ\w*|ПОДКЛЮЧ\w*|CONNECT\w*)\b",),
        "project_download": (r"\b(?:ЗАГРУЗ\w*\s+ПРОЕКТ\w*|DOWNLOAD\s+DENIED)\b",),
        "installation_start": (r"\b(?:УСТАНОВ\w*|ИНСТАЛЛ\w*|ЗАПУСК\w*)\b",),
        "synchronization": (r"\b(?:СИНХРОНИЗ\w*|РАССИНХРОНИЗ\w*|SYNC\b)\b",),
        "documentation": (r"\b(?:ДОКУМЕНТАЦ\w*|РУКОВОДСТВ\w*|ИНСТРУКЦ\w*)\b",),
        "licensing": (r"\b(?:ЛИЦЕНЗ\w*|LICENSE\w*)\b",),
        "firmware": (r"\b(?:ПРОШИВ\w*|FIRMWARE\w*)\b",),
        "performance": (r"\b(?:ЗАГРУЗК\w*\s+(?:ЦП|CPU)|ПРОИЗВОДИТЕЛЬНОСТ\w*|100\s*%)\b",),
        "diagnostics": (r"\b(?:САМОДИАГНОСТ\w*|SELF[\s._-]*DIAGNOSTIC|ДИАГНОСТ\w*)\b",),
        "reset_backup": (r"\b(?:СБРОС\w*|FACTORY\s+RESET|РЕЗЕРВН\w*\s+КОПИ\w*|BACKUP)\b",),
    },
}

COMPILED_PATTERNS = {
    group: {
        name: tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
        for name, patterns in values.items()
    }
    for group, values in ENTITY_PATTERNS.items()
}


@dataclass(frozen=True)
class EntityProfile:
    series: frozenset[str]
    modules: frozenset[str]
    protocols: frozenset[str]
    components: frozenset[str]
    issue_types: frozenset[str]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "series": sorted(self.series),
            "modules": sorted(self.modules),
            "protocols": sorted(self.protocols),
            "components": sorted(self.components),
            "issue_types": sorted(self.issue_types),
        }


def normalize_series(match: re.Match[str]) -> str:
    return f"R{match.group(1).upper()}"


def detect_entities(text: str, equipment_values: list[str] | None = None) -> EntityProfile:
    source = " ".join([text, *(equipment_values or [])])
    explicit_series = frozenset(normalize_series(match) for match in SERIES_RE.finditer(text))
    equipment_series = frozenset(
        normalize_series(match)
        for value in (equipment_values or [])
        for match in SERIES_RE.finditer(value)
    )
    # A precise model in the symptom is stronger than a broad equipment category.
    # A multi-series equipment bucket is only taxonomy metadata and must not act as an anchor.
    series = explicit_series or (equipment_series if len(equipment_series) == 1 else frozenset())
    modules = frozenset(
        f"{match.group(1).upper()} {match.group(2)} {match.group(3)}"
        for match in MODULE_RE.finditer(source)
    )
    dynamic_libraries = {match.group(1).casefold() for match in LIBRARY_RE.finditer(source)}
    groups: dict[str, frozenset[str]] = {}
    for group, rules in COMPILED_PATTERNS.items():
        matched = {
            name
            for name, patterns in rules.items()
            if any(pattern.search(source) for pattern in patterns)
        }
        groups[group] = frozenset(matched)
    components = frozenset(set(groups["components"]) | {f"library:{name}" for name in dynamic_libraries})
    return EntityProfile(
        series=series,
        modules=modules,
        protocols=groups["protocols"],
        components=components,
        issue_types=groups["issue_types"],
    )


def profile_node(node: SymptomNode) -> EntityProfile:
    equipment = sorted({occurrence.equipment for occurrence in node.occurrences if occurrence.equipment})
    return detect_entities(node.display_text, equipment)


def profile_nodes(nodes: list[SymptomNode]) -> list[EntityProfile]:
    return [profile_node(node) for node in nodes]


def adjusted_similarity(
    base_similarity: float,
    left: EntityProfile,
    right: EntityProfile,
    config: dict,
) -> tuple[float, str]:
    # Exact single-model conflicts are unsafe: R500 and R500S must not become one topic.
    if len(left.series) == len(right.series) == 1 and left.series.isdisjoint(right.series):
        if bool(config.get("block_series_conflicts", True)):
            return base_similarity, "series_conflict"

    score = base_similarity
    weights = config.get("bonuses", {})
    for field in ("series", "modules", "protocols", "components", "issue_types"):
        left_values = getattr(left, field)
        right_values = getattr(right, field)
        if left_values and right_values and left_values.intersection(right_values):
            score += float(weights.get(field, 0.0))

    penalties = config.get("penalties", {})
    left_anchors = left.modules | left.protocols | left.components
    right_anchors = right.modules | right.protocols | right.components
    if bool(left_anchors) != bool(right_anchors):
        score -= float(penalties.get("unmatched_anchor", 0.0))
    if len(left.protocols) == len(right.protocols) == 1 and left.protocols.isdisjoint(right.protocols):
        score -= float(penalties.get("protocol_conflict", 0.0))
    if left.components and right.components and left.components.isdisjoint(right.components):
        score -= float(penalties.get("component_conflict", 0.0))
    return max(-1.0, min(1.0, score)), "ok"
