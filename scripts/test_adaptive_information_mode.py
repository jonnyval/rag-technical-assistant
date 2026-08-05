"""Regression checks for Adaptive's neutral information-search role."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.api_server import _format_adaptive_answer
from src.context_formatting import SourceReference, format_adaptive_search_body
from src.engine import (
    ADAPTIVE_INFORMATION_PROMPT,
    AdaptiveInformationResponse,
    apply_adaptive_information_role,
)
from src.evidence_guard import apply_diagnostic_scope_guard


def make_response():
    return SimpleNamespace(
        docs_answer=(
            "Параметр Bootproject.HardwareSelfDiagFail.Deny описан в приложении "
            "руководства [D1]. Значение 1 запрещает загрузку при ошибке самодиагностики [D1]."
        ),
        similar_tickets=[
            SimpleNamespace(
                ticket_id="RL-100",
                problem_summary="После обновления возникало сообщение Download denied.",
                solution_summary="Версии AstraIDE и прошивки были приведены к совместимым.",
                relevance_reason="Совпадают сообщение и этап загрузки проекта.",
            )
        ],
        doc_sources=[
            SourceReference(source_id="D1", title="Приложение A", source_file="guide.docx")
        ],
        ticket_sources=[
            SourceReference(
                source_id="T1",
                title="RL-100",
                source_file="RL-100.json",
                url="https://support.example/RL-100",
            )
        ],
        missing_context="Точная версия прошивки в источниках не указана.",
        recommended_questions=["Проверьте версию"],
        internal_notes=["Посоветовать обновление"],
        draft_private_comment="Проверьте прошивку и обновите контроллер.",
    )


def test_formatter_separates_docs_from_historical_outcomes() -> None:
    body = format_adaptive_search_body(make_response())
    assert "**По документации**" in body
    assert "**Как решали в похожих обращениях**" in body
    assert "[T1] **RL-100**" in body
    assert "Как было решено:" in body
    assert "**Что не подтверждено найденными источниками**" in body
    assert "Проверьте версию" not in body


def test_adaptive_role_discards_advice_fields_and_draft() -> None:
    response = apply_adaptive_information_role(make_response())
    assert response.recommended_questions == []
    assert response.internal_notes == []
    assert "Проверьте прошивку" not in response.draft_private_comment
    assert "Значение 1 запрещает загрузку" in response.draft_private_comment


def test_api_answer_uses_search_digest_and_source_footer() -> None:
    answer = _format_adaptive_answer(make_response())
    assert "Как решали в похожих обращениях" in answer
    assert "https://support.example/RL-100" in answer
    assert "Проверьте прошивку" not in answer


def test_information_guard_reports_missing_evidence_without_advice() -> None:
    response = make_response()
    guarded = apply_diagnostic_scope_guard(
        response,
        "ошибка self-diagnostic что делать?",
        "общее руководство runtime",
        information_mode=True,
    )
    assert "не подтверждена" in guarded.docs_answer
    assert "Для предметного разбора нужны" not in guarded.docs_answer
    assert "не следует" not in guarded.docs_answer


def test_adaptive_schema_preserves_provider_ticket_aliases() -> None:
    response = AdaptiveInformationResponse.model_validate({
        "docs_answer": "Факт [D1].",
        "draft_private_comment": "Факт [D1].",
        "similar_tickets": [{
            "ticket_id": "RL-5754",
            "situation": "Требовался аппаратный сброс.",
            "resolution": "В обращении применили сервисный режим.",
            "similarity": "Совпадает модель и операция.",
        }],
        "evidence_notes": ["Факт подтверждён источником [D1]."],
    })
    assert response.similar_tickets[0].problem_summary == "Требовался аппаратный сброс."
    assert response.similar_tickets[0].solution_summary == "В обращении применили сервисный режим."
    assert response.similar_tickets[0].relevance_reason == "Совпадает модель и операция."
    assert response.evidence_notes[0].source_ids == ["D1"]


def test_adaptive_schema_declares_information_fields() -> None:
    schema = AdaptiveInformationResponse.model_json_schema()
    assert "без советов" in schema["properties"]["docs_answer"]["description"]
    assert schema["properties"]["recommended_questions"]["description"] == "Всегда пустой список"


def test_adaptive_prompt_defines_search_not_advisor_role() -> None:
    assert "Ты не инженер техподдержки, не советчик" in ADAPTIVE_INFORMATION_PROMPT
    assert "recommended_questions: всегда []" in ADAPTIVE_INFORMATION_PROMPT
    assert "Не давай рекомендаций" in ADAPTIVE_INFORMATION_PROMPT
    assert "Раскрывай все относящиеся" in ADAPTIVE_INFORMATION_PROMPT
    assert "объедини их в один порядок действий" in ADAPTIVE_INFORMATION_PROMPT


if __name__ == "__main__":
    test_formatter_separates_docs_from_historical_outcomes()
    test_adaptive_role_discards_advice_fields_and_draft()
    test_api_answer_uses_search_digest_and_source_footer()
    test_information_guard_reports_missing_evidence_without_advice()
    test_adaptive_schema_preserves_provider_ticket_aliases()
    test_adaptive_schema_declares_information_fields()
    test_adaptive_prompt_defines_search_not_advisor_role()
    print("Adaptive information-mode checks: OK")