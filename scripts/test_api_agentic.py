"""Fast routing checks for the OpenAI-compatible Agentic RAG model."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import api_server


def test_agentic_model_is_exposed_and_reuses_deep_profile() -> None:
    payload = asyncio.run(api_server.list_models())
    model_ids = {item["id"] for item in payload["data"]}
    assert api_server.AGENTIC_MODEL_ID in model_ids
    assert api_server.MODEL_PROFILES[api_server.AGENTIC_MODEL_ID] == "deep"


def test_agentic_pipeline_is_cached_around_shared_engine() -> None:
    engine = SimpleNamespace(name="shared-deep")
    state = SimpleNamespace(agentic_rag=None, agentic_rag_lock=asyncio.Lock())
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    first = asyncio.run(api_server._get_agentic_rag(request, engine))
    second = asyncio.run(api_server._get_agentic_rag(request, engine))
    assert first is second
    assert first.engine is engine


def test_chat_completion_routes_agentic_model_and_uses_latest_question() -> None:
    calls = []

    class FakeAgenticRAG:
        def run(self, query: str):
            calls.append(query)
            answer = SimpleNamespace(
                draft_private_comment="Agentic answer",
                docs_answer="",
                doc_sources=[],
                ticket_sources=[],
            )
            return SimpleNamespace(
                answer=answer,
                stop_reason="enough_evidence",
                rounds_used=2,
                tool_calls_used=3,
                elapsed_seconds=1.25,
            )

    deep_engine = SimpleNamespace()
    state = SimpleNamespace(
        rag_engine=deep_engine,
        rag_engines={"deep": deep_engine},
        engine_locks={"deep": asyncio.Lock()},
        agentic_rag=FakeAgenticRAG(),
        agentic_rag_lock=asyncio.Lock(),
        inference_semaphore=asyncio.Semaphore(1),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    payload = api_server.ChatCompletionRequest(
        model=api_server.AGENTIC_MODEL_ID,
        messages=[
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "latest question"},
        ],
        stream=False,
    )
    response = asyncio.run(api_server.chat_completions(request, payload))
    assert calls == ["latest question"]
    assert response["model"] == api_server.AGENTIC_MODEL_ID
    assert "Agentic answer" in response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    test_agentic_model_is_exposed_and_reuses_deep_profile()
    test_agentic_pipeline_is_cached_around_shared_engine()
    test_chat_completion_routes_agentic_model_and_uses_latest_question()
    print("Agentic API routing checks: OK")