"""Configure Hermes to use the project's Groq keys with round-robin rotation.

The script never prints key values.  It reads ``GROQ_API_KEYS`` from the
project ``.env`` and stores missing keys in Hermes' own credential pool.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

from dotenv import dotenv_values


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "qwen/qwen3.6-27b"
DEFAULT_MAX_TOKENS = 600
POOL_PROVIDER = "custom:groq"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hermes-root",
        type=Path,
        default=Path(os.environ["LOCALAPPDATA"]) / "hermes" / "hermes-agent",
    )
    parser.add_argument("--project-env", type=Path, default=Path(".env"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return parser.parse_args()


def _load_project_keys(env_path: Path) -> list[str]:
    raw = str(dotenv_values(env_path).get("GROQ_API_KEYS") or "")
    keys = list(dict.fromkeys(key.strip() for key in raw.split(",") if key.strip()))
    if not keys:
        raise SystemExit(f"GROQ_API_KEYS is empty or missing in {env_path}")
    return keys


def main() -> int:
    args = _parse_args()
    hermes_root = args.hermes_root.resolve()
    if not (hermes_root / "hermes_cli").is_dir():
        raise SystemExit(f"Hermes installation not found: {hermes_root}")

    sys.path.insert(0, str(hermes_root))
    from agent.credential_pool import (  # noqa: PLC0415
        AUTH_TYPE_API_KEY,
        SOURCE_MANUAL,
        PooledCredential,
        load_pool,
    )
    from hermes_cli.config import load_config, save_config  # noqa: PLC0415

    project_keys = _load_project_keys(args.project_env.resolve())

    config = load_config()
    providers = config.get("providers")
    if not isinstance(providers, dict):
        providers = {}
    groq = providers.get("groq")
    if not isinstance(groq, dict):
        groq = {}
    groq.update(
        {
            "api": GROQ_BASE_URL,
            "default_model": args.model,
            "transport": "chat_completions",
        }
    )
    # Do not put a key in config.yaml: all credentials live in the pool.
    groq.pop("api_key", None)
    groq.pop("key_env", None)
    providers["groq"] = groq
    config["providers"] = providers

    mcp_servers = config.get("mcp_servers")
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}
    reglab_articles = mcp_servers.get("reglab_articles")
    if isinstance(reglab_articles, dict):
        # Retrieval owns heavyweight Torch/Qdrant clients. Keep one stdio
        # process alive and never dispatch multiple searches concurrently.
        reglab_articles["timeout"] = 600
        reglab_articles["connect_timeout"] = 120
        reglab_articles["idle_timeout_seconds"] = 0
        reglab_articles["supports_parallel_tool_calls"] = False
        reglab_articles["url"] = "http://127.0.0.1:8765/mcp"
        reglab_articles.pop("command", None)
        reglab_articles.pop("args", None)
        reglab_articles.pop("env", None)
        mcp_servers["reglab_articles"] = reglab_articles
        config["mcp_servers"] = mcp_servers

    model = config.get("model")
    if not isinstance(model, dict):
        model = {}
    model.update(
        {
            "default": args.model,
            "provider": POOL_PROVIDER,
            "base_url": GROQ_BASE_URL,
            # Groq's on-demand tier accounts for the requested output budget
            # in TPM checks. Hermes otherwise uses the model's very large
            # native output limit and even a tiny prompt is rejected with 413.
            "max_tokens": args.max_tokens,
            "context_length": 131072,
        }
    )
    config["model"] = model

    agent = config.get("agent")
    if not isinstance(agent, dict):
        agent = {}
    # Hermes' generic custom-provider profile otherwise emits Ollama/GLM-only
    # `reasoning_effort` and `think` fields, which Groq rejects for Llama.
    agent["reasoning_effort"] = ""
    config["agent"] = agent

    compression = config.get("compression")
    if not isinstance(compression, dict):
        compression = {}
    compression.update(
        {
            "enabled": False,
            "threshold": 0.9,
            "threshold_tokens": None,
            "proactive_prune_tokens": 5500,
            "proactive_prune_min_result_chars": 300,
            "proactive_prune_min_reclaim_tokens": 0,
            "protect_last_n": 4,
        }
    )
    config["compression"] = compression

    auxiliary = config.get("auxiliary")
    if not isinstance(auxiliary, dict):
        auxiliary = {}
    title_generation = auxiliary.get("title_generation")
    if not isinstance(title_generation, dict):
        title_generation = {}
    title_generation["enabled"] = False
    auxiliary["title_generation"] = title_generation
    config["auxiliary"] = auxiliary

    strategies = config.get("credential_pool_strategies")
    if not isinstance(strategies, dict):
        strategies = {}
    strategies[POOL_PROVIDER] = "round_robin"
    config["credential_pool_strategies"] = strategies
    save_config(config)

    pool = load_pool(POOL_PROVIDER)
    existing_keys = {entry.runtime_api_key for entry in pool.entries()}
    added = 0
    for index, key in enumerate(project_keys, start=1):
        if key in existing_keys:
            continue
        pool.add_entry(
            PooledCredential(
                provider=POOL_PROVIDER,
                id=uuid.uuid4().hex[:6],
                label=f"project-groq-{index}",
                auth_type=AUTH_TYPE_API_KEY,
                priority=0,
                source=SOURCE_MANUAL,
                access_token=key,
                base_url=GROQ_BASE_URL,
            )
        )
        existing_keys.add(key)
        added += 1

    print(
        "Hermes Groq configured: "
        f"model={args.model}, keys={len(project_keys)}, added={added}, "
        f"strategy=round_robin, max_tokens={args.max_tokens}, reasoning=provider_default"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
