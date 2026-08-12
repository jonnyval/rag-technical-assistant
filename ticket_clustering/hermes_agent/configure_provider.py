"""Apply a named project LLM profile to the local Hermes configuration."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_CONFIG = Path(__file__).resolve().with_name("config.yaml")
HERMES_ROOT = Path(os.environ["LOCALAPPDATA"]) / "hermes" / "hermes-agent"
ARTICLE_TOOLS = [
    "list_article_candidates",
    "get_article_seed",
    "create_research_seed",
    "search_documentation",
    "search_historical_tickets",
    "search_dual_retriever",
    "read_search_results",
    "get_cluster_tickets",
    "save_article_draft",
]
SEARCH_TOOLS = [
    "search_documentation",
    "search_historical_tickets",
    "search_dual_retriever",
    "read_search_results",
    "get_cluster_tickets",
]


def load_catalog() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    config = yaml.safe_load(PROJECT_CONFIG.read_text(encoding="utf-8")) or {}
    llm = config.get("llm") or {}
    profiles = llm.get("profiles") or {}
    if not isinstance(profiles, dict) or not profiles:
        raise SystemExit(f"No llm.profiles configured in {PROJECT_CONFIG}")
    return config, profiles


def resolve_profile(name: str | None = None) -> tuple[str, dict[str, Any]]:
    config, profiles = load_catalog()
    selected = str(name or (config.get("llm") or {}).get("active") or "").strip()
    if selected not in profiles:
        choices = ", ".join(sorted(profiles))
        raise SystemExit(f"Unknown Hermes LLM profile '{selected}'. Available: {choices}")
    profile = dict(profiles[selected] or {})
    required = ("type", "base_url", "model", "context_length", "max_tokens")
    missing = [key for key in required if profile.get(key) in (None, "")]
    if missing:
        raise SystemExit(f"Profile '{selected}' is missing: {', '.join(missing)}")
    return selected, profile


def _project_env() -> dict[str, str]:
    return {key: str(value) for key, value in dotenv_values(PROJECT_ROOT / ".env").items() if value is not None}


def _apply_ollama(config: dict[str, Any], name: str, profile: dict[str, Any]) -> None:
    base_url = str(profile["base_url"]).rstrip("/")
    model = str(profile["model"])
    config.setdefault("providers", {})[name] = {
        "api": base_url,
        "default_model": model,
        "transport": "chat_completions",
    }
    config["model"] = {
        **config.get("model", {}),
        "default": model,
        "provider": "ollama",
        "base_url": base_url,
        "max_tokens": int(profile["max_tokens"]),
        "context_length": int(profile["context_length"]),
        "ollama_num_ctx": int(profile["context_length"]),
    }


def _apply_yandex(config: dict[str, Any], name: str, profile: dict[str, Any]) -> None:
    values = _project_env()
    key_name = str(profile.get("api_key_env") or "YANDEX_API_KEY")
    folder_name = str(profile.get("folder_id_env") or "YANDEX_FOLDER_ID")
    api_key = values.get(key_name, "").strip()
    folder_id = values.get(folder_name, "").strip()
    if not api_key or not folder_id:
        raise SystemExit(f"{key_name} or {folder_name} is missing in {PROJECT_ROOT / '.env'}")

    base_url = str(profile["base_url"]).rstrip("/")
    configured_model = str(profile["model"])
    model = configured_model if configured_model.startswith("gpt://") else f"gpt://{folder_id}/{configured_model}"
    pool_provider = f"custom:{name}"
    config.setdefault("providers", {})[name] = {
        "api": base_url,
        "default_model": model,
        "transport": "chat_completions",
    }
    config["model"] = {
        **config.get("model", {}),
        "default": model,
        "provider": pool_provider,
        "base_url": base_url,
        "max_tokens": int(profile["max_tokens"]),
        "context_length": int(profile["context_length"]),
    }
    config["model"].pop("ollama_num_ctx", None)
    config.setdefault("credential_pool_strategies", {})[pool_provider] = "round_robin"

    from agent.credential_pool import (  # noqa: PLC0415
        AUTH_TYPE_API_KEY,
        SOURCE_MANUAL,
        PooledCredential,
        load_pool,
    )

    pool = load_pool(pool_provider)
    if not any(entry.runtime_api_key == api_key for entry in pool.entries()):
        pool.add_entry(
            PooledCredential(
                provider=pool_provider,
                id=uuid.uuid4().hex[:6],
                label=f"project-{name}",
                auth_type=AUTH_TYPE_API_KEY,
                priority=0,
                source=SOURCE_MANUAL,
                access_token=api_key,
                base_url=base_url,
            )
        )


def apply_profile(name: str | None = None) -> tuple[str, dict[str, Any]]:
    selected, profile = resolve_profile(name)
    project_config, _ = load_catalog()
    mcp_config = project_config.get("mcp") or {}
    api_search_url = str(mcp_config.get("api_search_url") or "http://127.0.0.1:8000/hermes/mcp")
    standalone_url = str(mcp_config.get("standalone_url") or "http://127.0.0.1:8765/mcp")
    sys.path.insert(0, str(HERMES_ROOT))
    from hermes_cli.config import load_config, save_config  # noqa: PLC0415

    config = load_config()
    profile_type = str(profile["type"]).lower()
    if profile_type == "ollama":
        _apply_ollama(config, selected, profile)
    elif profile_type == "yandex":
        _apply_yandex(config, selected, profile)
    else:
        raise SystemExit(f"Unsupported profile type: {profile_type}")

    config.setdefault("agent", {})["reasoning_effort"] = str(profile.get("reasoning_effort") or "none")
    config.setdefault("compression", {}).update(
        {"enabled": True, "threshold": 0.8, "threshold_tokens": None, "proactive_prune_tokens": 0}
    )
    config.setdefault("auxiliary", {}).setdefault("title_generation", {})["enabled"] = False
    reglab_articles = config.setdefault("mcp_servers", {}).setdefault("reglab_articles", {})
    reglab_articles.update(
        {
            "url": standalone_url,
            "timeout": 600,
            "connect_timeout": 120,
            "idle_timeout_seconds": 0,
            "supports_parallel_tool_calls": False,
            "tools": {"include": ARTICLE_TOOLS},
        }
    )
    reglab_search = config.setdefault("mcp_servers", {}).setdefault("reglab_search", {})
    reglab_search.update(
        {
            "url": api_search_url,
            "timeout": 600,
            "connect_timeout": 120,
            "idle_timeout_seconds": 0,
            "supports_parallel_tool_calls": False,
            "tools": {"include": SEARCH_TOOLS},
        }
    )
    reglab_search_local = config.setdefault("mcp_servers", {}).setdefault("reglab_search_local", {})
    reglab_search_local.update(
        {
            "url": standalone_url,
            "timeout": 600,
            "connect_timeout": 120,
            "idle_timeout_seconds": 0,
            "supports_parallel_tool_calls": False,
            "tools": {"include": SEARCH_TOOLS},
        }
    )
    save_config(config)
    return selected, profile


def check_profile(profile: dict[str, Any]) -> tuple[bool, str]:
    if str(profile["type"]).lower() != "ollama":
        return True, "remote authenticated profile; connectivity is checked on first inference"
    base_url = str(profile["base_url"]).rstrip("/")
    request = urllib.request.Request(f"{base_url}/models", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        return False, f"{type(error).__name__}: {error}"
    models = {str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)}
    expected = str(profile["model"])
    return expected in models, f"model {'found' if expected in models else 'not found'}; endpoint returned {len(models)} models"


def set_active(name: str) -> None:
    _, profiles = load_catalog()
    if name not in profiles:
        raise SystemExit(f"Unknown profile: {name}")
    lines = PROJECT_CONFIG.read_text(encoding="utf-8").splitlines(keepends=True)
    inside_llm = False
    replaced = False
    for index, line in enumerate(lines):
        if line.startswith("llm:"):
            inside_llm = True
            continue
        if inside_llm and line and not line[0].isspace():
            break
        if inside_llm and line.lstrip().startswith("active:"):
            newline = "\r\n" if line.endswith("\r\n") else "\n"
            indent = line[: len(line) - len(line.lstrip())]
            lines[index] = f'{indent}active: "{name}"{newline}'
            replaced = True
            break
    if not replaced:
        raise SystemExit("Could not locate llm.active in project config")
    PROJECT_CONFIG.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="Apply this profile instead of llm.active")
    parser.add_argument("--set-active", action="store_true", help="Persist --profile as llm.active")
    parser.add_argument("--list", action="store_true", help="List configured profiles and exit")
    parser.add_argument("--check", action="store_true", help="Check the selected endpoint/model")
    args = parser.parse_args()

    config, profiles = load_catalog()
    active = str((config.get("llm") or {}).get("active") or "")
    if args.list:
        for name, profile in profiles.items():
            marker = "*" if name == active else " "
            print(f"{marker} {name}: {profile.get('model')} @ {profile.get('base_url')}")
        return 0
    if args.set_active and not args.profile:
        raise SystemExit("--set-active requires --profile")
    if args.set_active:
        set_active(args.profile)

    selected, profile = apply_profile(args.profile)
    print(
        f"Hermes profile applied: {selected}; model={profile['model']}; "
        f"base_url={profile['base_url']}; context_length={profile['context_length']}"
    )
    if args.check:
        ok, detail = check_profile(profile)
        print(f"Check: {'OK' if ok else 'FAILED'} — {detail}")
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
