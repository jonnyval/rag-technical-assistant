"""Validate Hermes Groq credentials without printing secret values."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove-invalid", action="store_true")
    parser.add_argument(
        "--hermes-root",
        type=Path,
        default=Path(os.environ["LOCALAPPDATA"]) / "hermes" / "hermes-agent",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.hermes_root.resolve()))
    from agent.credential_pool import load_pool  # noqa: PLC0415

    pool = load_pool("custom:groq")
    entries = pool.entries()
    invalid: list[int] = []
    for index, entry in enumerate(entries, start=1):
        try:
            response = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {entry.runtime_api_key}"},
                timeout=20,
            )
            status = response.status_code
        except requests.RequestException:
            status = 0
        valid = status not in {401, 403}
        print(f"key #{index}: status={status or 'network_error'}, valid={valid}")
        if not valid:
            invalid.append(index - 1)

    if args.remove_invalid:
        for zero_based_index in reversed(invalid):
            pool.remove_index(zero_based_index)
        print(f"removed={len(invalid)}, remaining={len(pool.entries())}")
    return 0 if entries and len(invalid) < len(entries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
