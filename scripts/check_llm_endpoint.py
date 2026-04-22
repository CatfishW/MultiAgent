#!/usr/bin/env python3
"""Pre-flight liveness check for the experiments LLM endpoint.

Reads the system config, queries `/models`, and issues tiny JSON-mode prompts
against the declared text + mllm models. Exits non-zero on any failure so
screen sessions can refuse to launch into a dead endpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import yaml


def _load(cfg: Path) -> dict:
    return yaml.safe_load(cfg.read_text()) or {}


def _check_models(base_url: str, timeout: float) -> list[str]:
    url = base_url.rstrip("/") + "/models"
    r = httpx.get(url, timeout=timeout, verify=False)
    r.raise_for_status()
    data = r.json().get("data", [])
    return [m["id"] for m in data]


def _probe_chat(base_url: str, model: str, timeout: float) -> tuple[bool, str]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return exactly {\"ok\": true} as JSON."},
            {"role": "user", "content": "ping"},
        ],
        "temperature": 0.0,
        "max_tokens": 32,
    }
    try:
        r = httpx.post(url, json=payload, timeout=timeout, verify=False)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"].get("content", "")
        return True, (content or "")[:120]
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/system.experiments.yaml")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--probe", action="store_true", help="issue a tiny chat completion per model")
    args = ap.parse_args()

    cfg = _load(Path(args.config))
    endpoints = (cfg or {}).get("endpoints") or {}
    failed = 0

    for name, ep in endpoints.items():
        base = ep.get("base_url")
        default_model = ep.get("default_model")
        fallback_model = ep.get("fallback_model")
        print(f"[endpoint:{name}] {base}")
        try:
            models = _check_models(base, args.timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL /models: {exc}")
            failed += 1
            continue
        print(f"  models ({len(models)}): {models}")

        targets = [m for m in [default_model, fallback_model] if m]
        for m in targets:
            if m not in models:
                print(f"  FAIL default_model not served: {m}")
                failed += 1
                continue
            if not args.probe:
                print(f"  OK listed: {m}")
                continue
            t0 = time.time()
            ok, body = _probe_chat(base, m, args.timeout)
            dt = time.time() - t0
            print(f"  {'OK' if ok else 'FAIL'} probe {m} ({dt:.2f}s) :: {body}")
            if not ok:
                failed += 1

    if failed:
        print(f"FAILED checks: {failed}", file=sys.stderr)
        return 1
    print("All endpoint checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
