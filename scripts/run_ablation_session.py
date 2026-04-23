#!/usr/bin/env python3
"""Run a single ablation-tagged evaluation session.

Delegates to ``scripts.run_eval_session`` after applying ablation overrides
selected via CLI. All overrides map to flags in ``PipelineConfig`` and
``RouterConfig`` that default to production behavior, so the baseline path
is preserved when ``--ablation baseline`` (default) is selected.

Supported ablations (see docs/EXPERIMENTS.md):

- ``baseline``                 : no overrides (sanity check).
- ``hybrid_force_retrieval``   : isolate the conditional retrieval gate.
- ``hybrid_disable_critic``    : isolate the value of the critic in hybrid_fast.
- ``non_rag_enable_retrieval`` : separate grounding from coordination.
- ``disable_critic_global``    : turn off the critic on every architecture.
- ``router_heuristic_only``    : ignore any trained router classifier.
- ``router_classifier_only``   : require the trained router classifier.

Outputs go into a parallel results tree so the baseline session is not
overwritten. The ablation label is recorded in ``meta.ablation`` and in the
``ablation.*`` telemetry keys on each record (populated by BasePipeline).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

# Make the sibling ``run_eval_session`` module importable without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval_session  # noqa: E402


ABLATIONS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "hybrid_force_retrieval": {"pipeline": {"hybrid_force_retrieval": True}},
    "hybrid_disable_critic": {"pipeline": {"hybrid_disable_critic": True}},
    "non_rag_enable_retrieval": {"pipeline": {"non_rag_enable_retrieval": True}},
    "disable_critic_global": {"pipeline": {"disable_critic_global": True}},
    "router_heuristic_only": {"router": {"use_heuristic_only": True}},
    "router_classifier_only": {"router": {"use_classifier_only": True}},
}


def _merge_overrides(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_overrides(merged[key], value)
        else:
            merged[key] = value
    return merged


def _materialize_config(
    base_config_path: str,
    overrides: dict[str, Any],
    ablation_tag: str,
    scratch_dir: Path,
) -> Path:
    base = yaml.safe_load(Path(base_config_path).read_text(encoding="utf-8")) or {}
    merged = _merge_overrides(base, overrides)
    pipeline = merged.setdefault("pipeline", {})
    pipeline["ablation_tag"] = ablation_tag
    scratch_dir.mkdir(parents=True, exist_ok=True)
    out_path = scratch_dir / f"ablation_{ablation_tag}.yaml"
    out_path.write_text(yaml.safe_dump(merged, sort_keys=True), encoding="utf-8")
    return out_path


def _build_session_args(raw: argparse.Namespace, config_path: Path, out_path: Path) -> argparse.Namespace:
    """Return a Namespace matching ``run_eval_session.main``'s parser contract."""
    return SimpleNamespace(
        dataset_name=raw.dataset_name,
        config=str(config_path),
        source=raw.source,
        split=raw.split,
        architecture=raw.architecture,
        limit=raw.limit,
        corpus=raw.corpus,
        index_path=raw.index_path,
        index_out=raw.index_out,
        progress_every=raw.progress_every,
        resume=raw.resume,
        max_example_retries=raw.max_example_retries,
        max_5xx_retries=raw.max_5xx_retries,
        retry_backoff_base=raw.retry_backoff_base,
        retry_backoff_max=raw.retry_backoff_max,
        checkpoint_every=raw.checkpoint_every,
        allow_partial=raw.allow_partial,
        example_timeout=raw.example_timeout,
        out=str(out_path),
    )


async def _invoke_session(args: argparse.Namespace) -> dict[str, Any]:
    payload = await run_eval_session._run(args)  # noqa: SLF001
    out_path = Path(args.out)
    run_eval_session._checkpoint_write(out_path, payload)  # noqa: SLF001
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_name")
    parser.add_argument("--config", required=True, help="Base system config (YAML).")
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default="baseline")
    parser.add_argument("--out-dir", default="artifacts/ablations", help="Directory for ablation outputs.")
    parser.add_argument("--tag-suffix", default="", help="Extra label appended to filenames.")
    parser.add_argument("--source", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--corpus", default=None)
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--index-out", default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--max-example-retries", type=int, default=6)
    parser.add_argument("--max-5xx-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-base", type=float, default=2.0)
    parser.add_argument("--retry-backoff-max", type=float, default=45.0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--allow-partial", dest="allow_partial", action="store_true", default=True)
    parser.add_argument("--no-allow-partial", dest="allow_partial", action="store_false")
    parser.add_argument("--example-timeout", type=float, default=300.0)
    raw = parser.parse_args()

    tag = raw.ablation + (f"_{raw.tag_suffix}" if raw.tag_suffix else "")
    out_dir = Path(raw.out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{raw.dataset_name.lower()}_{raw.architecture}.json"
    config_scratch = out_dir / "configs"
    overrides = ABLATIONS[raw.ablation]
    config_path = _materialize_config(raw.config, overrides, ablation_tag=tag, scratch_dir=config_scratch)

    session_args = _build_session_args(raw, config_path, result_path)
    started = time.time()
    payload = asyncio.run(_invoke_session(session_args))
    duration_s = round(time.time() - started, 3)

    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    meta["ablation"] = tag
    meta["ablation_config_path"] = str(config_path)
    if isinstance(payload, dict):
        payload["meta"] = meta
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "ablation": tag,
                "architecture": raw.architecture,
                "dataset": raw.dataset_name,
                "out": str(result_path),
                "duration_s": duration_s,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
