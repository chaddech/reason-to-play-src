# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""Command-line analysis for gameplay and human replay behavior metrics.

Examples
--------
Summarize one or more replay traces::

    python -m src.llm_eval.behavior_analysis summarize out/*.replay.json.gz

Compare a persistent-state run with its matched reset-state control::

    python -m src.llm_eval.behavior_analysis compare persistent.replay.json.gz reset.replay.json.gz

Compare model and human discovery cohorts::

    python -m src.llm_eval.behavior_analysis cohort \
        --model out/model/*.replay.json.gz --human out/human/*.replay.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Sequence

from .shared.behavior_metrics import (
    DEFAULT_DELAYED_USE_MIN_STEPS,
    compare_behavior_summaries,
    compare_first_win_cohorts,
    summarize_behavior,
)


def _load_json(path: str | Path) -> dict:
    """Load a replay without expanding its unused, delta-encoded states."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def summary_for_replay(
    replay: dict,
    *,
    delayed_use_min_steps: int = DEFAULT_DELAYED_USE_MIN_STEPS,
    level_count: int | None = None,
    progress_horizon_decisions: int | None = None,
) -> dict:
    """Recompute a behavior summary from either a new or legacy replay."""
    meta = replay.get("meta") or {}
    steps = replay.get("steps") or []
    start_level = int(replay.get("start_level", meta.get("start_level", 0)))
    if level_count is None:
        raw_level_count = meta.get("level_count")
        if raw_level_count is not None:
            level_count = int(raw_level_count)
        else:
            observed_levels = [
                int(step.get("level", start_level))
                for step in steps
                if step.get("level") is not None
            ]
            level_count = (
                max(observed_levels) - start_level + 1 if observed_levels else 1
            )

    consecutive_wins_required = meta.get("consecutive_wins_required")
    if consecutive_wins_required is None:
        consecutive_wins_required = 1

    return summarize_behavior(
        steps,
        total_frames=replay.get("total_frames"),
        start_level=start_level,
        level_count=level_count,
        final_level=replay.get("final_level"),
        consecutive_wins_required=int(consecutive_wins_required),
        delayed_use_min_steps=delayed_use_min_steps,
        progress_horizon_decisions=progress_horizon_decisions,
    )


def _run_summary(path: str, args: argparse.Namespace) -> dict:
    return {
        "path": str(Path(path).resolve()),
        "summary": summary_for_replay(
            _load_json(path),
            delayed_use_min_steps=args.delay_steps,
            level_count=args.level_count,
            progress_horizon_decisions=args.progress_horizon,
        ),
    }


def _write_result(result: dict, output: str | None) -> None:
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered)
    else:
        print(rendered, end="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze Reason-to-Play model and human replay traces."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--delay-steps",
            type=int,
            default=DEFAULT_DELAYED_USE_MIN_STEPS,
            help="Minimum decision gap for the delayed-use proxy (default: 50).",
        )
        subparser.add_argument(
            "--level-count",
            type=int,
            default=None,
            help="Override the number of playable levels in every input replay.",
        )
        subparser.add_argument(
            "--progress-horizon",
            type=int,
            default=None,
            help=(
                "Blocked-curriculum decision horizon; use 1600 to match the "
                "paper's capability plot."
            ),
        )
        subparser.add_argument("--output", help="Write JSON here instead of stdout.")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Compute metrics for replay traces."
    )
    summarize_parser.add_argument("replays", nargs="+")
    add_common(summarize_parser)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare one treatment replay with a matched control."
    )
    compare_parser.add_argument("treatment")
    compare_parser.add_argument("control")
    add_common(compare_parser)

    cohort_parser = subparsers.add_parser(
        "cohort", help="Compare model and human level-discovery distributions."
    )
    cohort_parser.add_argument("--model", nargs="+", required=True)
    cohort_parser.add_argument("--human", nargs="+", required=True)
    add_common(cohort_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "summarize":
        result = {"runs": [_run_summary(path, args) for path in args.replays]}
    elif args.command == "compare":
        treatment = _run_summary(args.treatment, args)
        control = _run_summary(args.control, args)
        result = {
            "treatment": treatment,
            "control": control,
            "benefit": compare_behavior_summaries(
                treatment["summary"], control["summary"]
            ),
            "direction": "positive values favor treatment",
        }
    else:
        model = [_run_summary(path, args) for path in args.model]
        human = [_run_summary(path, args) for path in args.human]
        result = {
            "model": model,
            "human": human,
            "comparison": compare_first_win_cohorts(
                [run["summary"] for run in model],
                [run["summary"] for run in human],
            ),
        }

    _write_result(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
