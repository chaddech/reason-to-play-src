# Copyright (c) 2026 Botos Csaba. MIT License. See LICENSE for details.
"""Behavioral and memory-oriented metrics for gameplay replay traces.

The metrics in this module are deliberately computed from the common replay
schema used by both generative gameplay and human replay.  They do not assume
that there is a single correct action sequence.

Some memory measures are necessarily operational proxies.  In particular,
encountering the same engine interaction twice does not prove that an agent
remembered it.  The replay-level proxies are useful diagnostics; causal memory
evidence comes from comparing matched persistent-state and state-reset runs.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Iterable, Sequence


METRICS_SCHEMA_VERSION = 1
DEFAULT_DELAYED_USE_MIN_STEPS = 50


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def _rate(values: Iterable[bool]) -> float | None:
    values = list(values)
    return sum(bool(value) for value in values) / len(values) if values else None


def _real_steps(steps: Sequence[dict]) -> list[dict]:
    """Return actual environment decisions, excluding markers and crashes."""
    result = []
    for step in steps:
        action = str(step.get("action", ""))
        if not action or action.startswith("_") or step.get("crash_reason"):
            continue
        result.append(step)
    return result


def _interaction_key(value) -> tuple[str, str, str] | None:
    if isinstance(value, dict):
        parts = (
            value.get("actor_color"),
            value.get("effect"),
            value.get("actee_color"),
        )
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        parts = tuple(value)
    else:
        return None
    if any(part is None for part in parts):
        return None
    return tuple(str(part) for part in parts)


def _attempt_rows(real_steps: Sequence[dict]) -> list[dict]:
    attempts: dict[tuple[int, int], dict] = {}
    for decision, step in enumerate(real_steps, start=1):
        level = int(step.get("level", 0))
        attempt = int(step.get("attempt", 0))
        key = (level, attempt)
        if key not in attempts:
            attempts[key] = {
                "level": level,
                "attempt": attempt,
                "start_decision": decision,
                "end_decision": decision,
                "decisions": 0,
                "reward": 0.0,
                "won": False,
                "lost": False,
                "timed_out": False,
            }
        row = attempts[key]
        row["end_decision"] = decision
        row["decisions"] += 1
        row["reward"] += float(step.get("reward", 0) or 0)
        row["won"] = row["won"] or bool(step.get("won"))
        row["lost"] = row["lost"] or bool(step.get("lose"))
        row["timed_out"] = row["timed_out"] or bool(step.get("timeout"))
    return list(attempts.values())


def _level_metrics(attempts: Sequence[dict], consecutive_wins_required: int) -> list[dict]:
    by_level: dict[int, list[dict]] = defaultdict(list)
    for attempt in attempts:
        by_level[attempt["level"]].append(attempt)

    result = []
    for level, rows in sorted(by_level.items()):
        rows.sort(key=lambda row: row["start_decision"])
        level_start = rows[0]["start_decision"] - 1
        winning_rows = [row for row in rows if row["won"]]
        first_win = winning_rows[0] if winning_rows else None

        consecutive = 0
        mastery_row = None
        for row in rows:
            if row["won"]:
                consecutive += 1
                if consecutive >= consecutive_wins_required:
                    mastery_row = row
                    break
            else:
                consecutive = 0

        first_win_length = first_win["decisions"] if first_win else None
        later_win_lengths = [row["decisions"] for row in winning_rows[1:]]
        later_win_median = (
            statistics.median(later_win_lengths) if later_win_lengths else None
        )
        savings_steps = (
            first_win_length - later_win_median
            if first_win_length is not None and later_win_median is not None
            else None
        )
        savings_fraction = (
            savings_steps / first_win_length
            if savings_steps is not None and first_win_length
            else None
        )

        result.append(
            {
                "level": level,
                "attempts": len(rows),
                "wins": len(winning_rows),
                "losses": sum(bool(row["lost"]) for row in rows),
                "decisions": sum(row["decisions"] for row in rows),
                "decisions_to_first_win": (
                    first_win["end_decision"] - level_start if first_win else None
                ),
                "cumulative_decision_to_first_win": (
                    first_win["end_decision"] if first_win else None
                ),
                "decisions_to_mastery": (
                    mastery_row["end_decision"] - level_start if mastery_row else None
                ),
                "first_winning_attempt_decisions": first_win_length,
                "later_winning_attempt_median_decisions": later_win_median,
                "subsequent_winning_attempt_decisions": later_win_lengths,
                "relearning_savings_steps": savings_steps,
                "relearning_savings_fraction": savings_fraction,
            }
        )
    return result


def _blocked_curriculum_view(
    attempts: Sequence[dict],
    *,
    start_level: int,
    level_count: int | None,
    consecutive_wins_required: int,
    decision_horizon: int,
) -> tuple[list[dict], dict]:
    """Project fixed-budget traces onto the paper's blocked curriculum.

    Attempts after mastery on a level are omitted.  If a level never reaches
    the consecutive-win criterion, all later levels are censored.  Attempt
    decision indexes are rebased after omissions so cumulative discovery and
    execution counts describe the counterfactual blocked trajectory.
    """
    by_level: dict[int, list[dict]] = defaultdict(list)
    for attempt in attempts:
        by_level[attempt["level"]].append(attempt)
    for rows in by_level.values():
        rows.sort(key=lambda row: row["start_decision"])

    if level_count is not None:
        expected_levels = list(range(start_level, start_level + level_count))
    else:
        expected_levels = sorted(
            level for level in by_level if level >= start_level
        )

    included: list[dict] = []
    curve: list[list[int]] = [[0, start_level]]
    cumulative_decisions = 0
    mastered_levels = 0
    stalled_level = None
    exhausted_horizon = False

    for level in expected_levels:
        rows = by_level.get(level, [])
        if not rows:
            stalled_level = level
            break

        consecutive = 0
        mastered = False
        for source_row in rows:
            remaining = decision_horizon - cumulative_decisions
            if remaining <= 0:
                exhausted_horizon = True
                break

            kept_decisions = min(int(source_row["decisions"]), remaining)
            row = dict(source_row)
            row["start_decision"] = cumulative_decisions + 1
            row["end_decision"] = cumulative_decisions + kept_decisions
            row["decisions"] = kept_decisions
            if kept_decisions < int(source_row["decisions"]):
                # The terminal event lies outside the analysis horizon.
                row["won"] = False
                row["lost"] = False
                row["timed_out"] = False
                exhausted_horizon = True
            included.append(row)
            cumulative_decisions += kept_decisions

            if exhausted_horizon:
                break
            if row["won"]:
                consecutive += 1
                if consecutive >= consecutive_wins_required:
                    mastered = True
                    mastered_levels += 1
                    curve.append([cumulative_decisions, level + 1])
                    break
            else:
                consecutive = 0

        if not mastered:
            stalled_level = level
            break

    final_level = start_level + mastered_levels
    previous_time = 0
    current_level = start_level
    area = 0.0
    for time, level in curve[1:]:
        time = min(time, decision_horizon)
        area += (time - previous_time) * (current_level - start_level)
        current_level = level
        previous_time = time
    area += (decision_horizon - previous_time) * (current_level - start_level)
    mean_progress = area / decision_horizon if decision_horizon else 0.0
    normalized = (
        mean_progress / level_count
        if level_count is not None and level_count > 0
        else None
    )
    censored_levels = (
        [level for level in expected_levels if level > stalled_level]
        if stalled_level is not None
        else []
    )
    return included, {
        "consecutive_wins_required": consecutive_wins_required,
        "decision_horizon": decision_horizon,
        "source_decisions_used": cumulative_decisions,
        "mastered_levels": mastered_levels,
        "final_level": final_level,
        "stalled_level": stalled_level,
        "censored_levels": censored_levels,
        "horizon_exhausted": exhausted_horizon,
        "level_progress_auc": area,
        "mean_level_progress": mean_progress,
        "normalized_level_progress_auc": normalized,
        "curve": curve,
    }


def _level_auc(
    real_steps: Sequence[dict],
    *,
    total_frames: int | None,
    start_level: int,
    level_count: int | None,
) -> dict:
    """Integrate level progress over decisions and, when present, frames."""
    # With N playable levels there are N - 1 within-game level advances.  A
    # one-level run therefore has no meaningful normalized progress score.
    max_level_progress = level_count - 1 if level_count is not None else None

    def integrate(points: list[tuple[int, int]], horizon: int) -> dict:
        if not points:
            return {
                "horizon": max(int(horizon), 0),
                "level_progress_auc": 0.0,
                "mean_level_progress": 0.0,
                "normalized_level_progress_auc": (
                    0.0 if max_level_progress and max_level_progress > 0 else None
                ),
                "curve": [[0, start_level]],
            }

        last_observed = max(time for time, _ in points)
        horizon = max(int(horizon), last_observed, 1)
        current_level = start_level
        previous_time = 0
        area = 0.0
        curve = [[0, start_level]]
        for time, level in points:
            time = min(max(time, previous_time), horizon)
            area += (time - previous_time) * max(current_level - start_level, 0)
            if level > current_level:
                current_level = level
                curve.append([time, current_level])
            previous_time = time
        area += (horizon - previous_time) * max(current_level - start_level, 0)
        mean_progress = area / horizon
        return {
            "horizon": horizon,
            "level_progress_auc": area,
            "mean_level_progress": mean_progress,
            "normalized_level_progress_auc": (
                mean_progress / max_level_progress
                if max_level_progress is not None and max_level_progress > 0
                else None
            ),
            "curve": curve,
        }

    decision_points = [
        (decision, int(step.get("level", start_level)))
        for decision, step in enumerate(real_steps, start=1)
    ]
    decision_metrics = integrate(decision_points, len(real_steps))
    decision_metrics["time_unit"] = "decisions"

    has_complete_frame_clock = bool(real_steps) and all(
        step.get("frame", step.get("frame_num")) is not None for step in real_steps
    )
    frame_metrics = None
    if has_complete_frame_clock:
        frame_points = [
            (
                max(int(step.get("frame", step.get("frame_num"))), 0),
                int(step.get("level", start_level)),
            )
            for step in real_steps
        ]
        frame_metrics = integrate(
            frame_points,
            int(total_frames or max(time for time, _ in frame_points)),
        )
        frame_metrics["time_unit"] = "frames"

    decision_metrics["frame_based"] = frame_metrics
    return decision_metrics


def _interaction_metrics(
    real_steps: Sequence[dict],
    attempts: Sequence[dict],
    *,
    delayed_use_min_steps: int,
) -> dict:
    attempt_outcomes = {
        (row["level"], row["attempt"]): row for row in attempts
    }
    occurrences: dict[tuple[str, str, str], list[dict]] = defaultdict(list)

    for decision, step in enumerate(real_steps, start=1):
        seen_this_step: set[tuple[str, str, str]] = set()
        for raw in step.get("interactions", []) or []:
            key = _interaction_key(raw)
            if key is None or key in seen_this_step:
                continue
            seen_this_step.add(key)
            occurrences[key].append(
                {
                    "decision": decision,
                    "frame": step.get("frame", step.get("frame_num")),
                    "level": int(step.get("level", 0)),
                    "attempt": int(step.get("attempt", 0)),
                    "reward": float(step.get("reward", 0) or 0),
                }
            )

    retention_rows = []
    cross_level_rows = []
    delayed_rows = []
    for key, rows in sorted(occurrences.items()):
        first = rows[0]
        first_trial = (first["level"], first["attempt"])

        later_trial = next(
            (
                row
                for row in rows[1:]
                if (row["level"], row["attempt"]) != first_trial
            ),
            None,
        )
        if later_trial is not None:
            outcome = attempt_outcomes.get(
                (later_trial["level"], later_trial["attempt"]), {}
            )
            retention_rows.append(
                {
                    "interaction": list(key),
                    "first_decision": first["decision"],
                    "reuse_decision": later_trial["decision"],
                    "gap_decisions": later_trial["decision"] - first["decision"],
                    "reuse_level": later_trial["level"],
                    "reuse_attempt": later_trial["attempt"],
                    "reuse_trial_won": bool(outcome.get("won")),
                }
            )

        later_level = next(
            (row for row in rows[1:] if row["level"] > first["level"]), None
        )
        if later_level is not None:
            outcome = attempt_outcomes.get(
                (later_level["level"], later_level["attempt"]), {}
            )
            cross_level_rows.append(
                {
                    "interaction": list(key),
                    "source_level": first["level"],
                    "reuse_level": later_level["level"],
                    "gap_decisions": later_level["decision"] - first["decision"],
                    "reuse_trial_won": bool(outcome.get("won")),
                }
            )

        delayed = next(
            (
                row
                for row in rows[1:]
                if row["decision"] - first["decision"] >= delayed_use_min_steps
            ),
            None,
        )
        if delayed is not None:
            outcome = attempt_outcomes.get((delayed["level"], delayed["attempt"]), {})
            delayed_rows.append(
                {
                    "interaction": list(key),
                    "gap_decisions": delayed["decision"] - first["decision"],
                    "reuse_level": delayed["level"],
                    "reuse_attempt": delayed["attempt"],
                    "reuse_trial_won": bool(outcome.get("won")),
                }
            )

    return {
        "available": bool(occurrences),
        "unique_interactions_observed": len(occurrences),
        "post_discovery_retention": {
            "eligible_interactions": len(retention_rows),
            "reuse_trial_win_rate": _rate(
                row["reuse_trial_won"] for row in retention_rows
            ),
            "events": retention_rows,
        },
        "cross_level_retention": {
            "eligible_interactions": len(cross_level_rows),
            "reuse_trial_win_rate": _rate(
                row["reuse_trial_won"] for row in cross_level_rows
            ),
            "events": cross_level_rows,
        },
        "delayed_use": {
            "minimum_gap_decisions": delayed_use_min_steps,
            "eligible_interactions": len(delayed_rows),
            "reuse_trial_win_rate": _rate(
                row["reuse_trial_won"] for row in delayed_rows
            ),
            "events": delayed_rows,
        },
    }


def summarize_behavior(
    steps: Sequence[dict],
    *,
    total_frames: int | None = None,
    start_level: int = 0,
    level_count: int | None = None,
    final_level: int | None = None,
    consecutive_wins_required: int = 2,
    delayed_use_min_steps: int = DEFAULT_DELAYED_USE_MIN_STEPS,
    progress_horizon_decisions: int | None = None,
) -> dict:
    """Compute paper-aligned and memory-oriented metrics for one replay."""
    if consecutive_wins_required < 1:
        raise ValueError("consecutive_wins_required must be >= 1")
    if delayed_use_min_steps < 1:
        raise ValueError("delayed_use_min_steps must be >= 1")
    if progress_horizon_decisions is not None and progress_horizon_decisions < 1:
        raise ValueError("progress_horizon_decisions must be >= 1")

    real_steps = _real_steps(steps)
    raw_attempts = _attempt_rows(real_steps)
    horizon = progress_horizon_decisions or max(len(real_steps), 1)
    attempts, blocked_progress = _blocked_curriculum_view(
        raw_attempts,
        start_level=start_level,
        level_count=level_count,
        consecutive_wins_required=consecutive_wins_required,
        decision_horizon=horizon,
    )
    levels = _level_metrics(attempts, consecutive_wins_required)
    raw_levels = _level_metrics(raw_attempts, consecutive_wins_required)
    first_win_decision = next(
        (index for index, step in enumerate(real_steps, start=1) if step.get("won")),
        None,
    )
    first_win_step = (
        real_steps[first_win_decision - 1] if first_win_decision is not None else None
    )
    first_frame = None
    if first_win_step is not None:
        first_frame = first_win_step.get("frame", first_win_step.get("frame_num"))

    relearning_rows = [
        row for row in levels if row["relearning_savings_fraction"] is not None
    ]
    discovery_steps = [
        row["decisions_to_first_win"]
        for row in levels
        if row["decisions_to_first_win"] is not None
    ]
    execution_steps = [
        value
        for row in levels
        for value in row["subsequent_winning_attempt_decisions"]
    ]
    auc = _level_auc(
        real_steps,
        total_frames=total_frames,
        start_level=start_level,
        level_count=level_count,
    )
    interactions = _interaction_metrics(
        real_steps,
        raw_attempts,
        delayed_use_min_steps=delayed_use_min_steps,
    )

    inferred_final_level = max(
        [int(step.get("level", start_level)) for step in real_steps],
        default=start_level,
    )
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "start_level": start_level,
        "level_count": level_count,
        "decisions": len(real_steps),
        "total_frames": total_frames,
        "first_win": {
            "won": first_win_decision is not None,
            "steps_to_first_win": first_win_decision,
            "frames_to_first_win": int(first_frame) if first_frame is not None else None,
        },
        "wins": sum(bool(step.get("won")) for step in real_steps),
        "losses": sum(bool(step.get("lose")) for step in real_steps),
        "total_reward": sum(float(step.get("reward", 0) or 0) for step in real_steps),
        "final_level": (
            int(final_level) if final_level is not None else inferred_final_level
        ),
        "attempts": attempts,
        "raw_attempts": raw_attempts,
        "levels": levels,
        "raw_levels": raw_levels,
        "discovery": {
            "attempted_level_instances": len(levels),
            "solved_level_instances": len(discovery_steps),
            "attempted_level_solve_rate": _rate(
                row["decisions_to_first_win"] is not None for row in levels
            ),
            "curriculum_solve_rate": (
                len(discovery_steps) / level_count
                if level_count is not None and level_count > 0
                else None
            ),
            "median_steps_to_first_win": (
                statistics.median(discovery_steps) if discovery_steps else None
            ),
        },
        "execution": {
            "subsequent_wins": len(execution_steps),
            "median_steps_per_subsequent_win": (
                statistics.median(execution_steps) if execution_steps else None
            ),
            "discovery_minus_execution_median_steps": (
                statistics.median(discovery_steps)
                - statistics.median(execution_steps)
                if discovery_steps and execution_steps
                else None
            ),
        },
        "relearning_savings": {
            "eligible_levels": len(relearning_rows),
            "mean_steps": _mean(
                row["relearning_savings_steps"] for row in relearning_rows
            ),
            "mean_fraction": _mean(
                row["relearning_savings_fraction"] for row in relearning_rows
            ),
        },
        "level_progress": auc,
        "blocked_curriculum_progress": blocked_progress,
        "interaction_memory_proxies": interactions,
    }


def compare_behavior_summaries(treatment: dict, control: dict) -> dict:
    """Compute directional treatment-minus-control memory benefits.

    Positive values always favor the treatment.  For steps to first win,
    fewer is better, so the reported benefit is ``control - treatment``.
    """

    def nested(summary: dict, *keys):
        value = summary
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    def delta_high(*keys):
        t_value = nested(treatment, *keys)
        c_value = nested(control, *keys)
        if t_value is None or c_value is None:
            return None
        return float(t_value) - float(c_value)

    t_first = nested(treatment, "first_win", "steps_to_first_win")
    c_first = nested(control, "first_win", "steps_to_first_win")
    first_win_benefit = (
        float(c_first) - float(t_first)
        if t_first is not None and c_first is not None
        else None
    )

    return {
        "steps_to_first_win_benefit": first_win_benefit,
        "first_win_success_delta": delta_high("first_win", "won"),
        "final_level_delta": delta_high("final_level"),
        "level_progress_auc_delta": delta_high(
            "level_progress", "normalized_level_progress_auc"
        ),
        "blocked_level_progress_auc_delta": delta_high(
            "blocked_curriculum_progress", "normalized_level_progress_auc"
        ),
        "blocked_final_level_delta": delta_high(
            "blocked_curriculum_progress", "final_level"
        ),
        "relearning_savings_delta": delta_high(
            "relearning_savings", "mean_fraction"
        ),
        "post_discovery_retention_delta": delta_high(
            "interaction_memory_proxies",
            "post_discovery_retention",
            "reuse_trial_win_rate",
        ),
        "cross_level_retention_delta": delta_high(
            "interaction_memory_proxies",
            "cross_level_retention",
            "reuse_trial_win_rate",
        ),
        "delayed_use_delta": delta_high(
            "interaction_memory_proxies", "delayed_use", "reuse_trial_win_rate"
        ),
    }


def behavior_scalar_metrics(summary: dict) -> dict[str, float | int | bool]:
    """Flatten summary headline fields for W&B or tabular logging."""
    first_win = summary["first_win"]
    discovery = summary["discovery"]
    execution = summary["execution"]
    proxies = summary["interaction_memory_proxies"]
    candidates = {
        "behavior/first_win_success": first_win["won"],
        "behavior/steps_to_first_win": first_win["steps_to_first_win"],
        "behavior/frames_to_first_win": first_win["frames_to_first_win"],
        "behavior/curriculum_solve_rate": discovery["curriculum_solve_rate"],
        "behavior/attempted_level_solve_rate": discovery[
            "attempted_level_solve_rate"
        ],
        "behavior/median_discovery_steps": discovery[
            "median_steps_to_first_win"
        ],
        "behavior/median_execution_steps": execution[
            "median_steps_per_subsequent_win"
        ],
        "behavior/discovery_execution_gap_steps": execution[
            "discovery_minus_execution_median_steps"
        ],
        "behavior/normalized_level_progress_auc": summary["level_progress"][
            "normalized_level_progress_auc"
        ],
        "behavior/blocked_level_progress_auc": summary[
            "blocked_curriculum_progress"
        ]["normalized_level_progress_auc"],
        "behavior/blocked_final_level": summary["blocked_curriculum_progress"][
            "final_level"
        ],
        "behavior/relearning_savings_fraction": summary["relearning_savings"][
            "mean_fraction"
        ],
        "behavior/post_discovery_retention": proxies[
            "post_discovery_retention"
        ]["reuse_trial_win_rate"],
        "behavior/cross_level_retention": proxies["cross_level_retention"][
            "reuse_trial_win_rate"
        ],
        "behavior/delayed_use": proxies["delayed_use"]["reuse_trial_win_rate"],
    }
    return {key: value for key, value in candidates.items() if value is not None}


def empirical_wasserstein_1d(left: Sequence[float], right: Sequence[float]) -> float:
    """Exact first Wasserstein distance between two 1-D empirical samples."""
    if not left or not right:
        raise ValueError("both samples must be non-empty")
    left = sorted(float(value) for value in left)
    right = sorted(float(value) for value in right)
    positions = sorted(set(left + right))
    i = j = 0
    cdf_left = cdf_right = 0.0
    distance = 0.0
    previous = positions[0]
    for position in positions:
        distance += abs(cdf_left - cdf_right) * (position - previous)
        while i < len(left) and left[i] == position:
            i += 1
        while j < len(right) and right[j] == position:
            j += 1
        cdf_left = i / len(left)
        cdf_right = j / len(right)
        previous = position
    return distance


def compare_first_win_cohorts(model: Sequence[dict], human: Sequence[dict]) -> dict:
    """Compare solved level-instance discovery distributions in log space.

    The paper applies EMD to KDE-smoothed distributions.  This dependency-free
    implementation reports empirical Wasserstein-1 on the underlying log10
    samples, so it is suitable for automated regression tests but will not
    numerically reproduce the paper's KDE-dependent values.
    """

    def cohort_values(summaries: Sequence[dict]) -> tuple[list[float], int, int]:
        values: list[float] = []
        attempted = 0
        expected = 0
        for summary in summaries:
            levels = summary.get("levels") or []
            if "levels" in summary:
                attempted += len(levels)
                expected += int(summary.get("level_count") or len(levels))
                values.extend(
                    float(row["decisions_to_first_win"])
                    for row in levels
                    if row.get("decisions_to_first_win") is not None
                )
                continue

            # Backward-compatible fallback for summaries produced before
            # per-level metrics existed.
            expected += 1
            attempted += 1
            value = summary.get("first_win", {}).get("steps_to_first_win")
            if value is not None:
                values.append(float(value))
        return values, attempted, expected

    model_steps, model_attempted, model_expected = cohort_values(model)
    human_steps, human_attempted, human_expected = cohort_values(human)
    emd = None
    if model_steps and human_steps:
        emd = empirical_wasserstein_1d(
            [math.log10(value) for value in model_steps],
            [math.log10(value) for value in human_steps],
        )

    def mean_nested(summaries: Sequence[dict], *keys) -> float | None:
        values = []
        for summary in summaries:
            value = summary
            for key in keys:
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None:
                values.append(float(value))
        return _mean(values)

    return {
        "model_runs": len(model),
        "human_runs": len(human),
        "model_attempted_level_instances": model_attempted,
        "human_attempted_level_instances": human_attempted,
        "model_expected_level_instances": model_expected,
        "human_expected_level_instances": human_expected,
        "model_solved_level_instances": len(model_steps),
        "human_solved_level_instances": len(human_steps),
        "model_attempted_level_solve_rate": (
            len(model_steps) / model_attempted if model_attempted else None
        ),
        "human_attempted_level_solve_rate": (
            len(human_steps) / human_attempted if human_attempted else None
        ),
        "model_curriculum_solve_rate": (
            len(model_steps) / model_expected if model_expected else None
        ),
        "human_curriculum_solve_rate": (
            len(human_steps) / human_expected if human_expected else None
        ),
        "model_median_steps_to_first_win": (
            statistics.median(model_steps) if model_steps else None
        ),
        "human_median_steps_to_first_win": (
            statistics.median(human_steps) if human_steps else None
        ),
        "log10_steps_to_first_win_emd": emd,
        "emd_population": "solved level-instances; solve rates reported separately",
        "emd_method": "empirical Wasserstein-1 on log10 counts; no KDE smoothing",
        "model_mean_blocked_final_level": mean_nested(
            model, "blocked_curriculum_progress", "final_level"
        ),
        "human_mean_blocked_final_level": mean_nested(
            human, "blocked_curriculum_progress", "final_level"
        ),
        "model_mean_blocked_progress_auc": mean_nested(
            model,
            "blocked_curriculum_progress",
            "normalized_level_progress_auc",
        ),
        "human_mean_blocked_progress_auc": mean_nested(
            human,
            "blocked_curriculum_progress",
            "normalized_level_progress_auc",
        ),
    }
