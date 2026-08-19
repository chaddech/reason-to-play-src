import copy
import gzip
import json
import sys
from pathlib import Path

import pytest

# The enclosing Polyphony checkout also has a top-level ``src`` package.
# Import this nested project's package root directly so pytest cannot resolve
# the parent's unrelated package during collection.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_eval.shared.behavior_metrics import (  # noqa: E402
    compare_behavior_summaries,
    compare_first_win_cohorts,
    empirical_wasserstein_1d,
    summarize_behavior,
)
from llm_eval.behavior_analysis import main as analysis_main  # noqa: E402


INTERACTION = {
    "actor_color": "BLUE",
    "effect": "collectResource",
    "actee_color": "YELLOW",
}


def _step(
    decision,
    level,
    attempt,
    *,
    won=False,
    lose=False,
    interaction=False,
):
    return {
        "step": decision - 1,
        "frame": decision * 5,
        "level": level,
        "attempt": attempt,
        "action": "right",
        "reward": 1 if won else 0,
        "won": won,
        "lose": lose,
        "timeout": False,
        "interactions": [INTERACTION] if interaction else [],
    }


@pytest.fixture
def trajectory():
    return [
        _step(1, 0, 0, interaction=True),
        _step(2, 0, 0),
        _step(3, 0, 0, lose=True),
        _step(4, 0, 1, interaction=True),
        _step(5, 0, 1, won=True),
        _step(6, 0, 2, won=True, interaction=True),
        _step(7, 1, 0, interaction=True),
        _step(8, 1, 0, won=True),
    ]


@pytest.fixture
def summary(trajectory):
    return summarize_behavior(
        trajectory,
        total_frames=40,
        start_level=0,
        level_count=3,
        final_level=2,
        consecutive_wins_required=2,
        delayed_use_min_steps=5,
    )


def test_paper_aligned_performance_metrics(summary):
    assert summary["first_win"] == {
        "won": True,
        "steps_to_first_win": 5,
        "frames_to_first_win": 25,
    }
    assert summary["discovery"]["solved_level_instances"] == 2
    assert summary["discovery"]["curriculum_solve_rate"] == pytest.approx(2 / 3)
    assert summary["discovery"]["median_steps_to_first_win"] == 3.5
    assert summary["execution"]["median_steps_per_subsequent_win"] == 1
    assert summary["execution"]["discovery_minus_execution_median_steps"] == 2.5
    assert summary["levels"][0]["decisions_to_first_win"] == 5
    assert summary["levels"][1]["decisions_to_first_win"] == 2


def test_relearning_and_interaction_memory_proxies(summary):
    assert summary["relearning_savings"]["mean_fraction"] == 0.5
    proxies = summary["interaction_memory_proxies"]
    assert proxies["post_discovery_retention"]["reuse_trial_win_rate"] == 1
    assert proxies["cross_level_retention"]["reuse_trial_win_rate"] == 1
    assert proxies["delayed_use"]["reuse_trial_win_rate"] == 1
    assert proxies["delayed_use"]["events"][0]["gap_decisions"] == 5


def test_decision_and_frame_level_progress_are_separate(summary):
    progress = summary["level_progress"]
    assert progress["time_unit"] == "decisions"
    assert progress["horizon"] == 8
    assert progress["normalized_level_progress_auc"] == pytest.approx(0.0625)
    assert progress["frame_based"]["time_unit"] == "frames"
    assert progress["frame_based"]["horizon"] == 40
    assert progress["frame_based"]["normalized_level_progress_auc"] == pytest.approx(
        0.0625
    )


def test_comparison_uses_positive_as_treatment_benefit(summary):
    control = copy.deepcopy(summary)
    control["first_win"]["steps_to_first_win"] = 8
    control["first_win"]["won"] = False
    control["final_level"] = 1
    control["relearning_savings"]["mean_fraction"] = 0.25
    benefit = compare_behavior_summaries(summary, control)
    assert benefit["steps_to_first_win_benefit"] == 3
    assert benefit["first_win_success_delta"] == 1
    assert benefit["final_level_delta"] == 1
    assert benefit["relearning_savings_delta"] == 0.25


def test_cohort_comparison_pools_level_instances(summary):
    comparison = compare_first_win_cohorts([summary], [copy.deepcopy(summary)])
    assert comparison["model_expected_level_instances"] == 3
    assert comparison["model_attempted_level_instances"] == 2
    assert comparison["model_solved_level_instances"] == 2
    assert comparison["model_curriculum_solve_rate"] == pytest.approx(2 / 3)
    assert comparison["log10_steps_to_first_win_emd"] == 0


def test_blocked_projection_censors_human_levels_after_first_stall():
    raw_fixed_budget_steps = [
        _step(1, 0, 0, won=True),
        _step(2, 0, 1, won=True),
        _step(3, 1, 0, lose=True),
        _step(4, 2, 0, won=True),
        _step(5, 2, 1, won=True),
    ]
    result = summarize_behavior(
        raw_fixed_budget_steps,
        start_level=0,
        level_count=3,
        consecutive_wins_required=2,
    )
    assert [row["level"] for row in result["levels"]] == [0, 1]
    assert [row["level"] for row in result["raw_levels"]] == [0, 1, 2]
    assert result["discovery"]["curriculum_solve_rate"] == pytest.approx(1 / 3)
    progress = result["blocked_curriculum_progress"]
    assert progress["mastered_levels"] == 1
    assert progress["final_level"] == 1
    assert progress["stalled_level"] == 1
    assert progress["censored_levels"] == [2]


def test_empirical_wasserstein():
    assert empirical_wasserstein_1d([0], [1]) == 1
    assert empirical_wasserstein_1d([0, 2], [1, 3]) == 1
    with pytest.raises(ValueError):
        empirical_wasserstein_1d([], [1])


def test_cli_automates_compressed_model_human_comparison(
    trajectory, tmp_path, capsys
):
    replay_path = tmp_path / "run.replay.json.gz"
    replay = {
        "start_level": 0,
        "final_level": 2,
        "total_frames": 40,
        "meta": {"level_count": 3, "consecutive_wins_required": 2},
        "steps": trajectory,
    }
    with gzip.open(replay_path, "wt") as handle:
        json.dump(replay, handle)

    assert analysis_main(
        [
            "cohort",
            "--model",
            str(replay_path),
            "--human",
            str(replay_path),
            "--delay-steps",
            "5",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["comparison"]["model_solved_level_instances"] == 2
    assert output["comparison"]["log10_steps_to_first_win_emd"] == 0
