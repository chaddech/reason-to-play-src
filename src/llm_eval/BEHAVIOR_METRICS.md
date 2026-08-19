# Behavioral metrics

Both generative gameplay and human action replay now emit a top-level
`behavior_metrics` object. The metrics operate on free-rollout trajectories;
they never compare actions with a single reference path and do not require
rationales.

## Performance measures

- **Level discovery:** decisions from entering a level through its first win,
  including failed attempts. Reports attempted and full-curriculum solve rates,
  the median over solved level-instances, and per-level values.
- **Execution:** decisions in every winning attempt after the first win on the
  same level, plus the median discovery-minus-execution gap.
- **Capability progression:** both the observed level trajectory and the
  paper-compatible blocked-curriculum projection (two consecutive wins to
  advance; later levels censored after the first failed mastery criterion).
  Each reports the curve and AUC. A separate frame-based observed curve is
  included when the replay has a complete engine-frame clock.
- **Run outcomes:** wins, losses, reward, final level, decisions, and both
  decisions and engine frames to the run's first win.
- **Human alignment:** the cohort analyzer pools solved level-instances and
  computes empirical Wasserstein-1 distance on log10 discovery counts. It also
  reports attempted and full-curriculum solve rates so failures are not hidden.
  The paper smooths its distributions with a KDE before EMD; the dependency-free
  analyzer intentionally does not, so its EMD is aligned in meaning but will not
  exactly reproduce a paper figure value.

`steps` in these fields means environment decisions represented by real replay
rows. `_level_advance` markers and crash rows are excluded. `frames` means raw
game-engine updates. Keeping both clocks prevents idle-frame injection or human
reaction intervals from silently changing the meaning of a comparison.

## Memory-oriented measures

- **Relearning savings:** reduction from the first winning attempt's length to
  the median later winning-attempt length on the same level.
- **Post-discovery retention proxy:** for each engine interaction observed on a
  later attempt, whether that later attempt was won.
- **Cross-level retention proxy:** the same outcome measure when the interaction
  recurs on a later level.
- **Delayed-use proxy:** the same outcome measure when recurrence is at least 50
  decisions after discovery (configurable in the analyzer).
- **Causal state benefit:** treatment-minus-control deltas for matched
  persistent-state and reset/shunted-state runs. Positive values always favor
  persistent state; for steps to first win the sign is reversed because fewer
  is better.

The three interaction measures are diagnostics, not proof of memory. A repeated
`(actor color, effect, actee color)` event followed by a win can happen without
recall. The decisive test is a matched intervention: same game, seed, model,
budget, and training checkpoint, with only persistent state retained versus
reset or shuffled state.

For raw human fixed-budget replays, `levels` and the headline discovery fields
use the blocked projection; `raw_levels` and `raw_attempts` retain the uncensored
record for diagnostics. Interaction-memory proxies likewise describe the raw
observed trajectory, not counterfactual events on censored levels.

## Commands

```bash
python -m src.llm_eval.behavior_analysis summarize RUN.replay.json.gz

python -m src.llm_eval.behavior_analysis compare \
  PERSISTENT.replay.json.gz RESET.replay.json.gz

python -m src.llm_eval.behavior_analysis cohort \
  --model out/model/*.replay.json.gz \
  --human out/human/*.replay.json.gz \
  --progress-horizon 1600 \
  --output out/model-vs-human.json
```

Old replays can be analyzed, but interaction-memory measures report unavailable
unless their step rows contain the new `interactions` field.
