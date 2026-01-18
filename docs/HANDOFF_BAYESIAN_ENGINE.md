# MOTSON Bayesian Engine - Design Decisions & Handoff Document

## Overview

MOTSON (Model Of Team Strength Outcome Network) is an EPL prediction system that uses Bayesian-inspired updates to track team strength over a season. This document captures the key design decisions made during development and open questions for future work.

## Core Philosophy

> "Track distributions, not point estimates. Update on cumulative calibration, not individual surprises."

The model doesn't react to single match results. Instead, it tracks whether a team is **systematically** over or under-performing across the entire season. A team needs to be consistently off expectations (z-score > 1.0) before their strength estimate (theta) significantly changes.

---

## Key Design Decisions

### 1. Theta Accumulation in Backfill (Bug Fix)

**File:** `scripts/backfill_history.py`

**Problem:** The original backfill script reset team states at the start of each week, meaning theta updates never accumulated. A team's week 22 theta was based only on week 22's cumulative z-score against their *original* theta, not their evolved theta.

**Solution:** Move `load_initial_team_states()` outside the week loop so updates persist across weeks.

```python
# BEFORE (bug): Reset each week
for week in range(start_week, end_week + 1):
    team_states = load_initial_team_states()  # <-- Inside loop!
    ...

# AFTER (fix): Load once, accumulate
team_states = load_initial_team_states()  # <-- Outside loop
for week in range(start_week, end_week + 1):
    ...
```

---

### 2. Promoted Team Default Theta

**File:** `backend/config.py`, `backend/models/bayesian_engine.py`

**Problem:** Sunderland (promoted after 8 years away) had a GPCM theta of +0.44 (implying ~6th place), but this was based on stale data from their pre-relegation seasons. They're actually a promoted team who should start with relegation-zone expectations.

**Solution:** Teams with `n_seasons <= 2` in the recent EPL window use a fixed "promoted team default" theta instead of their GPCM value.

```python
PROMOTED_TEAM_THETA: float = -0.5789  # ~16th place expectation
```

**Rationale:** Historically, promoted teams finish around positions 15-17 on average. This is the mean theta for those positions.

**Open Question:** Should promoted teams instead get a higher sigma (more uncertainty) rather than a fixed theta? This would let the model learn their true strength faster while still being uncertain initially.

---

### 3. GPCM Theta Normalization

**File:** `backend/models/bayesian_engine.py`

**Problem:** GPCM (Generalized Partial Credit Model) theta values aren't anchored to a fixed scale. Different years' calculations may have different scales depending on which teams were in the dataset. A "worst team" in 2020 might have theta -0.6, while 2023's worst might have -0.9.

**Solution:** Z-score normalize GPCM thetas to have mean=0 and a target standard deviation.

```python
THETA_TARGET_STD: float = 0.50  # Target std for normalized thetas
```

**Result:** Top teams end up around +0.8 to +0.9, bottom teams around -0.9 to -1.1.

**Trade-off:** This compresses the original distribution (which had std ~0.56, range ~2.1). The normalized version is tighter but ensures year-over-year consistency.

**Open Question:** Is std=0.5 the right target? The model's prediction functions were originally calibrated against a different scale. Should we increase to 0.55-0.6 for more expressiveness?

---

### 4. Drift Updates for Below-Threshold Teams

**File:** `backend/models/bayesian_engine.py`

**Problem:** Teams with z-scores below the UPDATE_THRESHOLD (1.0) showed zero theta movement. This created flat lines on the theta chart for teams performing "as expected" - which looked wrong visually even if philosophically defensible.

**Solution:** Add a small "drift" update even when below threshold. The drift is 20% of what a full update would be, scaled by the z-score.

```python
drift_factor = 0.20
drift = (
    z_score
    * drift_factor
    * cfg.LEARNING_RATE
    * (team.sigma / cfg.BASELINE_SIGMA)
    * (1.0 / team.stickiness)
)
```

**Rationale:** This preserves the core philosophy (big moves require systematic deviation) while creating realistic week-to-week wobble. A team with z=+0.5 will drift slightly upward; z=-0.3 will drift slightly downward.

**Open Questions:**
- Is 20% the right drift factor? It was chosen somewhat arbitrarily.
- Should drift also include a gravity component pulling teams toward their historical mean?

---

### 5. Title Probability Normalization

**File:** `backend/services/survival_calibration.py`

**Problem:** After survival calibration adjusted the leader's title probability downward (e.g., from MC's 91% to calibrated 85%), the total across all teams no longer summed to 100%.

**Solution:** Normalize title probabilities after calibration.

```python
total_title_prob = sum(r["title_prob"] for r in calibrated_results.values())
if total_title_prob > 0 and abs(total_title_prob - 1.0) > 0.001:
    for team in calibrated_results:
        calibrated_results[team]["title_prob"] /= total_title_prob
```

---

### 6. Survival Calibration for Title Probabilities

**File:** `backend/services/survival_calibration.py`

**Context:** Pure Monte Carlo simulations tend to be overconfident about title probabilities. A 7-point lead with 16 games to go might show 91% title probability, but historically such leads don't hold 91% of the time.

**Solution:** Blend MC probabilities with empirical survival rates from `data/lead_survival.json`. The blend weight varies based on:
- Games remaining (more weight to empirical early in season)
- Lead size (moderate leads trust empirical more)
- Theta advantage (strong teams get slightly less empirical weight)

---

## Configuration Summary

Key hyperparameters in `backend/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `UPDATE_THRESHOLD` | 1.0 | Z-score needed for full theta update |
| `LEARNING_RATE` | 0.10 | Base theta change per excess Z |
| `BASELINE_SIGMA` | 0.30 | Reference sigma for scaling |
| `PROMOTED_TEAM_THETA` | -0.5789 | Default theta for newly promoted teams |
| `THETA_TARGET_STD` | 0.50 | Target std for GPCM normalization |
| `HOME_ADVANTAGE` | 0.30 | Theta boost for home team |
| `PRESEASON_SIGMA_BOOST` | 0.20 | Extra uncertainty at season start |

---

## Open Questions for Future Work

### 1. Drift Factor Tuning
The 20% drift factor was chosen to create visible wobble without being too volatile. Should this be tuned based on backtesting? What's the right balance between responsiveness and stability?

### 2. Gravity in Drift Updates
Currently, gravity (pull toward historical mean position) only applies when a full update triggers. Should below-threshold drift also include a small gravity component? This would prevent teams from drifting too far from their "true level" over many small updates.

### 3. Promoted Team Handling
Fixed theta (-0.5789) vs higher sigma - which is better? A higher sigma would let the model learn faster but start uncertain. A fixed theta makes a strong prior assumption about where promoted teams belong.

### 4. GPCM Scale Sensitivity
The model's match prediction functions (`predict_match_probs`) use hardcoded scaling factors (K_SCALE, B0_DRAW, C_MISMATCH) that were calibrated against a certain theta scale. After normalization, are these still optimal?

### 5. Stickiness Usage
Stickiness (how resistant a team is to theta changes) is used in the update formula, but it's derived from historical position variance. Is this the right proxy? Should established clubs like Liverpool (stickiness ~0.9) really move 2x slower than volatile clubs like Brentford (stickiness ~0.4)?

---

## File Map

| File | Purpose |
|------|---------|
| `backend/config.py` | All hyperparameters |
| `backend/models/bayesian_engine.py` | Core update logic, theta normalization |
| `backend/models/team_state.py` | TeamState dataclass |
| `backend/services/survival_calibration.py` | Title probability calibration |
| `backend/services/monte_carlo.py` | Season simulation |
| `backend/services/weekly_update.py` | Orchestrates weekly updates |
| `scripts/backfill_history.py` | Replays season week-by-week |
| `data/team_parameters.csv` | Stickiness, sigma, gravity per team |
| `data/MOTSON_GPCM_Theta_Alpha.csv` | Initial GPCM theta values |
| `data/lead_survival.json` | Historical lead survival rates |

---

## Testing the Changes

To verify the system works correctly:

```bash
# Run backfill and check output
python scripts/backfill_history.py

# Verify all teams show theta movement
python -c "
import sqlite3
conn = sqlite3.connect('data/motson.db')
cursor = conn.cursor()
cursor.execute('''
    SELECT team, MAX(theta_home) - MIN(theta_home) as range
    FROM team_state_history GROUP BY team ORDER BY range
''')
for team, r in cursor.fetchall():
    print(f'{team}: {r:.4f}')
"

# Verify title probabilities sum to 100%
python -c "
import sqlite3
conn = sqlite3.connect('data/motson.db')
cursor = conn.cursor()
cursor.execute('SELECT SUM(title_prob) FROM season_predictions WHERE week = 22')
print(f'Title prob sum: {cursor.fetchone()[0]:.4f}')
"
```

---

## Summary

The system now:
1. Accumulates theta updates across weeks (bug fix)
2. Handles promoted teams with a sensible default
3. Normalizes GPCM thetas for year-over-year consistency
4. Creates realistic theta wobble via drift updates
5. Ensures title probabilities sum to 100%

The core philosophy remains intact: big belief changes require systematic deviation, not single-match surprises. The drift mechanism adds visual realism without compromising this principle.
