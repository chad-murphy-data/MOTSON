# MOTSON Prediction Evaluation

This document describes how to evaluate MOTSON's prediction accuracy against betting odds and Opta's weekly predictions.

## Quick Start

```bash
# Run evaluation (MOTSON only if no external data available)
python scripts/evaluate_predictions.py

# Fetch external data first (requires network access)
python scripts/fetch_external_odds.py

# Then run full comparison
python scripts/evaluate_predictions.py
```

## Scripts

### `evaluate_predictions.py`

Main evaluation script that compares MOTSON predictions against:
1. **Betting odds** from football-data.co.uk (implied probabilities)
2. **Opta predictions** from The Analyst (theanalyst.com)

**Features:**
- Regenerates historical match predictions from stored theta values
- Calculates Brier Score, Log Loss, and Accuracy
- Computes ROI if betting based on MOTSON edge
- Produces calibration analysis
- Saves results to JSON

**Usage:**
```bash
# Basic evaluation
python scripts/evaluate_predictions.py

# Evaluate up to specific week
python scripts/evaluate_predictions.py --week 20

# Use local betting odds CSV
python scripts/evaluate_predictions.py --betting-csv data/my_odds.csv

# Use local Opta JSON
python scripts/evaluate_predictions.py --opta-json data/opta.json

# Refresh cached web data
python scripts/evaluate_predictions.py --refresh

# Custom output file
python scripts/evaluate_predictions.py --output results.json
```

### `fetch_external_odds.py`

Helper script to fetch external data sources.

**Usage:**
```bash
# Fetch all external data
python scripts/fetch_external_odds.py

# Fetch for specific season
python scripts/fetch_external_odds.py --season 2025

# Create sample templates (if fetching fails)
python scripts/fetch_external_odds.py --create-samples
```

## Data Sources

### football-data.co.uk

Free historical betting odds for EPL matches.

**URL Format:** `https://www.football-data.co.uk/mmz4281/{SEASON}/E0.csv`
- Season code: 2526 for 2025-26 season

**CSV Columns:**
| Column | Description |
|--------|-------------|
| Date | Match date |
| HomeTeam | Home team name |
| AwayTeam | Away team name |
| FTHG | Full-time home goals |
| FTAG | Full-time away goals |
| FTR | Full-time result (H/D/A) |
| B365H | Bet365 home odds |
| B365D | Bet365 draw odds |
| B365A | Bet365 away odds |

### The Analyst (Opta)

Opta's weekly match predictions published at:
- https://theanalyst.com/articles/premier-league-match-predictions

No public API - predictions must be manually extracted or scraped.

**JSON Format:**
```json
{
  "Arsenal|Liverpool": {
    "home_team": "Arsenal",
    "away_team": "Liverpool",
    "home_win_prob": 0.42,
    "draw_prob": 0.28,
    "away_win_prob": 0.30,
    "source": "opta",
    "matchweek": 1,
    "match_id": ""
  }
}
```

## Metrics

### Brier Score

Measures accuracy of probabilistic predictions:
```
Brier Score = (1/N) × Σ[(p_home - y_home)² + (p_draw - y_draw)² + (p_away - y_away)²]
```

- **0** = Perfect predictions
- **2** = Worst possible
- **~0.6** = Typical for EPL predictions

### Log Loss

Penalizes confident wrong predictions more heavily:
```
Log Loss = -(1/N) × Σ[y_home×log(p_home) + y_draw×log(p_draw) + y_away×log(p_away)]
```

- Lower is better
- **~1.0-1.1** = Typical for EPL predictions

### Accuracy

Percentage of matches where highest-probability outcome was correct.

- **~50%** = Typical for EPL (better than random 33%)
- Home win predictions are easiest (~80% when predicted)
- Draw predictions are hardest (~10% when predicted)

### ROI (Return on Investment)

Calculates profit/loss if betting on MOTSON predictions with edge > threshold.

```
ROI = (Total Returned - Total Staked) / Total Staked × 100%
```

Positive ROI = MOTSON finds value the market misses.

## Example Output

```
================================================================================
MOTSON PREDICTION EVALUATION REPORT
================================================================================
Generated: 2026-01-24 22:00:00

--------------------------------------------------------------------------------
Metric                          MOTSON      Betting         Opta
--------------------------------------------------------------------------------
Matches Evaluated                  225          220          180
Brier Score                     0.5878       0.5952       0.5890
Log Loss                        0.9837       0.9956       0.9880
Accuracy                         52.4%        51.8%        52.0%
--------------------------------------------------------------------------------

Per-Outcome Accuracy (MOTSON):
  Home wins:  86.4%
  Draws:      6.7%
  Away wins:  40.3%

--------------------------------------------------------------------------------
ROI ANALYSIS (Betting on MOTSON edge > 5%)
--------------------------------------------------------------------------------
  Bets placed:    45
  Wins:           24 (53.3%)
  Total staked:   $45.00
  Total returned: $51.20
  ROI:            +13.8%
  → MOTSON shows positive expected value!

--------------------------------------------------------------------------------
INTERPRETATION
--------------------------------------------------------------------------------
✓ MOTSON Brier Score is 0.0074 BETTER than betting markets
≈ MOTSON Brier Score is similar to Opta (diff: -0.0012)

Brier Score: 0 = perfect, 2 = worst possible for 3-way classification
Log Loss: Lower is better. ~1.1 is typical for EPL predictions
================================================================================
```

## Cache Files

Data is cached in `data/evaluation_cache/`:

| File | Description |
|------|-------------|
| `betting_odds.csv` | Football-data.co.uk odds |
| `opta_predictions.json` | Parsed Opta predictions |
| `opta_raw.html` | Raw HTML for debugging |
| `evaluation_results.json` | Latest evaluation results |

## Team Name Normalization

The scripts normalize team names across sources:

| Source Name | Normalized |
|------------|------------|
| Man United | Manchester Utd |
| Man City | Manchester City |
| Nott'm Forest | Nott'ham Forest |
| Spurs | Tottenham |
| Wolves | Wolves |

## Troubleshooting

### Network Errors

If fetching fails (403 Forbidden):
1. Create sample templates: `python scripts/fetch_external_odds.py --create-samples`
2. Manually download from football-data.co.uk
3. Place in `data/evaluation_cache/betting_odds.csv`

### No Predictions Found

If historical predictions are missing:
1. The script regenerates predictions from stored theta values
2. Requires `team_state_history` table to have data
3. Run backfill: `python scripts/backfill_history.py`

### Opta Parsing Fails

Opta's HTML structure may change:
1. Check `data/evaluation_cache/opta_raw.html` for structure
2. Update parsing patterns in `fetch_external_odds.py`
3. Or manually create `opta_predictions.json`

## Development

### Adding New Metrics

Add to `evaluate_predictions.py`:
1. Create calculation function
2. Add to `EvaluationMetrics` dataclass
3. Call in `evaluate_source()`
4. Add to `print_comparison_report()`

### Adding New Data Sources

1. Create fetch function in `fetch_external_odds.py`
2. Add parsing function
3. Update CLI arguments in both scripts
4. Update this documentation
