#!/usr/bin/env python3
"""
MOTSON Prediction Evaluation Script

Compares MOTSON predictions against:
1. Betting odds from football-data.co.uk (implied probabilities)
2. Opta weekly predictions from The Analyst

Metrics calculated:
- Brier Score (lower is better)
- Log Loss (lower is better)
- Accuracy (percentage of correct predictions)
- Calibration (predicted vs actual outcome rates)
- ROI (return on investment if betting on MOTSON)

Data sources:
- football-data.co.uk: Historical betting odds CSV files (free)
- The Analyst (theanalyst.com): Opta's weekly match predictions

Usage:
    python scripts/evaluate_predictions.py [--week N] [--refresh]
"""

import argparse
import csv
import io
import json
import re
import sqlite3
import ssl
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Configuration
DATA_DIR = project_root / "data"
CACHE_DIR = DATA_DIR / "evaluation_cache"
DB_PATH = DATA_DIR / "motson.db"

# Football-data.co.uk URLs (season format: 2526 for 2025-26)
FOOTBALL_DATA_URL = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"

# The Analyst (Opta) match predictions page
OPTA_PREDICTIONS_URL = "https://theanalyst.com/articles/premier-league-match-predictions"
OPTA_TITLE_RACE_URL = "https://theanalyst.com/articles/who-will-win-the-premier-league-opta-supercomputer"


@dataclass
class MatchPrediction:
    """A match prediction from any source."""
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    source: str  # 'motson', 'betting', 'opta'
    matchweek: int = 0
    match_id: str = ""
    # Actual decimal odds (for betting sources - includes bookmaker margin)
    home_odds: float = 0.0
    draw_odds: float = 0.0
    away_odds: float = 0.0


@dataclass
class MatchResult:
    """Actual match result."""
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    matchweek: int
    outcome: str = ""  # 'H', 'D', 'A'

    def __post_init__(self):
        if not self.outcome:
            if self.home_goals > self.away_goals:
                self.outcome = 'H'
            elif self.home_goals < self.away_goals:
                self.outcome = 'A'
            else:
                self.outcome = 'D'


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for a prediction source."""
    source: str
    n_matches: int = 0
    brier_score: float = 0.0
    log_loss: float = 0.0
    accuracy: float = 0.0
    home_accuracy: float = 0.0
    draw_accuracy: float = 0.0
    away_accuracy: float = 0.0
    roi: float = 0.0  # Return on investment
    calibration: Dict[str, Dict] = field(default_factory=dict)


# Team name normalization (match different sources)
TEAM_NAME_MAP = {
    # football-data.co.uk names
    "Man United": "Manchester Utd",
    "Man City": "Manchester City",
    "Newcastle": "Newcastle Utd",
    "Nott'm Forest": "Nott'ham Forest",
    "Nottingham Forest": "Nott'ham Forest",
    "Spurs": "Tottenham",
    "Wolves": "Wolves",
    "West Ham": "West Ham",
    "Brighton": "Brighton",
    "Bournemouth": "Bournemouth",
    # Opta/The Analyst names
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Nottingham Forest": "Nott'ham Forest",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Brighton & Hove Albion": "Brighton",
    "AFC Bournemouth": "Bournemouth",
    "Wolverhampton Wanderers": "Wolves",
    "Leeds United": "Leeds United",
    "Leicester": "Leicester City",
}


def normalize_team_name(name: str) -> str:
    """Normalize team name to standard format."""
    name = name.strip()
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    return name


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_url(url: str, timeout: int = 30) -> Optional[str]:
    """Fetch URL content with SSL handling."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as response:
            return response.read().decode('utf-8', errors='replace')
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def fetch_betting_odds(refresh: bool = False) -> List[Dict]:
    """
    Fetch betting odds from football-data.co.uk.

    Returns list of dicts with match data and odds.
    CSV columns include: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR,
                         B365H, B365D, B365A (Bet365 odds), etc.
    """
    ensure_cache_dir()
    cache_file = CACHE_DIR / "betting_odds.csv"

    # Try to load from cache if not refreshing
    if not refresh and cache_file.exists():
        print(f"Loading betting odds from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)

    print(f"Fetching betting odds from {FOOTBALL_DATA_URL}...")
    content = fetch_url(FOOTBALL_DATA_URL)

    if not content:
        print("Failed to fetch betting odds. Using cached data if available.")
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                reader = csv.DictReader(f)
                return list(reader)
        return []

    # Save to cache
    with open(cache_file, 'w') as f:
        f.write(content)

    # Parse CSV
    reader = csv.DictReader(io.StringIO(content))
    return list(reader)


def odds_to_probability(odds: float) -> float:
    """Convert decimal odds to probability."""
    if odds <= 1:
        return 0.0
    return 1.0 / odds


def parse_betting_predictions(odds_data: List[Dict]) -> Dict[Tuple[str, str], MatchPrediction]:
    """
    Parse betting odds into match predictions.

    Uses Bet365 odds (B365H, B365D, B365A) as primary, with fallback to others.
    Normalizes probabilities to sum to 1.0, but also stores actual decimal odds
    for accurate ROI calculation (which includes bookmaker margin).
    """
    predictions = {}

    for row in odds_data:
        try:
            home_team = normalize_team_name(row.get('HomeTeam', ''))
            away_team = normalize_team_name(row.get('AwayTeam', ''))

            if not home_team or not away_team:
                continue

            # Try different bookmakers in order of preference
            odds_sources = [
                ('B365H', 'B365D', 'B365A'),  # Bet365
                ('BWH', 'BWD', 'BWA'),         # Betway
                ('PSH', 'PSD', 'PSA'),         # Pinnacle
                ('WHH', 'WHD', 'WHA'),         # William Hill
                ('AvgH', 'AvgD', 'AvgA'),      # Average odds
            ]

            home_prob = draw_prob = away_prob = 0.0
            actual_home_odds = actual_draw_odds = actual_away_odds = 0.0

            for h_key, d_key, a_key in odds_sources:
                if h_key in row and row[h_key]:
                    try:
                        actual_home_odds = float(row[h_key])
                        actual_draw_odds = float(row[d_key])
                        actual_away_odds = float(row[a_key])

                        home_prob = odds_to_probability(actual_home_odds)
                        draw_prob = odds_to_probability(actual_draw_odds)
                        away_prob = odds_to_probability(actual_away_odds)

                        if home_prob > 0:
                            break
                    except (ValueError, KeyError):
                        continue

            if home_prob == 0:
                continue

            # Normalize to remove bookmaker margin (for fair probability comparison)
            total = home_prob + draw_prob + away_prob
            if total > 0:
                home_prob /= total
                draw_prob /= total
                away_prob /= total

            pred = MatchPrediction(
                home_team=home_team,
                away_team=away_team,
                home_win_prob=home_prob,
                draw_prob=draw_prob,
                away_win_prob=away_prob,
                source='betting',
                # Store actual odds for ROI calculation
                home_odds=actual_home_odds,
                draw_odds=actual_draw_odds,
                away_odds=actual_away_odds,
            )

            predictions[(home_team, away_team)] = pred

        except Exception as e:
            print(f"Error parsing row: {e}")
            continue

    print(f"Parsed {len(predictions)} betting predictions")
    return predictions


def fetch_opta_predictions(refresh: bool = False) -> Dict[Tuple[str, str], MatchPrediction]:
    """
    Fetch Opta predictions from The Analyst.

    Note: The Analyst doesn't have a public API, so this scrapes the predictions page.
    The page format may change, so this may need updates.
    """
    ensure_cache_dir()
    cache_file = CACHE_DIR / "opta_predictions.json"

    # Try to load from cache if not refreshing
    if not refresh and cache_file.exists():
        print(f"Loading Opta predictions from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            data = json.load(f)
            return {
                tuple(k.split('|')): MatchPrediction(**v)
                for k, v in data.items()
            }

    print(f"Fetching Opta predictions from {OPTA_PREDICTIONS_URL}...")
    content = fetch_url(OPTA_PREDICTIONS_URL)

    if not content:
        print("Failed to fetch Opta predictions. Using cached data if available.")
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return {
                    tuple(k.split('|')): MatchPrediction(**v)
                    for k, v in data.items()
                }
        return {}

    predictions = parse_opta_html(content)

    # Save to cache
    cache_data = {
        f"{k[0]}|{k[1]}": {
            'home_team': v.home_team,
            'away_team': v.away_team,
            'home_win_prob': v.home_win_prob,
            'draw_prob': v.draw_prob,
            'away_win_prob': v.away_win_prob,
            'source': v.source,
            'matchweek': v.matchweek,
            'match_id': v.match_id,
        }
        for k, v in predictions.items()
    }
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f, indent=2)

    return predictions


def parse_opta_html(html: str) -> Dict[Tuple[str, str], MatchPrediction]:
    """
    Parse Opta predictions from The Analyst HTML.

    Looks for patterns like:
    - Team A vs Team B
    - Home: XX.X%, Draw: XX.X%, Away: XX.X%

    This is fragile and may need updates if the page format changes.
    """
    predictions = {}

    # Try to find match prediction blocks
    # The format varies, so we try multiple patterns

    # Pattern 1: "Team A X.X% Draw X.X% Team B X.X%"
    pattern1 = r'(\w[\w\s]+?)\s+(\d+\.?\d*)[%]?\s*(?:Draw|draw)\s*(\d+\.?\d*)[%]?\s*(\w[\w\s]+?)\s+(\d+\.?\d*)[%]?'

    # Pattern 2: JSON-like data embedded in the page
    pattern2 = r'"homeTeam":\s*"([^"]+)".*?"awayTeam":\s*"([^"]+)".*?"homeWinProb":\s*(\d+\.?\d*).*?"drawProb":\s*(\d+\.?\d*).*?"awayWinProb":\s*(\d+\.?\d*)'

    # Pattern 3: Simple percentage patterns
    pattern3 = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:vs?\.?|v)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)[^0-9]*(\d+\.?\d*)[%]?\s*[-–/]\s*(\d+\.?\d*)[%]?\s*[-–/]\s*(\d+\.?\d*)[%]?'

    for pattern in [pattern1, pattern2, pattern3]:
        matches = re.findall(pattern, html, re.IGNORECASE | re.DOTALL)
        for match in matches:
            try:
                if len(match) == 5:
                    if 'homeTeam' in pattern:
                        home_team, away_team, home_prob, draw_prob, away_prob = match
                    else:
                        home_team, home_prob, draw_prob, away_team, away_prob = match

                    home_team = normalize_team_name(home_team)
                    away_team = normalize_team_name(away_team)

                    # Convert to float and normalize
                    home_prob = float(home_prob)
                    draw_prob = float(draw_prob)
                    away_prob = float(away_prob)

                    # If percentages are > 1, they're already in percent form
                    if home_prob > 1 or draw_prob > 1 or away_prob > 1:
                        home_prob /= 100
                        draw_prob /= 100
                        away_prob /= 100

                    # Skip if doesn't look like valid probabilities
                    total = home_prob + draw_prob + away_prob
                    if total < 0.9 or total > 1.1:
                        continue

                    # Normalize
                    home_prob /= total
                    draw_prob /= total
                    away_prob /= total

                    pred = MatchPrediction(
                        home_team=home_team,
                        away_team=away_team,
                        home_win_prob=home_prob,
                        draw_prob=draw_prob,
                        away_win_prob=away_prob,
                        source='opta',
                    )

                    predictions[(home_team, away_team)] = pred

            except (ValueError, IndexError):
                continue

    print(f"Parsed {len(predictions)} Opta predictions")
    return predictions


def predict_match_probs_static(
    home_theta: float,
    away_theta: float,
    home_advantage: float = 0.30,
    k_scale: float = 1.0,
    b0_draw: float = 0.05,
    c_mismatch: float = 0.50,
) -> Tuple[float, float, float]:
    """
    Static version of predict_match_probs for evaluation script.

    Uses the same algorithm as backend/models/bayesian_engine.py.
    Pure Python implementation (no numpy required).
    """
    def softmax(x: List[float]) -> List[float]:
        """Numerically stable softmax in pure Python."""
        max_x = max(x)
        e_x = [math.exp(val - max_x) for val in x]
        sum_e_x = sum(e_x)
        return [val / sum_e_x for val in e_x]

    delta = home_theta + home_advantage - away_theta

    z_away = -k_scale * delta
    z_draw = b0_draw - c_mismatch * abs(delta)
    z_home = k_scale * delta

    probs = softmax([z_away, z_draw, z_home])

    return probs[2], probs[1], probs[0]  # home, draw, away


def regenerate_historical_predictions(week: Optional[int] = None) -> Dict[Tuple[str, str], MatchPrediction]:
    """
    Regenerate match predictions using stored historical theta values.

    For each completed match, finds the team state just before the match
    and generates the prediction that would have been made.
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all match results
    if week:
        cursor.execute("""
            SELECT * FROM match_results WHERE matchweek <= ?
            ORDER BY matchweek, date
        """, (week,))
    else:
        cursor.execute("SELECT * FROM match_results ORDER BY matchweek, date")

    match_results = cursor.fetchall()

    predictions = {}

    for match in match_results:
        matchweek = match['matchweek']
        home_team = match['home_team']
        away_team = match['away_team']

        # Get team states from the week before the match
        # (prediction made before match was played)
        prev_week = matchweek - 1 if matchweek > 0 else 0

        cursor.execute("""
            SELECT theta_home, theta_away, sigma
            FROM team_state_history
            WHERE team = ? AND week = ?
        """, (home_team, prev_week))
        home_state = cursor.fetchone()

        cursor.execute("""
            SELECT theta_home, theta_away, sigma
            FROM team_state_history
            WHERE team = ? AND week = ?
        """, (away_team, prev_week))
        away_state = cursor.fetchone()

        if not home_state or not away_state:
            # Try week 0 as fallback
            cursor.execute("""
                SELECT theta_home, theta_away, sigma
                FROM team_state_history
                WHERE team = ? AND week = 0
            """, (home_team,))
            home_state = cursor.fetchone()

            cursor.execute("""
                SELECT theta_home, theta_away, sigma
                FROM team_state_history
                WHERE team = ? AND week = 0
            """, (away_team,))
            away_state = cursor.fetchone()

        if not home_state or not away_state:
            continue

        # Generate prediction using stored thetas
        home_prob, draw_prob, away_prob = predict_match_probs_static(
            home_theta=home_state['theta_home'],
            away_theta=away_state['theta_away'],
        )

        pred = MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            home_win_prob=home_prob,
            draw_prob=draw_prob,
            away_win_prob=away_prob,
            source='motson',
            matchweek=matchweek,
            match_id=match['match_id'],
        )

        predictions[(home_team, away_team)] = pred

    conn.close()

    print(f"Regenerated {len(predictions)} historical predictions from stored team states")
    return predictions


def load_motson_predictions(week: Optional[int] = None, regenerate: bool = True) -> Tuple[Dict[Tuple[str, str], MatchPrediction], List[MatchResult]]:
    """
    Load MOTSON predictions and match results from database.

    Args:
        week: Optional week limit
        regenerate: If True, regenerate historical predictions from stored theta values
                   (recommended, as stored match_predictions only has future matches)

    Returns:
        - Dict of (home, away) -> MatchPrediction
        - List of MatchResult for completed matches
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get match results
    if week:
        cursor.execute("""
            SELECT * FROM match_results WHERE matchweek <= ?
        """, (week,))
    else:
        cursor.execute("SELECT * FROM match_results")

    results = []
    for row in cursor.fetchall():
        result = MatchResult(
            home_team=row['home_team'],
            away_team=row['away_team'],
            home_goals=row['home_goals'],
            away_goals=row['away_goals'],
            matchweek=row['matchweek'],
        )
        results.append(result)

    conn.close()

    # Regenerate predictions from historical theta values
    if regenerate:
        predictions = regenerate_historical_predictions(week)
    else:
        # Load stored predictions (only has future matches)
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if week:
            cursor.execute("""
                SELECT * FROM match_predictions WHERE matchweek <= ?
            """, (week,))
        else:
            cursor.execute("SELECT * FROM match_predictions")

        predictions = {}
        for row in cursor.fetchall():
            pred = MatchPrediction(
                home_team=row['home_team'],
                away_team=row['away_team'],
                home_win_prob=row['home_win_prob'],
                draw_prob=row['draw_prob'],
                away_win_prob=row['away_win_prob'],
                source='motson',
                matchweek=row['matchweek'],
                match_id=row['match_id'],
            )
            predictions[(pred.home_team, pred.away_team)] = pred

        conn.close()

    print(f"Loaded {len(predictions)} MOTSON predictions and {len(results)} match results")
    return predictions, results


def calculate_brier_score(predictions: List[MatchPrediction], results: List[MatchResult]) -> float:
    """
    Calculate Brier Score for multiclass predictions.

    Brier Score = (1/N) * sum((p_h - y_h)^2 + (p_d - y_d)^2 + (p_a - y_a)^2)

    Where y_h, y_d, y_a are 1 for the actual outcome, 0 otherwise.
    Lower is better. Perfect = 0, worst = 2 (for 3-way).
    """
    if not predictions or not results:
        return float('inf')

    results_map = {(r.home_team, r.away_team): r for r in results}

    total_score = 0.0
    n = 0

    for pred in predictions:
        key = (pred.home_team, pred.away_team)
        if key not in results_map:
            continue

        result = results_map[key]

        # Actual outcome as one-hot
        y_h = 1.0 if result.outcome == 'H' else 0.0
        y_d = 1.0 if result.outcome == 'D' else 0.0
        y_a = 1.0 if result.outcome == 'A' else 0.0

        # Brier component
        score = (pred.home_win_prob - y_h)**2 + (pred.draw_prob - y_d)**2 + (pred.away_win_prob - y_a)**2
        total_score += score
        n += 1

    return total_score / n if n > 0 else float('inf')


def calculate_log_loss(predictions: List[MatchPrediction], results: List[MatchResult]) -> float:
    """
    Calculate Log Loss for multiclass predictions.

    Log Loss = -(1/N) * sum(y_h*log(p_h) + y_d*log(p_d) + y_a*log(p_a))

    Lower is better. Uses epsilon to avoid log(0).
    """
    if not predictions or not results:
        return float('inf')

    results_map = {(r.home_team, r.away_team): r for r in results}

    total_loss = 0.0
    n = 0
    epsilon = 1e-15  # Avoid log(0)

    for pred in predictions:
        key = (pred.home_team, pred.away_team)
        if key not in results_map:
            continue

        result = results_map[key]

        # Clip probabilities
        p_h = max(epsilon, min(1 - epsilon, pred.home_win_prob))
        p_d = max(epsilon, min(1 - epsilon, pred.draw_prob))
        p_a = max(epsilon, min(1 - epsilon, pred.away_win_prob))

        # Log loss component
        if result.outcome == 'H':
            loss = -math.log(p_h)
        elif result.outcome == 'D':
            loss = -math.log(p_d)
        else:
            loss = -math.log(p_a)

        total_loss += loss
        n += 1

    return total_loss / n if n > 0 else float('inf')


def calculate_accuracy(predictions: List[MatchPrediction], results: List[MatchResult]) -> Dict[str, float]:
    """
    Calculate prediction accuracy.

    Returns dict with overall accuracy and per-outcome accuracy.
    """
    results_map = {(r.home_team, r.away_team): r for r in results}

    correct = 0
    home_correct = home_total = 0
    draw_correct = draw_total = 0
    away_correct = away_total = 0
    n = 0

    for pred in predictions:
        key = (pred.home_team, pred.away_team)
        if key not in results_map:
            continue

        result = results_map[key]

        # Predicted outcome (highest probability)
        if pred.home_win_prob >= pred.draw_prob and pred.home_win_prob >= pred.away_win_prob:
            predicted = 'H'
        elif pred.draw_prob >= pred.away_win_prob:
            predicted = 'D'
        else:
            predicted = 'A'

        n += 1
        if predicted == result.outcome:
            correct += 1

        # Per-outcome tracking
        if result.outcome == 'H':
            home_total += 1
            if predicted == 'H':
                home_correct += 1
        elif result.outcome == 'D':
            draw_total += 1
            if predicted == 'D':
                draw_correct += 1
        else:
            away_total += 1
            if predicted == 'A':
                away_correct += 1

    return {
        'overall': correct / n if n > 0 else 0.0,
        'home': home_correct / home_total if home_total > 0 else 0.0,
        'draw': draw_correct / draw_total if draw_total > 0 else 0.0,
        'away': away_correct / away_total if away_total > 0 else 0.0,
        'n_matches': n,
    }


def calculate_calibration(predictions: List[MatchPrediction], results: List[MatchResult], n_bins: int = 10) -> Dict:
    """
    Calculate calibration - how well predicted probabilities match actual frequencies.

    Groups predictions into bins by predicted probability and calculates
    actual frequency of that outcome occurring.

    Perfect calibration: predicted probability = actual frequency in each bin.
    """
    results_map = {(r.home_team, r.away_team): r for r in results}

    # Initialize bins for each outcome
    calibration = {
        'home': {'bins': [], 'predicted': [], 'actual': [], 'count': []},
        'draw': {'bins': [], 'predicted': [], 'actual': [], 'count': []},
        'away': {'bins': [], 'predicted': [], 'actual': [], 'count': []},
    }

    bin_edges = [i / n_bins for i in range(n_bins + 1)]

    for outcome_key, prob_attr, actual_val in [
        ('home', 'home_win_prob', 'H'),
        ('draw', 'draw_prob', 'D'),
        ('away', 'away_win_prob', 'A'),
    ]:
        bin_preds = [[] for _ in range(n_bins)]
        bin_actuals = [[] for _ in range(n_bins)]

        for pred in predictions:
            key = (pred.home_team, pred.away_team)
            if key not in results_map:
                continue

            result = results_map[key]
            prob = getattr(pred, prob_attr)
            actual = 1.0 if result.outcome == actual_val else 0.0

            # Find bin
            bin_idx = min(int(prob * n_bins), n_bins - 1)
            bin_preds[bin_idx].append(prob)
            bin_actuals[bin_idx].append(actual)

        for i in range(n_bins):
            bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
            if bin_preds[i]:
                avg_pred = sum(bin_preds[i]) / len(bin_preds[i])
                avg_actual = sum(bin_actuals[i]) / len(bin_actuals[i])
                count = len(bin_preds[i])
            else:
                avg_pred = bin_center
                avg_actual = 0.0
                count = 0

            calibration[outcome_key]['bins'].append(bin_center)
            calibration[outcome_key]['predicted'].append(avg_pred)
            calibration[outcome_key]['actual'].append(avg_actual)
            calibration[outcome_key]['count'].append(count)

    return calibration


def calculate_roi(predictions: List[MatchPrediction], results: List[MatchResult],
                  betting_odds: Dict[Tuple[str, str], MatchPrediction],
                  stake: float = 1.0, edge_threshold: float = 0.05) -> Dict:
    """
    Calculate return on investment if betting based on MOTSON predictions.

    Strategy: Bet on outcomes where MOTSON probability > implied odds probability + edge_threshold

    Returns:
        - total_staked: Total amount bet
        - total_returned: Total winnings
        - roi: Return on investment percentage
        - n_bets: Number of bets placed
    """
    results_map = {(r.home_team, r.away_team): r for r in results}

    total_staked = 0.0
    total_returned = 0.0
    n_bets = 0
    wins = 0

    bet_details = []

    for pred in predictions:
        key = (pred.home_team, pred.away_team)
        if key not in results_map or key not in betting_odds:
            continue

        result = results_map[key]
        odds_pred = betting_odds[key]

        # Check each outcome for edge
        # Use actual bookmaker odds for ROI calculation (includes their margin)
        for outcome, motson_prob, odds_prob, actual_odds in [
            ('H', pred.home_win_prob, odds_pred.home_win_prob, odds_pred.home_odds),
            ('D', pred.draw_prob, odds_pred.draw_prob, odds_pred.draw_odds),
            ('A', pred.away_win_prob, odds_pred.away_win_prob, odds_pred.away_odds),
        ]:
            edge = motson_prob - odds_prob
            if edge > edge_threshold:
                # Place bet
                total_staked += stake
                n_bets += 1

                # Use actual bookmaker odds (not fair odds derived from normalized probs)
                decimal_odds = actual_odds if actual_odds > 0 else (1.0 / odds_prob if odds_prob > 0 else 0)

                if result.outcome == outcome:
                    returns = stake * decimal_odds
                    total_returned += returns
                    wins += 1

                bet_details.append({
                    'match': f"{pred.home_team} vs {pred.away_team}",
                    'bet': outcome,
                    'motson_prob': motson_prob,
                    'odds_prob': odds_prob,
                    'actual_odds': decimal_odds,
                    'edge': edge,
                    'result': result.outcome,
                    'won': result.outcome == outcome,
                })

    roi = ((total_returned - total_staked) / total_staked * 100) if total_staked > 0 else 0.0

    return {
        'total_staked': total_staked,
        'total_returned': total_returned,
        'roi': roi,
        'n_bets': n_bets,
        'wins': wins,
        'win_rate': wins / n_bets if n_bets > 0 else 0.0,
        'details': bet_details,
    }


def calculate_betting_strategies(
    predictions: List[MatchPrediction],
    results: List[MatchResult],
    betting_odds: Dict[Tuple[str, str], MatchPrediction],
    initial_bankroll: float = 1000.0,
) -> Dict:
    """
    Analyze multiple betting strategies to determine profitability.

    Strategies tested:
    1. Value Betting (various edge thresholds: 2%, 5%, 10%, 15%)
    2. Kelly Criterion (full and fractional)
    3. Flat betting on favorites
    4. Flat betting on underdogs

    Returns comprehensive analysis including:
    - ROI for each strategy
    - Bankroll progression
    - Best/worst bets
    - Profit by matchweek
    """
    results_map = {(r.home_team, r.away_team): r for r in results}

    # Sort predictions by matchweek for bankroll tracking
    sorted_preds = sorted(
        [p for p in predictions if (p.home_team, p.away_team) in results_map],
        key=lambda p: p.matchweek
    )

    strategies = {}

    # Strategy 1: Value Betting at different edge thresholds
    for edge_threshold in [0.02, 0.05, 0.10, 0.15]:
        strategy_name = f"value_{int(edge_threshold*100)}pct"
        bankroll = initial_bankroll
        bankroll_history = [bankroll]
        bets = []
        stake = initial_bankroll * 0.02  # 2% of initial bankroll per bet

        for pred in sorted_preds:
            key = (pred.home_team, pred.away_team)
            if key not in betting_odds:
                continue

            result = results_map[key]
            odds_pred = betting_odds[key]

            # Use actual bookmaker odds for realistic ROI
            for outcome, motson_prob, odds_prob, actual_odds in [
                ('H', pred.home_win_prob, odds_pred.home_win_prob, odds_pred.home_odds),
                ('D', pred.draw_prob, odds_pred.draw_prob, odds_pred.draw_odds),
                ('A', pred.away_win_prob, odds_pred.away_win_prob, odds_pred.away_odds),
            ]:
                edge = motson_prob - odds_prob
                if edge > edge_threshold and bankroll >= stake:
                    # Use actual odds (with bookmaker margin) not fair odds
                    decimal_odds = actual_odds if actual_odds > 0 else (1.0 / odds_prob if odds_prob > 0 else 0)
                    won = result.outcome == outcome
                    profit = (stake * decimal_odds - stake) if won else -stake
                    bankroll += profit

                    bets.append({
                        'match': f"{pred.home_team} vs {pred.away_team}",
                        'week': pred.matchweek,
                        'outcome': outcome,
                        'edge': edge,
                        'odds': decimal_odds,
                        'won': won,
                        'profit': profit,
                        'bankroll': bankroll,
                    })

            bankroll_history.append(bankroll)

        strategies[strategy_name] = {
            'final_bankroll': bankroll,
            'profit': bankroll - initial_bankroll,
            'roi': ((bankroll - initial_bankroll) / initial_bankroll) * 100,
            'n_bets': len(bets),
            'wins': sum(1 for b in bets if b['won']),
            'win_rate': sum(1 for b in bets if b['won']) / len(bets) if bets else 0,
            'avg_edge': sum(b['edge'] for b in bets) / len(bets) if bets else 0,
            'max_drawdown': calculate_max_drawdown(bankroll_history),
            'bankroll_history': bankroll_history,
            'bets': bets,
        }

    # Strategy 2: Kelly Criterion (fractional - 25% Kelly for safety)
    for kelly_fraction in [0.25, 0.5, 1.0]:
        strategy_name = f"kelly_{int(kelly_fraction*100)}pct"
        bankroll = initial_bankroll
        bankroll_history = [bankroll]
        bets = []
        min_edge = 0.05  # Only bet with 5% edge minimum

        for pred in sorted_preds:
            key = (pred.home_team, pred.away_team)
            if key not in betting_odds:
                continue

            result = results_map[key]
            odds_pred = betting_odds[key]

            # Use actual bookmaker odds for realistic ROI
            for outcome, motson_prob, odds_prob, actual_odds in [
                ('H', pred.home_win_prob, odds_pred.home_win_prob, odds_pred.home_odds),
                ('D', pred.draw_prob, odds_pred.draw_prob, odds_pred.draw_odds),
                ('A', pred.away_win_prob, odds_pred.away_win_prob, odds_pred.away_odds),
            ]:
                edge = motson_prob - odds_prob
                if edge > min_edge:
                    # Use actual odds (with bookmaker margin) not fair odds
                    decimal_odds = actual_odds if actual_odds > 0 else (1.0 / odds_prob if odds_prob > 0 else 0)

                    # Kelly formula: f* = (bp - q) / b
                    # where b = decimal_odds - 1, p = our probability, q = 1-p
                    b = decimal_odds - 1
                    p = motson_prob
                    q = 1 - p

                    if b > 0:
                        kelly_stake = (b * p - q) / b
                        kelly_stake = max(0, kelly_stake) * kelly_fraction
                        stake = min(kelly_stake * bankroll, bankroll * 0.25)  # Cap at 25%

                        if stake > 0 and bankroll >= stake:
                            won = result.outcome == outcome
                            profit = (stake * decimal_odds - stake) if won else -stake
                            bankroll += profit

                            bets.append({
                                'match': f"{pred.home_team} vs {pred.away_team}",
                                'week': pred.matchweek,
                                'outcome': outcome,
                                'edge': edge,
                                'kelly_pct': kelly_stake * 100,
                                'stake': stake,
                                'odds': decimal_odds,
                                'won': won,
                                'profit': profit,
                                'bankroll': bankroll,
                            })

            bankroll_history.append(bankroll)

        strategies[strategy_name] = {
            'final_bankroll': bankroll,
            'profit': bankroll - initial_bankroll,
            'roi': ((bankroll - initial_bankroll) / initial_bankroll) * 100,
            'n_bets': len(bets),
            'wins': sum(1 for b in bets if b['won']),
            'win_rate': sum(1 for b in bets if b['won']) / len(bets) if bets else 0,
            'avg_stake_pct': sum(b['stake'] for b in bets) / (len(bets) * initial_bankroll) * 100 if bets else 0,
            'max_drawdown': calculate_max_drawdown(bankroll_history),
            'bankroll_history': bankroll_history,
            'bets': bets,
        }

    # Strategy 3: Bet on MOTSON's predicted winner (flat stake)
    strategy_name = "predicted_winner"
    bankroll = initial_bankroll
    bankroll_history = [bankroll]
    bets = []
    stake = initial_bankroll * 0.02

    for pred in sorted_preds:
        key = (pred.home_team, pred.away_team)
        if key not in betting_odds or bankroll < stake:
            continue

        result = results_map[key]
        odds_pred = betting_odds[key]

        # Find predicted winner and get actual bookmaker odds
        if pred.home_win_prob >= pred.draw_prob and pred.home_win_prob >= pred.away_win_prob:
            predicted = 'H'
            odds_prob = odds_pred.home_win_prob
            actual_odds = odds_pred.home_odds
        elif pred.draw_prob >= pred.away_win_prob:
            predicted = 'D'
            odds_prob = odds_pred.draw_prob
            actual_odds = odds_pred.draw_odds
        else:
            predicted = 'A'
            odds_prob = odds_pred.away_win_prob
            actual_odds = odds_pred.away_odds

        # Use actual odds (with bookmaker margin) not fair odds
        decimal_odds = actual_odds if actual_odds > 0 else (1.0 / odds_prob if odds_prob > 0 else 0)
        won = result.outcome == predicted
        profit = (stake * decimal_odds - stake) if won else -stake
        bankroll += profit

        bets.append({
            'match': f"{pred.home_team} vs {pred.away_team}",
            'week': pred.matchweek,
            'predicted': predicted,
            'actual': result.outcome,
            'odds': decimal_odds,
            'won': won,
            'profit': profit,
            'bankroll': bankroll,
        })

        bankroll_history.append(bankroll)

    strategies[strategy_name] = {
        'final_bankroll': bankroll,
        'profit': bankroll - initial_bankroll,
        'roi': ((bankroll - initial_bankroll) / initial_bankroll) * 100,
        'n_bets': len(bets),
        'wins': sum(1 for b in bets if b['won']),
        'win_rate': sum(1 for b in bets if b['won']) / len(bets) if bets else 0,
        'max_drawdown': calculate_max_drawdown(bankroll_history),
        'bankroll_history': bankroll_history,
        'bets': bets,
    }

    # Find best and worst strategies
    best_strategy = max(strategies.items(), key=lambda x: x[1]['roi'])
    worst_strategy = min(strategies.items(), key=lambda x: x[1]['roi'])

    # Calculate profit by week for the best value strategy
    profit_by_week = {}
    if 'value_5pct' in strategies:
        for bet in strategies['value_5pct']['bets']:
            week = bet['week']
            if week not in profit_by_week:
                profit_by_week[week] = 0
            profit_by_week[week] += bet['profit']

    return {
        'strategies': strategies,
        'best_strategy': {
            'name': best_strategy[0],
            'roi': best_strategy[1]['roi'],
            'profit': best_strategy[1]['profit'],
        },
        'worst_strategy': {
            'name': worst_strategy[0],
            'roi': worst_strategy[1]['roi'],
            'profit': worst_strategy[1]['profit'],
        },
        'profit_by_week': profit_by_week,
        'initial_bankroll': initial_bankroll,
    }


def calculate_max_drawdown(bankroll_history: List[float]) -> float:
    """Calculate maximum drawdown from peak."""
    if not bankroll_history:
        return 0.0

    peak = bankroll_history[0]
    max_dd = 0.0

    for value in bankroll_history:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak if peak > 0 else 0
        max_dd = max(max_dd, drawdown)

    return max_dd * 100  # Return as percentage


def print_betting_analysis(betting_analysis: Dict):
    """Print detailed betting strategy analysis."""
    print("\n" + "=" * 80)
    print("BETTING STRATEGY ANALYSIS")
    print("=" * 80)
    print(f"Initial Bankroll: ${betting_analysis['initial_bankroll']:.2f}")
    print()

    # Summary table
    print("-" * 80)
    print(f"{'Strategy':<20} {'Bets':>6} {'Wins':>6} {'Win%':>8} {'ROI':>10} {'Final $':>12} {'Max DD':>8}")
    print("-" * 80)

    for name, data in sorted(betting_analysis['strategies'].items(), key=lambda x: -x[1]['roi']):
        print(f"{name:<20} {data['n_bets']:>6} {data['wins']:>6} {data['win_rate']*100:>7.1f}% "
              f"{data['roi']:>+9.1f}% ${data['final_bankroll']:>11.2f} {data['max_drawdown']:>7.1f}%")

    print("-" * 80)

    # Best strategy highlight
    best = betting_analysis['best_strategy']
    print(f"\n🏆 BEST STRATEGY: {best['name']}")
    print(f"   ROI: {best['roi']:+.1f}%")
    print(f"   Profit: ${best['profit']:+.2f}")

    # Interpretation
    print("\n" + "-" * 80)
    print("INTERPRETATION")
    print("-" * 80)

    if best['roi'] > 0:
        print(f"✅ MOTSON shows PROFITABLE edge against bookmakers!")
        print(f"   Using {best['name']} strategy would have grown $1000 to ${1000 + best['profit']/betting_analysis['initial_bankroll']*1000:.2f}")
    else:
        print(f"⚠️  No profitable strategy found against bookmaker odds")
        print(f"   Best result: {best['roi']:.1f}% ROI")

    # Value betting breakdown
    if 'value_5pct' in betting_analysis['strategies']:
        v5 = betting_analysis['strategies']['value_5pct']
        print(f"\n📊 Value Betting (5% edge threshold):")
        print(f"   - {v5['n_bets']} bets placed")
        print(f"   - Average edge: {v5.get('avg_edge', 0)*100:.1f}%")
        print(f"   - Win rate: {v5['win_rate']*100:.1f}%")

    print()


def evaluate_source(predictions: Dict[Tuple[str, str], MatchPrediction],
                    results: List[MatchResult],
                    source_name: str) -> EvaluationMetrics:
    """Evaluate predictions from a single source."""
    pred_list = list(predictions.values())

    brier = calculate_brier_score(pred_list, results)
    log_loss = calculate_log_loss(pred_list, results)
    accuracy = calculate_accuracy(pred_list, results)
    calibration = calculate_calibration(pred_list, results)

    return EvaluationMetrics(
        source=source_name,
        n_matches=accuracy['n_matches'],
        brier_score=brier,
        log_loss=log_loss,
        accuracy=accuracy['overall'],
        home_accuracy=accuracy['home'],
        draw_accuracy=accuracy['draw'],
        away_accuracy=accuracy['away'],
        calibration=calibration,
    )


def print_comparison_report(motson_metrics: EvaluationMetrics,
                           betting_metrics: Optional[EvaluationMetrics],
                           opta_metrics: Optional[EvaluationMetrics],
                           roi_results: Optional[Dict]):
    """Print a formatted comparison report."""
    print("\n" + "=" * 80)
    print("MOTSON PREDICTION EVALUATION REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Summary table
    print("-" * 80)
    print(f"{'Metric':<25} {'MOTSON':>12} {'Betting':>12} {'Opta':>12}")
    print("-" * 80)

    def fmt(val, is_pct=False):
        if val is None or val == float('inf'):
            return "N/A"
        if is_pct:
            return f"{val:.1%}"
        return f"{val:.4f}"

    betting_brier = betting_metrics.brier_score if betting_metrics else None
    betting_log = betting_metrics.log_loss if betting_metrics else None
    betting_acc = betting_metrics.accuracy if betting_metrics else None

    opta_brier = opta_metrics.brier_score if opta_metrics else None
    opta_log = opta_metrics.log_loss if opta_metrics else None
    opta_acc = opta_metrics.accuracy if opta_metrics else None

    print(f"{'Matches Evaluated':<25} {motson_metrics.n_matches:>12} {betting_metrics.n_matches if betting_metrics else 'N/A':>12} {opta_metrics.n_matches if opta_metrics else 'N/A':>12}")
    print(f"{'Brier Score':<25} {fmt(motson_metrics.brier_score):>12} {fmt(betting_brier):>12} {fmt(opta_brier):>12}")
    print(f"{'Log Loss':<25} {fmt(motson_metrics.log_loss):>12} {fmt(betting_log):>12} {fmt(opta_log):>12}")
    print(f"{'Accuracy':<25} {fmt(motson_metrics.accuracy, True):>12} {fmt(betting_acc, True):>12} {fmt(opta_acc, True):>12}")
    print("-" * 80)

    # Per-outcome accuracy
    print("\nPer-Outcome Accuracy (MOTSON):")
    print(f"  Home wins:  {motson_metrics.home_accuracy:.1%}")
    print(f"  Draws:      {motson_metrics.draw_accuracy:.1%}")
    print(f"  Away wins:  {motson_metrics.away_accuracy:.1%}")

    # ROI analysis
    if roi_results:
        print("\n" + "-" * 80)
        print("ROI ANALYSIS (Betting on MOTSON edge > 5%)")
        print("-" * 80)
        print(f"  Bets placed:    {roi_results['n_bets']}")
        print(f"  Wins:           {roi_results['wins']} ({roi_results['win_rate']:.1%})")
        print(f"  Total staked:   ${roi_results['total_staked']:.2f}")
        print(f"  Total returned: ${roi_results['total_returned']:.2f}")
        print(f"  ROI:            {roi_results['roi']:+.1f}%")

        if roi_results['roi'] > 0:
            print(f"  → MOTSON shows positive expected value!")
        else:
            print(f"  → MOTSON would lose money at these odds")

    # Interpretation
    print("\n" + "-" * 80)
    print("INTERPRETATION")
    print("-" * 80)

    # Brier Score comparison
    if betting_brier and betting_brier != float('inf'):
        brier_diff = motson_metrics.brier_score - betting_brier
        if brier_diff < -0.01:
            print(f"✓ MOTSON Brier Score is {abs(brier_diff):.4f} BETTER than betting markets")
        elif brier_diff > 0.01:
            print(f"✗ MOTSON Brier Score is {brier_diff:.4f} WORSE than betting markets")
        else:
            print(f"≈ MOTSON Brier Score is similar to betting markets (diff: {brier_diff:.4f})")

    if opta_brier and opta_brier != float('inf'):
        brier_diff = motson_metrics.brier_score - opta_brier
        if brier_diff < -0.01:
            print(f"✓ MOTSON Brier Score is {abs(brier_diff):.4f} BETTER than Opta")
        elif brier_diff > 0.01:
            print(f"✗ MOTSON Brier Score is {brier_diff:.4f} WORSE than Opta")
        else:
            print(f"≈ MOTSON Brier Score is similar to Opta (diff: {brier_diff:.4f})")

    print()
    print("Brier Score: 0 = perfect, 2 = worst possible for 3-way classification")
    print("Log Loss: Lower is better. ~1.1 is typical for EPL predictions")
    print("=" * 80)


def save_evaluation_results(motson_metrics: EvaluationMetrics,
                           betting_metrics: Optional[EvaluationMetrics],
                           opta_metrics: Optional[EvaluationMetrics],
                           roi_results: Optional[Dict],
                           betting_analysis: Optional[Dict],
                           output_file: Path):
    """Save evaluation results to JSON file."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'motson': {
            'n_matches': motson_metrics.n_matches,
            'brier_score': motson_metrics.brier_score,
            'log_loss': motson_metrics.log_loss,
            'accuracy': motson_metrics.accuracy,
            'home_accuracy': motson_metrics.home_accuracy,
            'draw_accuracy': motson_metrics.draw_accuracy,
            'away_accuracy': motson_metrics.away_accuracy,
            'calibration': motson_metrics.calibration,
        },
    }

    if betting_metrics:
        results['betting'] = {
            'n_matches': betting_metrics.n_matches,
            'brier_score': betting_metrics.brier_score,
            'log_loss': betting_metrics.log_loss,
            'accuracy': betting_metrics.accuracy,
        }

    if opta_metrics:
        results['opta'] = {
            'n_matches': opta_metrics.n_matches,
            'brier_score': opta_metrics.brier_score,
            'log_loss': opta_metrics.log_loss,
            'accuracy': opta_metrics.accuracy,
        }

    if roi_results:
        results['roi'] = {
            'n_bets': roi_results['n_bets'],
            'wins': roi_results['wins'],
            'win_rate': roi_results['win_rate'],
            'total_staked': roi_results['total_staked'],
            'total_returned': roi_results['total_returned'],
            'roi': roi_results['roi'],
        }

    if betting_analysis:
        # Save summary of betting strategies (without full bet history to keep file size reasonable)
        results['betting_strategies'] = {
            'initial_bankroll': betting_analysis['initial_bankroll'],
            'best_strategy': betting_analysis['best_strategy'],
            'worst_strategy': betting_analysis['worst_strategy'],
            'strategies': {
                name: {
                    'final_bankroll': data['final_bankroll'],
                    'profit': data['profit'],
                    'roi': data['roi'],
                    'n_bets': data['n_bets'],
                    'wins': data['wins'],
                    'win_rate': data['win_rate'],
                    'max_drawdown': data['max_drawdown'],
                }
                for name, data in betting_analysis['strategies'].items()
            },
            'profit_by_week': betting_analysis['profit_by_week'],
        }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def load_betting_odds_from_csv(csv_path: Path) -> Dict[Tuple[str, str], MatchPrediction]:
    """
    Load betting odds from a local CSV file.

    Expected format (football-data.co.uk style):
        Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,...

    Or simplified format:
        HomeTeam,AwayTeam,HomeOdds,DrawOdds,AwayOdds
    """
    predictions = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                # Get team names
                home_team = normalize_team_name(row.get('HomeTeam', row.get('home_team', '')))
                away_team = normalize_team_name(row.get('AwayTeam', row.get('away_team', '')))

                if not home_team or not away_team:
                    continue

                # Try to find odds columns
                home_odds = draw_odds = away_odds = None

                # Check for decimal odds columns
                for h_key, d_key, a_key in [
                    ('B365H', 'B365D', 'B365A'),
                    ('PSH', 'PSD', 'PSA'),
                    ('WHH', 'WHD', 'WHA'),
                    ('AvgH', 'AvgD', 'AvgA'),
                    ('HomeOdds', 'DrawOdds', 'AwayOdds'),
                    ('home_odds', 'draw_odds', 'away_odds'),
                ]:
                    if h_key in row and row[h_key]:
                        try:
                            home_odds = float(row[h_key])
                            draw_odds = float(row[d_key])
                            away_odds = float(row[a_key])
                            break
                        except (ValueError, KeyError):
                            continue

                # Check for probability columns (already normalized)
                if home_odds is None:
                    for h_key, d_key, a_key in [
                        ('HomeProb', 'DrawProb', 'AwayProb'),
                        ('home_prob', 'draw_prob', 'away_prob'),
                        ('P_H', 'P_D', 'P_A'),
                    ]:
                        if h_key in row and row[h_key]:
                            try:
                                home_prob = float(row[h_key])
                                draw_prob = float(row[d_key])
                                away_prob = float(row[a_key])

                                # Normalize if needed
                                if home_prob > 1:
                                    home_prob /= 100
                                    draw_prob /= 100
                                    away_prob /= 100

                                total = home_prob + draw_prob + away_prob
                                if total > 0:
                                    home_prob /= total
                                    draw_prob /= total
                                    away_prob /= total

                                pred = MatchPrediction(
                                    home_team=home_team,
                                    away_team=away_team,
                                    home_win_prob=home_prob,
                                    draw_prob=draw_prob,
                                    away_win_prob=away_prob,
                                    source='betting',
                                )
                                predictions[(home_team, away_team)] = pred
                                continue

                            except (ValueError, KeyError):
                                continue

                if home_odds is None:
                    continue

                # Convert decimal odds to probabilities
                home_prob = odds_to_probability(home_odds)
                draw_prob = odds_to_probability(draw_odds)
                away_prob = odds_to_probability(away_odds)

                # Normalize to remove margin
                total = home_prob + draw_prob + away_prob
                if total > 0:
                    home_prob /= total
                    draw_prob /= total
                    away_prob /= total

                pred = MatchPrediction(
                    home_team=home_team,
                    away_team=away_team,
                    home_win_prob=home_prob,
                    draw_prob=draw_prob,
                    away_win_prob=away_prob,
                    source='betting',
                    # Store actual odds for ROI calculation
                    home_odds=home_odds,
                    draw_odds=draw_odds,
                    away_odds=away_odds,
                )
                predictions[(home_team, away_team)] = pred

            except Exception as e:
                print(f"Error parsing row: {e}")
                continue

    print(f"Loaded {len(predictions)} betting predictions from {csv_path}")
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate MOTSON predictions against betting odds and Opta")
    parser.add_argument('--week', type=int, help="Evaluate up to this matchweek (default: all)")
    parser.add_argument('--refresh', action='store_true', help="Refresh cached data from web sources")
    parser.add_argument('--output', type=str, help="Output JSON file path")
    parser.add_argument('--betting-csv', type=str, help="Path to local betting odds CSV file")
    parser.add_argument('--opta-json', type=str, help="Path to local Opta predictions JSON file")
    parser.add_argument('--no-regenerate', action='store_true',
                        help="Don't regenerate historical predictions (use stored only)")
    args = parser.parse_args()

    print("MOTSON Prediction Evaluation")
    print("=" * 40)

    # Load MOTSON predictions and results
    regenerate = not args.no_regenerate
    motson_preds, results = load_motson_predictions(args.week, regenerate=regenerate)

    if not results:
        print("No match results found. Cannot evaluate.")
        return

    # Fetch/load external predictions
    betting_preds = {}
    opta_preds = {}

    # Try local betting CSV first
    if args.betting_csv:
        betting_csv_path = Path(args.betting_csv)
        if betting_csv_path.exists():
            betting_preds = load_betting_odds_from_csv(betting_csv_path)
        else:
            print(f"Warning: Betting CSV not found: {args.betting_csv}")

    # Fall back to web fetch
    if not betting_preds:
        try:
            odds_data = fetch_betting_odds(args.refresh)
            if odds_data:
                betting_preds = parse_betting_predictions(odds_data)
        except Exception as e:
            print(f"Error fetching betting odds: {e}")

    # Load Opta predictions
    if args.opta_json:
        opta_json_path = Path(args.opta_json)
        if opta_json_path.exists():
            with open(opta_json_path, 'r') as f:
                data = json.load(f)
                opta_preds = {
                    tuple(k.split('|')): MatchPrediction(**v)
                    for k, v in data.items()
                }
            print(f"Loaded {len(opta_preds)} Opta predictions from {opta_json_path}")
        else:
            print(f"Warning: Opta JSON not found: {args.opta_json}")

    # Fall back to web fetch
    if not opta_preds:
        try:
            opta_preds = fetch_opta_predictions(args.refresh)
        except Exception as e:
            print(f"Error fetching Opta predictions: {e}")

    # Evaluate each source
    print("\nEvaluating predictions...")

    motson_metrics = evaluate_source(motson_preds, results, "MOTSON")

    betting_metrics = None
    if betting_preds:
        betting_metrics = evaluate_source(betting_preds, results, "Betting")

    opta_metrics = None
    if opta_preds:
        opta_metrics = evaluate_source(opta_preds, results, "Opta")

    # Calculate ROI
    roi_results = None
    betting_analysis = None
    if betting_preds:
        roi_results = calculate_roi(list(motson_preds.values()), results, betting_preds)
        betting_analysis = calculate_betting_strategies(
            list(motson_preds.values()), results, betting_preds
        )

    # Print report
    print_comparison_report(motson_metrics, betting_metrics, opta_metrics, roi_results)

    # Print detailed betting analysis
    if betting_analysis:
        print_betting_analysis(betting_analysis)

    # Save results
    output_file = Path(args.output) if args.output else CACHE_DIR / "evaluation_results.json"
    ensure_cache_dir()
    save_evaluation_results(motson_metrics, betting_metrics, opta_metrics, roi_results, betting_analysis, output_file)


if __name__ == "__main__":
    main()
