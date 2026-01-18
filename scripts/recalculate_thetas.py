#!/usr/bin/env python3
"""
MOTSON Theta Recalculation Script

Fits a Generalized Partial Credit Model (GPCM) or Bradley-Terry-like model
on recent seasons' match results to estimate team strength (theta).

The GPCM treats each match outcome (Loss=0, Draw=1, Win=2) as an ordinal
response, with team strength determining the probability of each outcome.

Usage:
    python scripts/recalculate_thetas.py [--seasons 3] [--output data/MOTSON_GPCM_Theta_Alpha.csv]

Requires FOOTBALL_DATA_API_KEY environment variable.
"""

import os
import sys
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.data_fetcher import FootballDataAPI
from backend.config import app_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MatchData:
    """Container for a single match result."""
    def __init__(self, home_team: str, away_team: str, home_goals: int, away_goals: int, season: int, weight: float = 1.0):
        self.home_team = home_team
        self.away_team = away_team
        self.home_goals = home_goals
        self.away_goals = away_goals
        self.season = season
        self.weight = weight

        # Outcome from home team's perspective: 0=loss, 1=draw, 2=win
        if home_goals > away_goals:
            self.home_outcome = 2
        elif home_goals < away_goals:
            self.home_outcome = 0
        else:
            self.home_outcome = 1


def calculate_match_probs(home_theta: float, away_theta: float, home_advantage: float = 0.3) -> np.ndarray:
    """
    Calculate probability of [away_win, draw, home_win] given team thetas.

    Uses a softmax model similar to the main MOTSON engine.
    """
    delta = home_theta + home_advantage - away_theta

    # Logits for each outcome
    z_away = -delta
    z_draw = 0.0 - 0.5 * abs(delta)  # Draw probability decreases with mismatch
    z_home = delta

    return softmax([z_away, z_draw, z_home])


def negative_log_likelihood(params: np.ndarray, matches: List[MatchData], team_to_idx: Dict[str, int],
                            home_advantage: float = 0.3, regularization: float = 0.1) -> float:
    """
    Calculate negative log-likelihood of observed match outcomes given theta parameters.

    Args:
        params: Array of theta values for each team
        matches: List of match results
        team_to_idx: Mapping from team name to parameter index
        home_advantage: Home field advantage in theta units
        regularization: L2 regularization strength (pulls thetas toward 0)

    Returns:
        Negative log-likelihood (to minimize)
    """
    nll = 0.0

    for match in matches:
        home_idx = team_to_idx.get(match.home_team)
        away_idx = team_to_idx.get(match.away_team)

        if home_idx is None or away_idx is None:
            continue

        home_theta = params[home_idx]
        away_theta = params[away_idx]

        probs = calculate_match_probs(home_theta, away_theta, home_advantage)

        # Outcome: 0=away_win, 1=draw, 2=home_win
        outcome_prob = probs[match.home_outcome]

        # Avoid log(0)
        outcome_prob = max(outcome_prob, 1e-10)

        # Weighted negative log likelihood
        nll -= match.weight * np.log(outcome_prob)

    # L2 regularization to prevent extreme values
    nll += regularization * np.sum(params ** 2)

    return nll


def fit_thetas(matches: List[MatchData], teams: List[str],
               home_advantage: float = 0.3, regularization: float = 0.1) -> Dict[str, float]:
    """
    Fit theta parameters using maximum likelihood estimation.

    Args:
        matches: List of match results
        teams: List of team names to estimate
        home_advantage: Home field advantage
        regularization: L2 regularization strength

    Returns:
        Dictionary mapping team name to theta estimate
    """
    team_to_idx = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    # Initialize with zeros
    initial_params = np.zeros(n_teams)

    # Optimize
    logger.info(f"Fitting {n_teams} team parameters on {len(matches)} matches...")

    result = minimize(
        negative_log_likelihood,
        initial_params,
        args=(matches, team_to_idx, home_advantage, regularization),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'disp': False}
    )

    if not result.success:
        logger.warning(f"Optimization did not fully converge: {result.message}")

    # Extract thetas
    thetas = {team: result.x[team_to_idx[team]] for team in teams}

    # Center thetas (subtract mean so average team has theta ~0)
    mean_theta = np.mean(list(thetas.values()))
    thetas = {team: theta - mean_theta for team, theta in thetas.items()}

    return thetas


def calculate_discrimination(matches: List[MatchData], thetas: Dict[str, float],
                            home_advantage: float = 0.3) -> Dict[str, float]:
    """
    Calculate discrimination parameter (alpha) for each team.

    Alpha represents how consistently the team performs relative to expectations.
    High alpha = consistent performer, low alpha = volatile results.

    For simplicity, we estimate this as inverse variance of residuals.
    """
    team_residuals = defaultdict(list)

    for match in matches:
        if match.home_team not in thetas or match.away_team not in thetas:
            continue

        home_theta = thetas[match.home_team]
        away_theta = thetas[match.away_team]

        probs = calculate_match_probs(home_theta, away_theta, home_advantage)
        expected_outcome = probs[0] * 0 + probs[1] * 1 + probs[2] * 2

        # Residual for home team
        home_residual = match.home_outcome - expected_outcome
        team_residuals[match.home_team].append(home_residual)

        # Residual for away team (inverted)
        away_outcome = 2 - match.home_outcome  # Invert outcome
        away_expected = 2 - expected_outcome
        away_residual = away_outcome - away_expected
        team_residuals[match.away_team].append(away_residual)

    alphas = {}
    for team, residuals in team_residuals.items():
        if len(residuals) > 1:
            variance = np.var(residuals)
            # Alpha inversely related to variance, scaled to reasonable range
            alphas[team] = 1.0 / (1.0 + variance)
        else:
            alphas[team] = 0.85  # Default

    return alphas


async def fetch_season_matches(api: FootballDataAPI, season: int) -> List[MatchData]:
    """Fetch all finished matches for a given season."""
    logger.info(f"Fetching matches for {season}-{season+1} season...")

    try:
        # get_finished_matches returns MatchResult objects with normalized team names
        match_results = await api.get_finished_matches(season=season)

        matches = []
        for m in match_results:
            if m.home_goals is not None and m.away_goals is not None:
                matches.append(MatchData(
                    home_team=m.home_team,
                    away_team=m.away_team,
                    home_goals=m.home_goals,
                    away_goals=m.away_goals,
                    season=season,
                ))

        logger.info(f"  Found {len(matches)} finished matches")
        return matches

    except Exception as e:
        logger.error(f"Failed to fetch {season} season: {e}")
        return []


def apply_season_weights(matches: List[MatchData], current_season: int, decay: float = 0.85) -> List[MatchData]:
    """
    Apply weights to matches based on recency.

    More recent seasons get higher weight, but we use a gentler decay (0.85)
    to preserve "institutional gravity" - the idea that clubs like City/Liverpool
    have structural advantages that persist across seasons.

    With decay=0.85:
      Current: 100%, 1yr: 85%, 2yr: 72%, 3yr: 61%, 4yr: 52%
    """
    for match in matches:
        years_ago = current_season - match.season
        match.weight = decay ** years_ago

    return matches


def map_team_names(matches: List[MatchData], name_map: Dict[str, str]) -> List[MatchData]:
    """Normalize team names using the standard mapping."""
    for match in matches:
        match.home_team = name_map.get(match.home_team, match.home_team)
        match.away_team = name_map.get(match.away_team, match.away_team)
    return matches


async def main():
    parser = argparse.ArgumentParser(description="Recalculate MOTSON theta parameters")
    parser.add_argument("--seasons", type=int, default=3, help="Number of seasons to use (default: 3)")
    parser.add_argument("--output", type=str, default="data/MOTSON_GPCM_Theta_Alpha.csv",
                        help="Output file path")
    parser.add_argument("--home-advantage", type=float, default=0.3,
                        help="Home advantage parameter (default: 0.3)")
    args = parser.parse_args()

    # Check for API key
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        logger.error("FOOTBALL_DATA_API_KEY environment variable not set!")
        logger.info("Set it with: export FOOTBALL_DATA_API_KEY=your_key_here")
        sys.exit(1)

    api = FootballDataAPI(api_key)

    # Determine seasons to fetch
    current_season = app_config.CURRENT_SEASON
    seasons_to_fetch = list(range(current_season - args.seasons + 1, current_season + 1))
    logger.info(f"Fetching seasons: {seasons_to_fetch}")

    # Fetch matches from each season
    all_matches = []
    for season in seasons_to_fetch:
        matches = await fetch_season_matches(api, season)
        all_matches.extend(matches)

    if not all_matches:
        logger.error("No matches fetched! Check API key and network connection.")
        sys.exit(1)

    logger.info(f"Total matches: {len(all_matches)}")

    # Apply recency weights
    all_matches = apply_season_weights(all_matches, current_season)

    # Map team names to standard format
    from backend.config import TEAM_NAME_MAP
    all_matches = map_team_names(all_matches, TEAM_NAME_MAP)

    # Get unique teams
    all_teams = set()
    for match in all_matches:
        all_teams.add(match.home_team)
        all_teams.add(match.away_team)

    teams = sorted(list(all_teams))
    logger.info(f"Teams found: {len(teams)}")

    # Fit thetas
    thetas = fit_thetas(all_matches, teams, home_advantage=args.home_advantage)

    # Calculate alphas (discrimination)
    alphas = calculate_discrimination(all_matches, thetas, home_advantage=args.home_advantage)

    # Sort by theta (strongest first)
    sorted_teams = sorted(thetas.keys(), key=lambda t: thetas[t], reverse=True)

    # Print results
    logger.info("\n" + "="*50)
    logger.info("ESTIMATED TEAM STRENGTHS (Theta)")
    logger.info("="*50)
    for i, team in enumerate(sorted_teams, 1):
        theta = thetas[team]
        alpha = alphas.get(team, 0.85)
        logger.info(f"{i:2}. {team:20} theta={theta:+.3f}  alpha={alpha:.3f}")

    # Save to CSV
    output_path = project_root / args.output
    output_path.parent.mkdir(exist_ok=True)

    df = pd.DataFrame([
        {"Team": team, "Theta": thetas[team], "Alpha": alphas.get(team, 0.85)}
        for team in sorted_teams
    ])
    df.to_csv(output_path, index=False)
    logger.info(f"\nSaved to {output_path}")

    # Also print current season teams that might be missing
    from backend.config import EPL_TEAMS_2025_26
    missing = set(EPL_TEAMS_2025_26) - set(thetas.keys())
    if missing:
        logger.warning(f"\nTeams in current season but not in historical data: {missing}")
        logger.info("These will use position-based theta estimates.")


if __name__ == "__main__":
    asyncio.run(main())
