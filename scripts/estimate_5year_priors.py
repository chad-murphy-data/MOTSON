#!/usr/bin/env python3
"""
MOTSON 5-Year Prior Estimation Script

Estimates the gravity priors (theta, b_home, b_away) from 5 seasons of historical data
using the Bayesian IRT model.

Promoted team handling:
- Teams not in the EPL for 5 seasons get mapped to "promoted slot archetypes"
- This captures the pattern that promoted teams behave like promoted teams,
  regardless of which specific club was promoted

Usage:
    python scripts/estimate_5year_priors.py [--seasons 5]

Requires FOOTBALL_DATA_API_KEY environment variable.
"""

import os
import sys
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import json

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.data_fetcher import FootballDataAPI
from backend.config import app_config, TEAM_NAME_MAP
from backend.models.irt_model import fit_irt_model, IRTParameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Promoted team archetypes
# Map recently promoted teams to archetype slots for prior estimation
ARCHETYPE_SLOTS = {
    "promoted_1": "Archetype_Promoted_1",  # Best promoted team slot
    "promoted_2": "Archetype_Promoted_2",  # Middle promoted team slot
    "promoted_3": "Archetype_Promoted_3",  # Weakest promoted team slot
}

# Teams that were promoted to EPL each season (most recent 5 seasons)
# Format: {season: [(team, archetype_slot), ...]}
PROMOTED_TEAMS_BY_SEASON = {
    2020: [("Leeds United", "promoted_1"), ("West Bromwich", "promoted_2"), ("Fulham", "promoted_3")],
    2021: [("Brentford", "promoted_1"), ("Watford", "promoted_2"), ("Norwich City", "promoted_3")],
    2022: [("Fulham", "promoted_1"), ("Bournemouth", "promoted_2"), ("Nott'ham Forest", "promoted_3")],
    2023: [("Burnley", "promoted_1"), ("Sheffield United", "promoted_2"), ("Luton Town", "promoted_3")],
    2024: [("Leicester City", "promoted_1"), ("Ipswich Town", "promoted_2"), ("Southampton", "promoted_3")],
    2025: [("Leeds United", "promoted_1"), ("Sunderland", "promoted_2"), ("Burnley", "promoted_3")],
}


class MatchData:
    """Container for a single match result."""
    def __init__(self, home_team: str, away_team: str, outcome: int, season: int):
        self.home_team = home_team
        self.away_team = away_team
        self.outcome = outcome  # 0=away_win, 1=draw, 2=home_win
        self.season = season


async def fetch_season_matches(api: FootballDataAPI, season: int) -> List[MatchData]:
    """Fetch all finished matches for a given season."""
    logger.info(f"Fetching matches for {season}-{season+1} season...")

    try:
        match_results = await api.get_finished_matches(season=season)

        matches = []
        for m in match_results:
            if m.home_goals is not None and m.away_goals is not None:
                if m.home_goals > m.away_goals:
                    outcome = 2
                elif m.home_goals < m.away_goals:
                    outcome = 0
                else:
                    outcome = 1

                matches.append(MatchData(
                    home_team=m.home_team,
                    away_team=m.away_team,
                    outcome=outcome,
                    season=season,
                ))

        logger.info(f"  Found {len(matches)} finished matches")
        return matches

    except Exception as e:
        logger.error(f"Failed to fetch {season} season: {e}")
        return []


def get_archetype_for_team(team: str, season: int) -> Optional[str]:
    """
    Check if a team should be mapped to a promoted team archetype for a given season.

    Returns the archetype slot name if the team was promoted that season, None otherwise.
    """
    promoted = PROMOTED_TEAMS_BY_SEASON.get(season, [])
    for promoted_team, slot in promoted:
        if promoted_team == team:
            return ARCHETYPE_SLOTS[slot]
    return None


def map_to_archetypes(matches: List[MatchData], current_season: int, n_seasons_available: int) -> Tuple[List[MatchData], Dict[str, str]]:
    """
    Map promoted teams to archetype slots for prior estimation.

    Teams that were promoted within our data window get their
    matches mapped to archetype slots. This captures the pattern that
    promoted teams behave similarly regardless of which specific club.

    Args:
        matches: All matches from historical seasons
        current_season: The current season we're building priors for
        n_seasons_available: Number of seasons we actually have data for

    Returns:
        (mapped_matches, team_mapping) where team_mapping shows original->archetype
    """
    # Count how many seasons each team appears in
    team_seasons = defaultdict(set)
    for m in matches:
        team_seasons[m.home_team].add(m.season)
        team_seasons[m.away_team].add(m.season)

    # Identify teams that need archetype mapping
    # Only map teams that were PROMOTED within our data window (not established teams)
    archetype_mapping = {}

    # Get list of promoted teams for seasons in our data
    seasons_in_data = set()
    for m in matches:
        seasons_in_data.add(m.season)

    for season in seasons_in_data:
        promoted = PROMOTED_TEAMS_BY_SEASON.get(season, [])
        for team, slot in promoted:
            if team in team_seasons:  # Only if team appears in our data
                archetype_mapping[team] = ARCHETYPE_SLOTS[slot]
                logger.info(f"  Mapping promoted team {team} (season {season}) -> {ARCHETYPE_SLOTS[slot]}")

    logger.info(f"Archetype mappings: {len(archetype_mapping)} promoted teams mapped to archetypes")

    # Apply mappings to matches
    mapped_matches = []
    for m in matches:
        home = archetype_mapping.get(m.home_team, m.home_team)
        away = archetype_mapping.get(m.away_team, m.away_team)
        mapped_matches.append(MatchData(
            home_team=home,
            away_team=away,
            outcome=m.outcome,
            season=m.season,
        ))

    return mapped_matches, archetype_mapping


def prepare_match_data(matches: List[MatchData]) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Convert MatchData to format expected by IRT model.

    Returns:
        (match_dicts, team_to_idx)
    """
    teams = set()
    for m in matches:
        teams.add(m.home_team)
        teams.add(m.away_team)

    team_list = sorted(list(teams))
    team_to_idx = {t: i for i, t in enumerate(team_list)}

    match_dicts = [
        {"home_team": m.home_team, "away_team": m.away_team, "outcome": m.outcome}
        for m in matches
    ]

    return match_dicts, team_to_idx


async def main():
    parser = argparse.ArgumentParser(description="Estimate 5-year MOTSON priors")
    parser.add_argument("--seasons", type=int, default=5, help="Number of seasons (default: 5)")
    parser.add_argument("--output", type=str, default="data/five_year_priors.json",
                        help="Output file path")
    parser.add_argument("--n-warmup", type=int, default=500, help="MCMC warmup samples")
    parser.add_argument("--n-samples", type=int, default=1000, help="MCMC posterior samples")
    args = parser.parse_args()

    # Check for API key
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        logger.error("FOOTBALL_DATA_API_KEY not set!")
        sys.exit(1)

    api = FootballDataAPI(api_key)

    # Determine seasons to fetch
    current_season = app_config.CURRENT_SEASON
    seasons_to_fetch = list(range(current_season - args.seasons, current_season))
    logger.info(f"Fetching seasons: {seasons_to_fetch}")

    # Fetch all matches
    all_matches = []
    for season in seasons_to_fetch:
        matches = await fetch_season_matches(api, season)
        all_matches.extend(matches)

    if not all_matches:
        logger.error("No matches fetched!")
        sys.exit(1)

    logger.info(f"Total matches: {len(all_matches)}")

    # Count actual seasons fetched
    seasons_fetched = len(set(m.season for m in all_matches))
    logger.info(f"Successfully fetched {seasons_fetched} seasons of data")

    # Map promoted teams to archetypes
    mapped_matches, archetype_mapping = map_to_archetypes(
        all_matches, current_season, seasons_fetched
    )

    # Prepare data for IRT model
    match_dicts, team_to_idx = prepare_match_data(mapped_matches)
    logger.info(f"Unique teams/archetypes: {len(team_to_idx)}")

    # Fit IRT model
    logger.info("Fitting Bayesian IRT model (this may take a few minutes)...")
    params = fit_irt_model(
        match_dicts,
        team_to_idx,
        theta_prior_mean=None,  # Uninformative priors for 5-year estimation
        theta_prior_std=0.75,   # Wider prior for historical estimation
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
    )

    # Build output structure
    output = {
        "estimated_at": datetime.utcnow().isoformat(),
        "seasons_used": seasons_to_fetch,
        "n_matches": len(all_matches),
        "teams": {},
        "archetype_mapping": archetype_mapping,
    }

    # Sort by theta (strongest first)
    sorted_teams = sorted(params.theta.keys(), key=lambda t: -params.theta[t])

    logger.info("\n" + "=" * 60)
    logger.info("5-YEAR PRIOR ESTIMATES")
    logger.info("=" * 60)
    logger.info(f"{'Team':25} {'Theta':>8} {'SE':>6} {'b_home':>8} {'b_away':>8}")
    logger.info("-" * 60)

    for team in sorted_teams:
        theta = params.theta[team]
        theta_se = params.theta_se[team]
        b_home = params.b_home[team]
        b_away = params.b_away[team]
        b_home_se = params.b_home_se[team]
        b_away_se = params.b_away_se[team]

        output["teams"][team] = {
            "theta": theta,
            "theta_se": theta_se,
            "b_home": b_home,
            "b_home_se": b_home_se,
            "b_away": b_away,
            "b_away_se": b_away_se,
        }

        logger.info(f"{team:25} {theta:+8.3f} {theta_se:6.3f} {b_home:+8.3f} {b_away:+8.3f}")

    # Save output
    output_path = project_root / args.output
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved to {output_path}")

    # Print archetype values (for promoted team assignment)
    logger.info("\nArchetype values (for promoted teams):")
    for slot, archetype in ARCHETYPE_SLOTS.items():
        if archetype in params.theta:
            logger.info(f"  {slot}: theta={params.theta[archetype]:+.3f}")


if __name__ == "__main__":
    asyncio.run(main())
