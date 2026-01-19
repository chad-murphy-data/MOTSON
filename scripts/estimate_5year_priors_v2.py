#!/usr/bin/env python3
"""
MOTSON 5-Year Prior Estimation Script (v2)

Estimates the gravity priors (theta, b_home, b_away) from 5 seasons of historical data
using the Bayesian IRT model.

Uses football-data.co.uk as the data source (free, no API key required).

Usage:
    python scripts/estimate_5year_priors_v2.py [--seasons 5]
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from collections import defaultdict
from io import StringIO
import json

import numpy as np
import pandas as pd
import requests

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.models.irt_model import fit_irt_model, IRTParameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Team name normalization map (football-data.co.uk names -> our standard names)
TEAM_NAME_MAP = {
    "Man City": "Manchester City",
    "Man United": "Manchester Utd",
    "Nott'm Forest": "Nott'ham Forest",
    "Nottingham Forest": "Nott'ham Forest",
    "Newcastle": "Newcastle Utd",
    "Spurs": "Tottenham",
    "Sheffield Utd": "Sheffield United",
    "Wolves": "Wolves",
    "West Brom": "West Brom",
    "Luton": "Luton Town",
    "Leicester": "Leicester City",
    "Ipswich": "Ipswich Town",
    "Leeds": "Leeds United",
}


# Season code mapping
SEASON_CODES = {
    2020: "2021",  # 2020-21 season
    2021: "2122",  # 2021-22 season
    2022: "2223",  # 2022-23 season
    2023: "2324",  # 2023-24 season
    2024: "2425",  # 2024-25 season (current, may be incomplete)
}


# Promoted team archetypes
ARCHETYPE_SLOTS = {
    "promoted_1": "Archetype_Promoted_1",
    "promoted_2": "Archetype_Promoted_2",
    "promoted_3": "Archetype_Promoted_3",
}

# Teams promoted each season
PROMOTED_TEAMS_BY_SEASON = {
    2020: [("Leeds", "promoted_1"), ("West Brom", "promoted_2"), ("Fulham", "promoted_3")],
    2021: [("Brentford", "promoted_1"), ("Watford", "promoted_2"), ("Norwich", "promoted_3")],
    2022: [("Fulham", "promoted_1"), ("Bournemouth", "promoted_2"), ("Nott'm Forest", "promoted_3")],
    2023: [("Burnley", "promoted_1"), ("Sheffield Utd", "promoted_2"), ("Luton", "promoted_3")],
    2024: [("Leicester", "promoted_1"), ("Ipswich", "promoted_2"), ("Southampton", "promoted_3")],
}


def normalize_team_name(name: str) -> str:
    """Normalize team name to our standard format."""
    return TEAM_NAME_MAP.get(name, name)


def fetch_season_data(season: int) -> pd.DataFrame:
    """
    Fetch season data from football-data.co.uk.

    Args:
        season: Season start year (e.g., 2023 for 2023-24)

    Returns:
        DataFrame with match results
    """
    # Map season year to code
    if season in SEASON_CODES:
        code = SEASON_CODES[season]
    else:
        # Generate code (last 2 digits of start and end year)
        code = f"{str(season)[-2:]}{str(season+1)[-2:]}"

    url = f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
    logger.info(f"Fetching season {season}-{season+1} from {url}...")

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        logger.info(f"  Found {len(df)} matches")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch season {season}: {e}")
        return pd.DataFrame()


def convert_to_matches(df: pd.DataFrame, season: int) -> List[Dict]:
    """Convert football-data.co.uk DataFrame to match format."""
    matches = []

    for _, row in df.iterrows():
        try:
            home_team = normalize_team_name(row['HomeTeam'])
            away_team = normalize_team_name(row['AwayTeam'])
            home_goals = int(row['FTHG'])
            away_goals = int(row['FTAG'])

            # Determine outcome
            if home_goals > away_goals:
                outcome = 2  # Home win
            elif home_goals < away_goals:
                outcome = 0  # Away win
            else:
                outcome = 1  # Draw

            matches.append({
                "home_team": home_team,
                "away_team": away_team,
                "outcome": outcome,
                "season": season,
            })
        except (ValueError, KeyError) as e:
            continue

    return matches


def map_promoted_to_archetypes(matches: List[Dict]) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Map promoted teams to archetype slots.

    Only maps teams that were promoted within our data window.
    """
    # Find which seasons are in the data
    seasons_in_data = set(m["season"] for m in matches)

    archetype_mapping = {}

    for season in seasons_in_data:
        promoted = PROMOTED_TEAMS_BY_SEASON.get(season, [])
        for team_name, slot in promoted:
            normalized = normalize_team_name(team_name)
            archetype_mapping[normalized] = ARCHETYPE_SLOTS[slot]

    logger.info(f"Archetype mappings: {len(archetype_mapping)} promoted teams")
    for team, archetype in sorted(archetype_mapping.items()):
        logger.info(f"  {team} -> {archetype}")

    # Apply mappings
    mapped_matches = []
    for m in matches:
        home = archetype_mapping.get(m["home_team"], m["home_team"])
        away = archetype_mapping.get(m["away_team"], m["away_team"])
        mapped_matches.append({
            "home_team": home,
            "away_team": away,
            "outcome": m["outcome"],
            "season": m["season"],
        })

    return mapped_matches, archetype_mapping


def main():
    parser = argparse.ArgumentParser(description="Estimate 5-year MOTSON priors")
    parser.add_argument("--seasons", type=int, default=5, help="Number of seasons (default: 5)")
    parser.add_argument("--output", type=str, default="data/five_year_priors.json",
                        help="Output file path")
    parser.add_argument("--n-warmup", type=int, default=500, help="MCMC warmup samples")
    parser.add_argument("--n-samples", type=int, default=1000, help="MCMC posterior samples")
    args = parser.parse_args()

    # Determine seasons to fetch (up to but not including current season)
    # Current season is 2025-26, so we fetch 2020-21 through 2024-25
    current_season = 2025
    seasons_to_fetch = list(range(current_season - args.seasons, current_season))
    logger.info(f"Fetching seasons: {seasons_to_fetch}")

    # Fetch all matches
    all_matches = []
    for season in seasons_to_fetch:
        df = fetch_season_data(season)
        if not df.empty:
            matches = convert_to_matches(df, season)
            all_matches.extend(matches)

    if not all_matches:
        logger.error("No matches fetched!")
        sys.exit(1)

    logger.info(f"Total matches: {len(all_matches)}")

    # Summary by season
    for season in seasons_to_fetch:
        count = len([m for m in all_matches if m["season"] == season])
        logger.info(f"  {season}-{season+1}: {count} matches")

    # Map promoted teams to archetypes
    mapped_matches, archetype_mapping = map_promoted_to_archetypes(all_matches)

    # Get unique teams/archetypes
    teams = set()
    for m in mapped_matches:
        teams.add(m["home_team"])
        teams.add(m["away_team"])
    teams = sorted(list(teams))
    team_to_idx = {t: i for i, t in enumerate(teams)}

    logger.info(f"Unique teams/archetypes: {len(teams)}")

    # Prepare match data for IRT
    match_dicts = [
        {"home_team": m["home_team"], "away_team": m["away_team"], "outcome": m["outcome"]}
        for m in mapped_matches
    ]

    # Fit IRT model
    logger.info("Fitting Bayesian IRT model...")
    params = fit_irt_model(
        match_dicts,
        team_to_idx,
        theta_prior_mean=None,  # Uninformative priors
        theta_prior_std=0.75,   # Wider prior for historical estimation
        n_warmup=args.n_warmup,
        n_samples=args.n_samples,
    )

    # Build output
    output = {
        "estimated_at": datetime.utcnow().isoformat(),
        "source": "football-data.co.uk",
        "seasons_used": seasons_to_fetch,
        "n_matches": len(all_matches),
        "teams": {},
        "archetype_mapping": archetype_mapping,
    }

    # Sort by theta
    sorted_teams = sorted(params.theta.keys(), key=lambda t: -params.theta[t])

    logger.info("\n" + "=" * 70)
    logger.info("5-YEAR PRIOR ESTIMATES")
    logger.info("=" * 70)
    logger.info(f"{'Team':25} {'Theta':>8} {'SE':>6} {'b_home':>8} {'b_away':>8}")
    logger.info("-" * 70)

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

    # Save
    output_path = project_root / args.output
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
