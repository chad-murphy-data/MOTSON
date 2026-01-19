#!/usr/bin/env python3
"""
Fetch Historical EPL Data from FBref

FBref provides free access to historical Premier League match data.
This script fetches match results for seasons beyond what the football-data.org
free tier allows.

Usage:
    python scripts/fetch_fbref_history.py --seasons 5
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import time

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FBref Premier League season URLs
# Format: https://fbref.com/en/comps/9/YYYY-YYYY/schedule/YYYY-YYYY-Premier-League-Scores-and-Fixtures
FBREF_BASE = "https://fbref.com/en/comps/9"


def get_season_url(season_start: int) -> str:
    """Get FBref URL for a season (e.g., 2023 for 2023-24 season)."""
    season_end = season_start + 1
    return f"{FBREF_BASE}/{season_start}-{season_end}/schedule/{season_start}-{season_end}-Premier-League-Scores-and-Fixtures"


def normalize_team_name(name: str) -> str:
    """Normalize team names to match our standard format."""
    # FBref uses full names, we need to map to our format
    name_map = {
        "Manchester Utd": "Manchester Utd",
        "Manchester United": "Manchester Utd",
        "Manchester City": "Manchester City",
        "Nott'ham Forest": "Nott'ham Forest",
        "Nottingham Forest": "Nott'ham Forest",
        "Newcastle Utd": "Newcastle Utd",
        "Newcastle United": "Newcastle Utd",
        "Tottenham": "Tottenham",
        "Tottenham Hotspur": "Tottenham",
        "Wolverhampton Wanderers": "Wolves",
        "Wolves": "Wolves",
        "Brighton and Hove Albion": "Brighton",
        "Brighton": "Brighton",
        "Leicester City": "Leicester City",
        "West Ham United": "West Ham",
        "West Ham": "West Ham",
        "Crystal Palace": "Crystal Palace",
        "Sheffield United": "Sheffield United",
        "Sheffield Utd": "Sheffield United",
        "AFC Bournemouth": "Bournemouth",
        "Bournemouth": "Bournemouth",
        "Luton Town": "Luton Town",
        "Ipswich Town": "Ipswich Town",
        "Leeds United": "Leeds United",
        "West Bromwich Albion": "West Brom",
        "West Brom": "West Brom",
        "Watford": "Watford",
        "Norwich City": "Norwich City",
        "Brentford": "Brentford",
        # Add more as needed
    }
    return name_map.get(name, name)


def fetch_season_matches(season_start: int) -> List[Dict]:
    """
    Fetch all matches for a season from FBref.

    Returns list of match dicts with:
    - home_team, away_team, home_goals, away_goals, date, matchweek
    """
    url = get_season_url(season_start)
    logger.info(f"Fetching season {season_start}-{season_start+1} from FBref...")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the scores & fixtures table
    table = soup.find('table', {'id': 'sched_all'})
    if not table:
        # Try alternative table ID
        table = soup.find('table', class_='stats_table')

    if not table:
        logger.error(f"Could not find match table for season {season_start}")
        return []

    matches = []
    rows = table.find('tbody').find_all('tr')

    for row in rows:
        # Skip spacer rows
        if 'spacer' in row.get('class', []):
            continue
        if not row.find('td'):
            continue

        cols = row.find_all('td')
        if len(cols) < 6:
            continue

        try:
            # Extract data - FBref table structure
            # Columns: Wk, Day, Date, Time, Home, Score, Away, Attendance, Venue, Referee, Match Report, Notes

            wk_cell = row.find('th', {'data-stat': 'gameweek'})
            matchweek = int(wk_cell.text) if wk_cell and wk_cell.text.isdigit() else 0

            home_cell = row.find('td', {'data-stat': 'home_team'})
            away_cell = row.find('td', {'data-stat': 'away_team'})
            score_cell = row.find('td', {'data-stat': 'score'})

            if not home_cell or not away_cell or not score_cell:
                continue

            home_team = normalize_team_name(home_cell.text.strip())
            away_team = normalize_team_name(away_cell.text.strip())

            score_text = score_cell.text.strip()
            if '–' not in score_text and '-' not in score_text:
                continue  # Match not played yet

            # Parse score (format: "2–1" or "2-1")
            score_parts = score_text.replace('–', '-').split('-')
            if len(score_parts) != 2:
                continue

            home_goals = int(score_parts[0].strip())
            away_goals = int(score_parts[1].strip())

            # Get date
            date_cell = row.find('td', {'data-stat': 'date'})
            match_date = date_cell.text.strip() if date_cell else ""

            matches.append({
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "matchweek": matchweek,
                "date": match_date,
                "season": season_start,
            })

        except (ValueError, AttributeError) as e:
            continue

    logger.info(f"  Found {len(matches)} matches")
    return matches


def main():
    parser = argparse.ArgumentParser(description="Fetch historical EPL data from FBref")
    parser.add_argument("--seasons", type=int, default=5, help="Number of seasons to fetch")
    parser.add_argument("--output", type=str, default="data/historical_matches.json",
                        help="Output file path")
    args = parser.parse_args()

    # Current season is 2025-26, so we want 2020-21 through 2024-25 for 5 years
    current_season = 2025
    seasons_to_fetch = list(range(current_season - args.seasons, current_season))

    logger.info(f"Fetching seasons: {seasons_to_fetch}")

    all_matches = []
    for season in seasons_to_fetch:
        matches = fetch_season_matches(season)
        all_matches.extend(matches)

        # Be polite to FBref
        if season != seasons_to_fetch[-1]:
            logger.info("Waiting 5 seconds before next request...")
            time.sleep(5)

    if not all_matches:
        logger.error("No matches fetched!")
        return

    # Summary by season
    logger.info("\nSummary:")
    for season in seasons_to_fetch:
        count = len([m for m in all_matches if m["season"] == season])
        logger.info(f"  {season}-{season+1}: {count} matches")

    logger.info(f"\nTotal matches: {len(all_matches)}")

    # Save to file
    output_path = project_root / args.output
    output_path.parent.mkdir(exist_ok=True)

    output = {
        "fetched_at": datetime.utcnow().isoformat(),
        "seasons": seasons_to_fetch,
        "n_matches": len(all_matches),
        "matches": all_matches,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
