#!/usr/bin/env python3
"""
MOTSON Lead Survival Curve Builder

Builds empirical lookup tables for:
    P(title | lead_points, week, theta_advantage)

This captures the "bottling coefficient" - how often do leads actually hold?
Historical EPL data shows that large leads early in the season are less
secure than the raw Monte Carlo simulations suggest.

Usage:
    python scripts/build_survival_curves.py [--seasons 10] [--output data/lead_survival.json]

Requires FOOTBALL_DATA_API_KEY environment variable.
"""

import os
import sys
import asyncio
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.data_fetcher import FootballDataAPI
from backend.config import app_config, TEAM_NAME_MAP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class WeeklyStanding:
    """A team's standing at a specific week."""
    team: str
    week: int
    season: int
    points: int
    played: int
    goal_diff: int
    position: int


@dataclass
class LeadSituation:
    """A snapshot of the title race at a given week."""
    season: int
    week: int
    leader: str
    leader_points: int
    second_place: str
    second_points: int
    lead_margin: int
    games_remaining: int  # For leader
    eventual_champion: str
    leader_won_title: bool


def normalize_team_name(name: str) -> str:
    """Normalize team name using standard mapping."""
    # Check direct mapping
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]

    # Try common variations
    name_lower = name.lower()

    # Handle FC suffix/prefix
    clean_name = name.replace(" FC", "").replace("FC ", "").strip()
    if clean_name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[clean_name]

    # Common abbreviations
    abbrev_map = {
        "man utd": "Manchester Utd",
        "man united": "Manchester Utd",
        "manchester united": "Manchester Utd",
        "man city": "Manchester City",
        "manchester city": "Manchester City",
        "newcastle": "Newcastle Utd",
        "newcastle united": "Newcastle Utd",
        "nottingham forest": "Nott'ham Forest",
        "nott'm forest": "Nott'ham Forest",
        "wolves": "Wolves",
        "wolverhampton": "Wolves",
        "wolverhampton wanderers": "Wolves",
        "west ham": "West Ham",
        "west ham united": "West Ham",
        "tottenham": "Tottenham",
        "tottenham hotspur": "Tottenham",
        "spurs": "Tottenham",
        "brighton": "Brighton",
        "brighton & hove albion": "Brighton",
        "brighton and hove albion": "Brighton",
        "bournemouth": "Bournemouth",
        "afc bournemouth": "Bournemouth",
    }

    if name_lower in abbrev_map:
        return abbrev_map[name_lower]

    return name  # Return original if no mapping found


async def fetch_season_standings(api: FootballDataAPI, season: int) -> List[Dict]:
    """
    Fetch standings for a complete season.

    The football-data.org API provides final standings, but we need to
    reconstruct week-by-week standings from match results.
    """
    logger.info(f"Fetching standings for {season}-{season+1} season...")

    try:
        # Get all matches for the season
        matches = await api.get_finished_matches(season=season)

        if not matches:
            logger.warning(f"No matches found for season {season}")
            return []

        # Build week-by-week standings
        # Group matches by matchday/week
        matches_by_week = defaultdict(list)
        for m in matches:
            if m.home_goals is not None and m.away_goals is not None:
                # Use matchday if available, otherwise estimate from date
                week = getattr(m, 'matchday', None)
                if week is None:
                    # Fallback: estimate week from match date
                    # This is imperfect but better than nothing
                    week = 1  # Default to week 1 if unknown
                matches_by_week[week].append(m)

        logger.info(f"  Found {len(matches)} matches across {len(matches_by_week)} weeks")
        return build_weekly_standings(matches_by_week, season)

    except Exception as e:
        logger.error(f"Failed to fetch season {season}: {e}")
        return []


def build_weekly_standings(matches_by_week: Dict[int, List], season: int) -> List[WeeklyStanding]:
    """
    Build cumulative standings after each matchweek.
    """
    # Track cumulative stats for each team
    team_stats = defaultdict(lambda: {"points": 0, "played": 0, "gf": 0, "ga": 0})

    all_standings = []

    # Process weeks in order
    for week in sorted(matches_by_week.keys()):
        week_matches = matches_by_week[week]

        # Update stats from this week's matches
        for m in week_matches:
            home = normalize_team_name(m.home_team)
            away = normalize_team_name(m.away_team)

            team_stats[home]["played"] += 1
            team_stats[away]["played"] += 1
            team_stats[home]["gf"] += m.home_goals
            team_stats[home]["ga"] += m.away_goals
            team_stats[away]["gf"] += m.away_goals
            team_stats[away]["ga"] += m.home_goals

            if m.home_goals > m.away_goals:
                team_stats[home]["points"] += 3
            elif m.home_goals < m.away_goals:
                team_stats[away]["points"] += 3
            else:
                team_stats[home]["points"] += 1
                team_stats[away]["points"] += 1

        # Calculate standings after this week
        standings_list = []
        for team, stats in team_stats.items():
            gd = stats["gf"] - stats["ga"]
            standings_list.append({
                "team": team,
                "points": stats["points"],
                "played": stats["played"],
                "gd": gd,
                "gf": stats["gf"]  # For tiebreaker
            })

        # Sort by points, then GD, then GF
        standings_list.sort(key=lambda x: (-x["points"], -x["gd"], -x["gf"]))

        # Create WeeklyStanding objects
        for pos, team_data in enumerate(standings_list, 1):
            all_standings.append(WeeklyStanding(
                team=team_data["team"],
                week=week,
                season=season,
                points=team_data["points"],
                played=team_data["played"],
                goal_diff=team_data["gd"],
                position=pos
            ))

    return all_standings


def extract_lead_situations(standings: List[WeeklyStanding], season: int) -> List[LeadSituation]:
    """
    Extract title race snapshots from weekly standings.
    """
    # Group standings by week
    by_week = defaultdict(list)
    for s in standings:
        by_week[s.week].append(s)

    if not by_week:
        return []

    # Find eventual champion (position 1 in final week)
    final_week = max(by_week.keys())
    final_standings = by_week[final_week]
    champion = next((s.team for s in final_standings if s.position == 1), None)

    if not champion:
        logger.warning(f"Could not determine champion for season {season}")
        return []

    logger.info(f"  Season {season}-{season+1} champion: {champion}")

    situations = []

    for week in sorted(by_week.keys()):
        week_standings = sorted(by_week[week], key=lambda x: x.position)

        if len(week_standings) < 2:
            continue

        leader = week_standings[0]
        second = week_standings[1]

        # Calculate games remaining (38 - played)
        games_remaining = 38 - leader.played

        situations.append(LeadSituation(
            season=season,
            week=week,
            leader=leader.team,
            leader_points=leader.points,
            second_place=second.team,
            second_points=second.points,
            lead_margin=leader.points - second.points,
            games_remaining=games_remaining,
            eventual_champion=champion,
            leader_won_title=(leader.team == champion)
        ))

    return situations


def build_survival_table(situations: List[LeadSituation]) -> Dict:
    """
    Build lookup table: P(title | lead_margin, games_remaining)

    Returns dict with structure:
    {
        "by_lead_and_games": {
            "7_16": {"wins": 45, "total": 50, "rate": 0.90},
            ...
        },
        "by_lead": {
            "7": {"wins": 120, "total": 140, "rate": 0.857},
            ...
        },
        "by_games_remaining": {
            "16": {"wins": 200, "total": 250, "rate": 0.80},
            ...
        }
    }
    """
    # Track outcomes
    by_lead_and_games = defaultdict(lambda: {"wins": 0, "total": 0})
    by_lead = defaultdict(lambda: {"wins": 0, "total": 0})
    by_games_remaining = defaultdict(lambda: {"wins": 0, "total": 0})

    for sit in situations:
        # Skip very early weeks (before week 5) - too noisy
        if sit.week < 5:
            continue

        lead_key = str(sit.lead_margin)
        games_key = str(sit.games_remaining)
        combined_key = f"{sit.lead_margin}_{sit.games_remaining}"

        won = 1 if sit.leader_won_title else 0

        by_lead_and_games[combined_key]["wins"] += won
        by_lead_and_games[combined_key]["total"] += 1

        by_lead[lead_key]["wins"] += won
        by_lead[lead_key]["total"] += 1

        by_games_remaining[games_key]["wins"] += won
        by_games_remaining[games_key]["total"] += 1

    # Calculate rates
    def add_rate(d):
        result = {}
        for k, v in d.items():
            if v["total"] > 0:
                result[k] = {
                    "wins": v["wins"],
                    "total": v["total"],
                    "rate": v["wins"] / v["total"]
                }
        return result

    return {
        "by_lead_and_games": add_rate(by_lead_and_games),
        "by_lead": add_rate(by_lead),
        "by_games_remaining": add_rate(by_games_remaining)
    }


def build_smoothed_lookup(survival_table: Dict) -> Dict:
    """
    Build a smoothed lookup that handles sparse data.

    For combinations we haven't seen, interpolate from nearby values.
    """
    by_lead_and_games = survival_table["by_lead_and_games"]
    by_lead = survival_table["by_lead"]
    by_games = survival_table["by_games_remaining"]

    # Create smoothed lookup for lead 0-15 and games 1-33
    smoothed = {}

    for lead in range(-5, 20):  # -5 to +19 point leads
        for games in range(1, 34):  # 1 to 33 games remaining
            key = f"{lead}_{games}"

            # Check if we have direct observation
            if key in by_lead_and_games and by_lead_and_games[key]["total"] >= 3:
                smoothed[key] = by_lead_and_games[key]["rate"]
                continue

            # Otherwise, estimate from marginals with Bayesian-ish blend
            lead_rate = by_lead.get(str(lead), {}).get("rate")
            games_rate = by_games.get(str(games), {}).get("rate")

            if lead_rate is not None and games_rate is not None:
                # Weighted geometric mean (gives more weight to lead margin)
                smoothed[key] = (lead_rate ** 0.6) * (games_rate ** 0.4)
            elif lead_rate is not None:
                smoothed[key] = lead_rate
            elif games_rate is not None:
                smoothed[key] = games_rate
            else:
                # Fallback: use logistic estimate based on expected points
                # At games_remaining=G, leader needs to drop (lead/3)*G points
                # Very rough heuristic
                if lead > 0:
                    # Probability increases with lead, decreases with games
                    points_needed_per_game = lead / (games * 3) if games > 0 else 0
                    smoothed[key] = min(0.99, 0.5 + points_needed_per_game * 2)
                elif lead < 0:
                    smoothed[key] = max(0.01, 0.5 + lead * 0.1)
                else:
                    smoothed[key] = 0.5

    return smoothed


async def main():
    parser = argparse.ArgumentParser(description="Build lead survival lookup tables")
    parser.add_argument("--seasons", type=int, default=10,
                        help="Number of seasons to analyze (default: 10)")
    parser.add_argument("--output", type=str, default="data/lead_survival.json",
                        help="Output file path")
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
    seasons_to_fetch = list(range(current_season - args.seasons, current_season))
    logger.info(f"Analyzing seasons: {seasons_to_fetch}")

    all_situations = []

    for season in seasons_to_fetch:
        # Fetch and process each season
        standings = await fetch_season_standings(api, season)
        if standings:
            situations = extract_lead_situations(standings, season)
            all_situations.extend(situations)
            logger.info(f"  Extracted {len(situations)} lead situations")

        # Rate limiting - be nice to the API
        await asyncio.sleep(1)

    if not all_situations:
        logger.error("No lead situations extracted! Check API responses.")
        sys.exit(1)

    logger.info(f"\nTotal lead situations analyzed: {len(all_situations)}")

    # Build survival table
    survival_table = build_survival_table(all_situations)

    # Build smoothed lookup
    smoothed_lookup = build_smoothed_lookup(survival_table)

    # Print some key findings
    logger.info("\n" + "="*60)
    logger.info("KEY FINDINGS: Lead Survival Rates")
    logger.info("="*60)

    # Show rates for common scenarios
    key_scenarios = [
        ("7 points, 16 games left", "7_16"),
        ("7 points, 10 games left", "7_10"),
        ("4 points, 16 games left", "4_16"),
        ("10 points, 10 games left", "10_10"),
        ("3 points, 5 games left", "3_5"),
    ]

    for desc, key in key_scenarios:
        if key in survival_table["by_lead_and_games"]:
            data = survival_table["by_lead_and_games"][key]
            logger.info(f"  {desc}: {data['rate']:.1%} ({data['wins']}/{data['total']} cases)")
        elif key in smoothed_lookup:
            logger.info(f"  {desc}: ~{smoothed_lookup[key]:.1%} (interpolated)")
        else:
            logger.info(f"  {desc}: No data")

    # Prepare output
    output_data = {
        "generated_at": datetime.utcnow().isoformat(),
        "seasons_analyzed": seasons_to_fetch,
        "total_situations": len(all_situations),
        "raw_survival_table": survival_table,
        "smoothed_lookup": smoothed_lookup,
        "usage_notes": {
            "key_format": "lead_gamesremaining (e.g., '7_16' = 7 point lead with 16 games left)",
            "rate": "Empirical probability that leader wins title",
            "smoothed": "Interpolated values for sparse cells"
        }
    }

    # Save to file
    output_path = project_root / args.output
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nSaved to {output_path}")

    # Also save a simple CSV for easy inspection
    csv_path = output_path.with_suffix('.csv')
    rows = []
    for key, rate in sorted(smoothed_lookup.items()):
        lead, games = key.split('_')
        rows.append({
            "lead_margin": int(lead),
            "games_remaining": int(games),
            "survival_rate": rate
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.info(f"Also saved CSV to {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
