#!/usr/bin/env python3
"""
Backfill IRT History Script

Replays the season week-by-week, running fresh IRT estimation at each week
to build the historical trajectory of team parameters.

Also runs season simulations at each week to generate:
- Predicted final points
- Predicted table position
- Title/Top4/Relegation probabilities

This creates the week-by-week data needed for the dashboard visualizations.
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.database.db import get_db
from backend.models.bayesian_blender import BayesianBlender
from backend.models.irt_state import IRTTeamState
from backend.models.season_simulator import simulate_season
from backend.config import EPL_TEAMS_2025_26

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Teams promoted in 2025-26 season
PROMOTED_TEAMS_2025 = ["Leeds United", "Sunderland", "Burnley"]


def get_all_results_as_dicts(db) -> List[Dict]:
    """Get all match results as dicts for simulator."""
    results = db.get_match_results()
    return [
        {
            "home_team": r.home_team,
            "away_team": r.away_team,
            "home_goals": r.home_goals,
            "away_goals": r.away_goals,
            "matchweek": r.matchweek,
        }
        for r in results
    ]


def get_all_fixtures_as_dicts(db) -> List[Dict]:
    """Get all fixtures as dicts for simulator."""
    fixtures = db.get_fixtures()
    result = []
    for f in fixtures:
        # Handle both Fixture objects and dicts
        if hasattr(f, 'home_team'):
            result.append({
                "home_team": f.home_team,
                "away_team": f.away_team,
                "matchweek": f.matchweek,
            })
        else:
            result.append({
                "home_team": f["home_team"],
                "away_team": f["away_team"],
                "matchweek": f["matchweek"],
            })
    return result


def get_matches_through_week(db, max_week: int) -> List[Dict]:
    """Get all match results through a given week."""
    results = db.get_match_results()

    matches = []
    for r in results:
        if r.matchweek <= max_week:
            # Convert to IRT format
            if r.home_goals > r.away_goals:
                outcome = 2  # Home win
            elif r.home_goals < r.away_goals:
                outcome = 0  # Away win
            else:
                outcome = 1  # Draw

            matches.append({
                "home_team": r.home_team,
                "away_team": r.away_team,
                "outcome": outcome,
                "matchweek": r.matchweek,
            })

    return matches


def run_simulation_for_week(
    states: Dict[str, IRTTeamState],
    all_fixtures: List[Dict],
    all_results: List[Dict],
    week: int,
    n_simulations: int = 500
) -> Dict[str, Dict]:
    """Run season simulation from a specific week.

    Returns dict mapping team -> simulation result dict
    """
    from collections import defaultdict

    # Calculate current points through this week
    current_points = defaultdict(int)
    for r in all_results:
        if r["matchweek"] <= week:
            if r["home_goals"] > r["away_goals"]:
                current_points[r["home_team"]] += 3
            elif r["home_goals"] < r["away_goals"]:
                current_points[r["away_team"]] += 3
            else:
                current_points[r["home_team"]] += 1
                current_points[r["away_team"]] += 1

    # Get remaining fixtures
    remaining = [f for f in all_fixtures if f["matchweek"] > week]

    if not remaining:
        # Season complete - just return current standings
        results = {}
        sorted_teams = sorted(states.keys(), key=lambda t: -current_points[t])
        for pos, team in enumerate(sorted_teams, 1):
            pts = current_points[team]
            results[team] = {
                "predicted_final_points": pts,
                "predicted_points_std": 0,
                "predicted_position": pos,
                "position_5th": pos,
                "position_95th": pos,
                "p_title": 1.0 if pos == 1 else 0.0,
                "p_top4": 1.0 if pos <= 4 else 0.0,
                "p_top6": 1.0 if pos <= 6 else 0.0,
                "p_relegation": 1.0 if pos >= 18 else 0.0,
            }
        return results

    # Run simulation
    sim_results = simulate_season(
        team_states=states,
        remaining_fixtures=remaining,
        current_points=dict(current_points),
        n_simulations=n_simulations,
        seed=42 + week,  # Reproducible but different per week
    )

    # Convert to dicts
    return {
        team: {
            "predicted_final_points": r.predicted_final_points,
            "predicted_points_std": r.predicted_points_std,
            "predicted_position": r.predicted_position,
            "position_5th": r.position_5th_percentile,
            "position_95th": r.position_95th_percentile,
            "p_title": r.p_title,
            "p_top4": r.p_top4,
            "p_top6": r.p_top6,
            "p_relegation": r.p_relegation,
        }
        for team, r in sim_results.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Backfill IRT history with simulations")
    parser.add_argument("--n-sims", type=int, default=500, help="Simulations per week")
    parser.add_argument("--skip-irt", action="store_true", help="Skip IRT fitting, only run simulations")
    args = parser.parse_args()

    db = get_db()

    # Initialize the blender with 5-year priors
    blender = BayesianBlender()
    blender.load_priors()

    # Get all teams
    teams = EPL_TEAMS_2025_26

    # Get all fixtures and results for simulation
    all_fixtures = get_all_fixtures_as_dicts(db)
    all_results = get_all_results_as_dicts(db)
    logger.info(f"Loaded {len(all_fixtures)} fixtures and {len(all_results)} results")

    # Initialize states at week 0 (pre-season)
    logger.info("Initializing team states from 5-year priors...")
    states = blender.initialize_season_states(teams, promoted_teams=PROMOTED_TEAMS_2025)

    # Run simulation for week 0
    logger.info("Running pre-season simulation...")
    sim_results = run_simulation_for_week(states, all_fixtures, all_results, 0, args.n_sims)

    # Save week 0 (pre-season) state
    for team, state in states.items():
        db.save_irt_team_state_history(state, week=0, simulation_result=sim_results.get(team))
    logger.info("Saved week 0 (pre-season) states with simulations")

    # Get current week from database
    current_week = db.get_current_week()
    logger.info(f"Current week in database: {current_week}")

    # Process each week
    for week in range(1, current_week + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing week {week}")
        logger.info(f"{'='*50}")

        # Get all matches through this week
        season_matches = get_matches_through_week(db, week)
        logger.info(f"Total matches through week {week}: {len(season_matches)}")

        if not season_matches:
            logger.warning(f"No matches for week {week}, skipping")
            continue

        # Run IRT update (unless skipped)
        if not args.skip_irt:
            try:
                states = blender.update_week(
                    week=week,
                    season_matches=season_matches,
                    current_states=states,
                    n_warmup=300,  # Faster for backfill
                    n_samples=500,
                )
            except Exception as e:
                logger.error(f"Failed to update week {week}: {e}")
                continue

        # Run simulation for this week
        logger.info(f"Running simulation from week {week}...")
        sim_results = run_simulation_for_week(states, all_fixtures, all_results, week, args.n_sims)

        # Save state history for this week
        for team, state in states.items():
            db.save_irt_team_state_history(state, week=week, simulation_result=sim_results.get(team))

        # Print top 5 teams by predicted position
        sorted_teams = sorted(sim_results.items(), key=lambda x: x[1]["predicted_position"])[:5]
        logger.info(f"Top 5 by predicted position (week {week}):")
        for team, sim in sorted_teams:
            state = states[team]
            logger.info(
                f"  {team:20} pos={sim['predicted_position']:.1f} "
                f"pts={sim['predicted_final_points']:.1f} "
                f"p_title={sim['p_title']*100:.1f}%"
            )

    # Save final states
    logger.info("\nSaving final IRT team states...")
    db.save_irt_team_states(states)

    logger.info("\n" + "=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Processed weeks 0-{current_week}")
    logger.info(f"Teams: {len(states)}")

    # Print final standings by predicted position
    logger.info("\nFinal predicted standings:")
    sorted_teams = sorted(sim_results.items(), key=lambda x: x[1]["predicted_position"])
    for i, (team, sim) in enumerate(sorted_teams, 1):
        state = states[team]
        logger.info(
            f"{i:2}. {team:20} pred_pts={sim['predicted_final_points']:.1f} "
            f"p_title={sim['p_title']*100:5.1f}% "
            f"p_top4={sim['p_top4']*100:5.1f}% "
            f"p_rel={sim['p_relegation']*100:5.1f}%"
        )


if __name__ == "__main__":
    main()
