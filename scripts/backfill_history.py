#!/usr/bin/env python3
"""
MOTSON Historical Backfill Script

Simulates the entire season week-by-week to generate historical trends data.
This replays the season from week 0 (preseason) through the current week,
running Monte Carlo simulations at each point as if we were predicting forward.

Usage:
    python scripts/backfill_history.py [--start-week 0] [--end-week 22]
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.monte_carlo import MonteCarloSimulator
from backend.models.team_state import TeamState, SeasonPrediction, MatchResult, Fixture
from backend.models.bayesian_engine import BayesianEngine, predict_match, initialize_team_states
from backend.database.db import Database
from backend.config import model_config
from backend.services.weekly_update import load_initial_team_states

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = project_root / "data"
DB_PATH = DATA_DIR / "motson.db"


def get_all_match_data_from_db(db: Database):
    """Get all fixtures and results from the database (no API calls needed)."""
    # Get match results (finished matches with scores)
    results = db.get_match_results()
    logger.info(f"Loaded {len(results)} match results from database")

    # Get all fixtures
    fixture_dicts = db.get_fixtures()
    all_fixtures = []
    for f in fixture_dicts:
        fixture = Fixture(
            match_id=f["match_id"],
            matchweek=f["matchweek"],
            date=datetime.fromisoformat(f["date"]) if isinstance(f["date"], str) else f["date"],
            home_team=f["home_team"],
            away_team=f["away_team"],
            status=f.get("status", "SCHEDULED"),
        )
        # Add goal data for finished matches
        if f.get("status") == "FINISHED":
            fixture.home_goals = f.get("home_goals")
            fixture.away_goals = f.get("away_goals")
        all_fixtures.append(fixture)

    logger.info(f"Loaded {len(all_fixtures)} fixtures from database")

    # Determine current matchweek from results
    current_week = max([r.matchweek for r in results]) if results else 0

    return all_fixtures, results, current_week


def simulate_week(
    week: int,
    team_states: Dict[str, TeamState],
    all_fixtures: List[Fixture],
    results_through_week: List[MatchResult],
    standings_at_week: Dict[str, int],
    simulator: MonteCarloSimulator,
    engine: BayesianEngine,
) -> Dict:
    """
    Simulate predictions as if we're at the end of the given week.

    Args:
        week: The week number (0 = preseason, before any matches)
        team_states: Current team states (will be modified)
        all_fixtures: All fixtures for the season
        results_through_week: Results through this week
        standings_at_week: Points standings at this week
        simulator: Monte Carlo simulator
        engine: Bayesian engine for updates

    Returns:
        Dict with season predictions
    """
    # Generate predictions for completed matches (for calibration)
    completed_fixtures = [f for f in all_fixtures if f.matchweek <= week and f.status == "FINISHED"]

    if week > 0:
        # Update team states based on results through this week
        for team_name, team in team_states.items():
            # Get this team's results
            team_results = [
                (res, res.home_team == team_name)
                for res in results_through_week
                if res.home_team == team_name or res.away_team == team_name
            ]

            # Generate predictions for calibration
            team_preds = []
            for f in completed_fixtures:
                if f.home_team == team_name or f.away_team == team_name:
                    try:
                        pred = predict_match(f, team_states)
                        is_home = f.home_team == team_name
                        team_preds.append((pred, is_home))
                    except:
                        pass

            # Run Bayesian update
            updated_team, explanation = engine.weekly_update(
                team=team,
                week=week,
                match_predictions=team_preds,
                match_results=team_results,
            )

            team_states[team_name] = updated_team
            team_states[team_name].last_updated = datetime.utcnow()

    # Remaining fixtures are everything after this week
    remaining_fixtures = [f for f in all_fixtures if f.matchweek > week]

    # Run Monte Carlo simulation
    season_outcomes = simulator.simulate_season(
        team_states=team_states,
        remaining_fixtures=remaining_fixtures,
        current_points=standings_at_week,
        n_simulations=model_config.DEFAULT_SIMULATIONS,
    )

    # Compile predictions
    predictions = []
    for team_name, outcomes in season_outcomes.items():
        predictions.append({
            "team": team_name,
            "position_probs": outcomes["position_probs"],
            "title_prob": outcomes["title_prob"],
            "top4_prob": outcomes["top4_prob"],
            "top6_prob": outcomes["top6_prob"],
            "relegation_prob": outcomes["relegation_prob"],
            "expected_position": outcomes["expected_position"],
            "position_std": outcomes["position_std"],
            "expected_points": outcomes["expected_points"],
            "points_std": outcomes["points_std"],
        })

    return {
        "week": week,
        "predictions": predictions,
        "team_states": {name: state.to_dict() for name, state in team_states.items()},
    }


def calculate_standings_at_week(results: List[MatchResult], week: int, teams: List[str]) -> Dict[str, int]:
    """Calculate points standings through a given week."""
    points = {team: 0 for team in teams}

    for result in results:
        if result.matchweek > week:
            continue

        home = result.home_team
        away = result.away_team

        if result.home_goals > result.away_goals:
            # Home win
            if home in points:
                points[home] += 3
        elif result.home_goals < result.away_goals:
            # Away win
            if away in points:
                points[away] += 3
        else:
            # Draw
            if home in points:
                points[home] += 1
            if away in points:
                points[away] += 1

    return points


def run_backfill(start_week: int = 0, end_week: int = None):
    """
    Run the historical backfill.

    Args:
        start_week: First week to simulate (0 = preseason)
        end_week: Last week to simulate (None = current week)
    """
    simulator = MonteCarloSimulator()
    engine = BayesianEngine()
    db = Database(str(DB_PATH))

    logger.info("Loading match data from database (no API calls needed)...")
    all_fixtures, all_results, current_week = get_all_match_data_from_db(db)

    if end_week is None:
        end_week = current_week

    logger.info(f"Backfilling weeks {start_week} through {end_week}")
    logger.info(f"Total fixtures: {len(all_fixtures)}, Total results: {len(all_results)}")

    # Get list of teams
    teams = list(set([f.home_team for f in all_fixtures] + [f.away_team for f in all_fixtures]))
    logger.info(f"Teams: {len(teams)}")

    # Load fresh initial team states for each backfill
    # We'll re-initialize for each run to get clean starting point

    for week in range(start_week, end_week + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Simulating Week {week}...")

        # Reload initial team states (fresh start)
        team_states = load_initial_team_states(str(DATA_DIR))

        # Get results through this week
        results_through_week = [r for r in all_results if r.matchweek <= week]

        # Calculate standings at this week
        standings_at_week = calculate_standings_at_week(all_results, week, teams)

        logger.info(f"  Results through week {week}: {len(results_through_week)} matches")
        logger.info(f"  Total points awarded: {sum(standings_at_week.values())}")

        # Run simulation for this week
        result = simulate_week(
            week=week,
            team_states=team_states,
            all_fixtures=all_fixtures,
            results_through_week=results_through_week,
            standings_at_week=standings_at_week,
            simulator=simulator,
            engine=engine,
        )

        # Save to database
        logger.info(f"  Saving week {week} to database...")

        # Save team state history
        for team_name, state in team_states.items():
            db.save_team_state_history(state, week)

        # Save season predictions
        for pred_dict in result["predictions"]:
            pred = SeasonPrediction(
                team=pred_dict["team"],
                position_probs=pred_dict["position_probs"],
                title_prob=pred_dict["title_prob"],
                top4_prob=pred_dict["top4_prob"],
                top6_prob=pred_dict["top6_prob"],
                relegation_prob=pred_dict["relegation_prob"],
                expected_position=pred_dict["expected_position"],
                position_std=pred_dict["position_std"],
                expected_points=pred_dict["expected_points"],
                points_std=pred_dict["points_std"],
            )
            db.save_season_prediction(pred, week)

        # Log some interesting stats
        title_leader = max(result["predictions"], key=lambda x: x["title_prob"])
        logger.info(f"  Title favorite: {title_leader['team']} ({title_leader['title_prob']*100:.1f}%)")

        relegation_leader = max(result["predictions"], key=lambda x: x["relegation_prob"])
        logger.info(f"  Most likely relegated: {relegation_leader['team']} ({relegation_leader['relegation_prob']*100:.1f}%)")

    logger.info(f"\n{'='*50}")
    logger.info("Backfill complete!")
    logger.info(f"Historical data now available for weeks {start_week}-{end_week}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="MOTSON Historical Backfill")
    parser.add_argument("--start-week", type=int, default=0, help="First week to simulate (0 = preseason)")
    parser.add_argument("--end-week", type=int, default=None, help="Last week to simulate (default: current)")
    args = parser.parse_args()

    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)

    try:
        run_backfill(start_week=args.start_week, end_week=args.end_week)
        logger.info("Backfill completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
