#!/usr/bin/env python3
"""
MOTSON Scheduled Update Script

This script is designed to be run by GitHub Actions on a schedule.
It checks if new match results are available and runs the update pipeline if so.

Usage:
    python scripts/scheduled_update.py [--force]

Options:
    --force     Run update even if no new results detected
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.data_fetcher import FootballDataAPI
from backend.services.weekly_update import WeeklyUpdatePipeline, load_initial_team_states
from backend.services.monte_carlo import MonteCarloSimulator
from backend.models.team_state import SeasonPrediction
from backend.database.db import Database
from backend.config import model_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = project_root / "data"
DB_PATH = DATA_DIR / "motson.db"
LAST_UPDATE_PATH = DATA_DIR / "last_update.json"


def load_last_update() -> dict:
    """Load last update metadata from JSON file."""
    if LAST_UPDATE_PATH.exists():
        with open(LAST_UPDATE_PATH, "r") as f:
            return json.load(f)
    return {"last_matchweek": 0, "last_update_timestamp": None}


def save_last_update(matchweek: int, teams_updated: list, simulations: int):
    """Save update metadata to JSON file."""
    data = {
        "last_matchweek": matchweek,
        "last_update_timestamp": datetime.utcnow().isoformat() + "Z",
        "teams_updated": teams_updated,
        "simulations_run": simulations,
    }
    with open(LAST_UPDATE_PATH, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved last_update.json: week {matchweek}")


async def check_for_new_results(api: FootballDataAPI, last_matchweek: int) -> tuple[bool, int]:
    """
    Check if there are new match results since last update.

    Returns:
        (has_new_results, current_matchweek)
    """
    try:
        current_week = await api.get_current_matchweek()
        logger.info(f"Current matchweek from API: {current_week}")
        logger.info(f"Last updated matchweek: {last_matchweek}")

        has_new = current_week > last_matchweek
        return has_new, current_week
    except Exception as e:
        logger.error(f"Failed to check for new results: {e}")
        raise


async def run_update_pipeline(db: Database, force: bool = False) -> dict:
    """
    Run the full update pipeline.

    Args:
        db: Database instance
        force: Run even if no new results

    Returns:
        Update result dict
    """
    api = FootballDataAPI()

    # Check for new results
    last_update = load_last_update()
    last_matchweek = last_update.get("last_matchweek", 0)

    has_new_results, current_week = await check_for_new_results(api, last_matchweek)

    if not has_new_results and not force:
        logger.info(f"No new results since week {last_matchweek}. Skipping update.")
        return {
            "skipped": True,
            "reason": "No new results",
            "last_matchweek": last_matchweek,
            "current_matchweek": current_week,
        }

    if force:
        logger.info("Force flag set - running update regardless of new results")

    # Load team states from database or CSV
    team_states = db.get_all_team_states()

    if not team_states:
        logger.info("No team states in database - initializing from CSV")
        team_states = load_initial_team_states(str(DATA_DIR))
        db.save_team_states(team_states)

    # Run the update pipeline
    logger.info("Running weekly update pipeline...")
    pipeline = WeeklyUpdatePipeline()
    result = await pipeline.run_update(team_states)

    # Save to database
    logger.info("Saving results to database...")
    db.save_team_states(team_states)

    # Save state history
    for team_name, state in team_states.items():
        db.save_team_state_history(state, result["week"])

    # Save season predictions
    teams_updated = []
    for pred_dict in result["season_predictions"]:
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
        db.save_season_prediction(pred, result["week"])

    # Track which teams had their theta updated
    for explanation in result["explanations"]:
        if explanation["update_triggered"]:
            teams_updated.append(explanation["team"])

    # Update metadata
    db.set_metadata("current_week", str(result["week"]))
    db.set_metadata("last_update", datetime.utcnow().isoformat())

    # Save last_update.json for quick checking
    save_last_update(
        matchweek=result["week"],
        teams_updated=teams_updated,
        simulations=model_config.DEFAULT_SIMULATIONS,
    )

    logger.info(f"Update complete! Week {result['week']}, {result['updates_triggered']} team(s) updated")

    return {
        "skipped": False,
        "week": result["week"],
        "updates_triggered": result["updates_triggered"],
        "teams_updated": teams_updated,
        "timestamp": result["timestamp"],
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="MOTSON Scheduled Update")
    parser.add_argument("--force", action="store_true", help="Force update even if no new results")
    args = parser.parse_args()

    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)

    # Initialize database with explicit path
    db = Database(str(DB_PATH))

    # Run the update
    try:
        result = asyncio.run(run_update_pipeline(db, force=args.force))

        if result.get("skipped"):
            logger.info(f"Update skipped: {result.get('reason')}")
            # Exit with code 0 - this is expected behavior
            sys.exit(0)
        else:
            logger.info(f"Update completed successfully for week {result.get('week')}")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Update failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
