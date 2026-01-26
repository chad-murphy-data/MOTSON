#!/usr/bin/env python3
"""
Fetch and track coaching changes from the football-data.org API.

This script:
1. Fetches current coach info for all teams
2. Compares with previous snapshot to detect changes
3. Records any changes to the coaching_history table
4. Saves current snapshot for next comparison

Run this weekly (part of the update workflow) to track manager changes.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.data_fetcher import FootballDataAPI
from backend.database.db import get_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    # Check for API key
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not api_key:
        logger.error("FOOTBALL_DATA_API_KEY not set!")
        sys.exit(1)

    api = FootballDataAPI(api_key)
    db = get_db()

    # Get current week
    current_week = db.get_current_week()
    logger.info(f"Current week: {current_week}")

    # Fetch current coaches from API
    logger.info("Fetching team coach data from API...")
    teams = await api.get_teams()

    changes_detected = []

    for team_data in teams:
        team = team_data["team"]
        coach_name = team_data["coach_name"]
        coach_id = team_data["coach_id"]

        if not coach_name:
            logger.warning(f"No coach data for {team}")
            continue

        # Get previous snapshot
        previous = db.get_latest_coaching_snapshot(team)

        if previous is None:
            # First time seeing this team - initialize coaching history
            logger.info(f"Initializing coach for {team}: {coach_name}")
            db.save_coaching_change(team, coach_name, coach_id, 0)
        elif previous["coach_id"] != coach_id:
            # Coach changed!
            logger.info(f"COACHING CHANGE: {team}: {previous['coach_name']} -> {coach_name}")
            db.save_coaching_change(team, coach_name, coach_id, current_week)
            changes_detected.append({
                "team": team,
                "old_coach": previous["coach_name"],
                "new_coach": coach_name,
                "week": current_week,
            })

        # Save current snapshot
        db.save_coaching_snapshot(team, current_week, coach_name, coach_id)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("COACHING UPDATE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Teams checked: {len(teams)}")
    logger.info(f"Changes detected: {len(changes_detected)}")

    if changes_detected:
        logger.info("\nChanges this week:")
        for change in changes_detected:
            logger.info(f"  {change['team']}: {change['old_coach']} -> {change['new_coach']}")

    # Show all current coaches
    logger.info("\nCurrent coaches:")
    current_coaches = db.get_current_coaches()
    for coach in sorted(current_coaches, key=lambda x: x["team"]):
        weeks = f"(since week {coach['week_started']})" if coach['week_started'] > 0 else "(start of season)"
        logger.info(f"  {coach['team']:20} {coach['coach_name']} {weeks}")


if __name__ == "__main__":
    asyncio.run(main())
