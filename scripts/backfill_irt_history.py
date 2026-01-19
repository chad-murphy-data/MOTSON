#!/usr/bin/env python3
"""
Backfill IRT History Script

Replays the season week-by-week, running fresh IRT estimation at each week
to build the historical trajectory of team parameters.

This creates the week-by-week data needed for the dashboard visualizations.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.database.db import get_db
from backend.models.bayesian_blender import BayesianBlender
from backend.models.irt_state import IRTTeamState
from backend.config import EPL_TEAMS_2025_26

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Teams promoted in 2025-26 season
PROMOTED_TEAMS_2025 = ["Leeds United", "Sunderland", "Burnley"]


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


def main():
    db = get_db()

    # Initialize the blender with 5-year priors
    blender = BayesianBlender()
    blender.load_priors()

    # Get all teams
    teams = EPL_TEAMS_2025_26

    # Initialize states at week 0 (pre-season)
    logger.info("Initializing team states from 5-year priors...")
    states = blender.initialize_season_states(teams, promoted_teams=PROMOTED_TEAMS_2025)

    # Save week 0 (pre-season) state
    for team, state in states.items():
        db.save_irt_team_state_history(state, week=0)
    logger.info("Saved week 0 (pre-season) states")

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

        # Run IRT update
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

        # Save state history for this week
        for team, state in states.items():
            db.save_irt_team_state_history(state, week=week)

        # Print top 5 teams
        sorted_teams = sorted(states.items(), key=lambda x: -x[1].theta)[:5]
        logger.info(f"Top 5 by theta (week {week}):")
        for team, state in sorted_teams:
            logger.info(f"  {team:20} theta={state.theta:+.3f} (SE={state.theta_se:.3f}) grav={state.gravity_weight:.2f}")

    # Save final states
    logger.info("\nSaving final IRT team states...")
    db.save_irt_team_states(states)

    logger.info("\n" + "=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Processed weeks 0-{current_week}")
    logger.info(f"Teams: {len(states)}")

    # Print final standings by theta
    logger.info("\nFinal team rankings by theta:")
    sorted_teams = sorted(states.items(), key=lambda x: -x[1].theta)
    for i, (team, state) in enumerate(sorted_teams, 1):
        logger.info(
            f"{i:2}. {team:20} theta={state.theta:+.3f} "
            f"b_home={state.b_home:+.3f} b_away={state.b_away:+.3f} "
            f"pts={state.actual_points_season}"
        )


if __name__ == "__main__":
    main()
