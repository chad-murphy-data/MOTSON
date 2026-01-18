"""
MOTSON v2 - Weekly Update Pipeline

Orchestrates the full weekly update:
1. Fetch latest results from API
2. Update team states using cumulative calibration
3. Re-run Monte Carlo for season outcomes
4. Generate next week's predictions
5. Store everything in database
"""

import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd

from ..models.team_state import (
    TeamState,
    MatchResult,
    Fixture,
    MatchPrediction,
    SeasonPrediction,
    UpdateExplanation,
)
from ..models.bayesian_engine import (
    BayesianEngine,
    predict_match,
    initialize_team_states,
)
from .data_fetcher import FootballDataAPI
from .monte_carlo import MonteCarloSimulator
from .survival_calibration import calibrate_title_probabilities
from ..config import model_config, app_config

logger = logging.getLogger(__name__)


class WeeklyUpdatePipeline:
    """
    Orchestrates the weekly prediction update.

    The key insight: we update based on CUMULATIVE calibration,
    not individual match surprises.
    """

    def __init__(
        self,
        api_key: str = None,
        db=None,  # Database connection (will be added)
    ):
        self.api = FootballDataAPI(api_key)
        self.engine = BayesianEngine()
        self.simulator = MonteCarloSimulator()
        self.db = db

    async def run_update(
        self,
        team_states: Dict[str, TeamState],
        force: bool = False,
    ) -> Dict:
        """
        Run the full weekly update pipeline.

        Returns a summary of what changed.
        """
        logger.info("Starting weekly update pipeline")

        # 1. Fetch current data from API
        logger.info("Fetching results from API...")
        results = await self.api.get_finished_matches()
        all_fixtures = await self.api.get_all_fixtures()
        standings = await self.api.get_standings()

        current_week = await self.api.get_current_matchweek()
        logger.info(f"Current matchweek: {current_week}")

        # 2. Organize results by team
        team_results = self._organize_results_by_team(results, team_states.keys())
        team_fixtures = self._organize_fixtures_by_team(all_fixtures, team_states.keys())

        # 3. Generate predictions for completed matches (for calibration)
        match_predictions = self._generate_match_predictions(
            [f for f in all_fixtures if f.status == "FINISHED"],
            team_states,
        )

        # 4. Update each team's state
        logger.info("Updating team states...")
        explanations = []
        for team_name, team in team_states.items():
            # Get this team's predictions and results
            team_preds = [
                (pred, pred.home_team == team_name)
                for pred in match_predictions
                if pred.home_team == team_name or pred.away_team == team_name
            ]
            team_res = [
                (res, res.home_team == team_name)
                for res in results
                if res.home_team == team_name or res.away_team == team_name
            ]

            # Run Bayesian update
            updated_team, explanation = self.engine.weekly_update(
                team=team,
                week=current_week,
                match_predictions=team_preds,
                match_results=team_res,
            )

            team_states[team_name] = updated_team
            team_states[team_name].last_updated = datetime.utcnow()
            explanations.append(explanation)

            if explanation.update_triggered:
                logger.info(f"  {team_name}: {explanation.reason}")

        # 5. Get remaining fixtures (anything not FINISHED)
        # Note: API uses "TIMED" for scheduled matches, "SCHEDULED" is less common
        remaining_fixtures = [f for f in all_fixtures if f.status != "FINISHED"]
        logger.info(f"Found {len(remaining_fixtures)} remaining fixtures to simulate")

        # 6. Get current points
        current_points = {s["team"]: s["points"] for s in standings}

        # 7. Run Monte Carlo simulation
        logger.info(f"Running Monte Carlo simulation ({model_config.DEFAULT_SIMULATIONS} sims)...")
        season_outcomes = self.simulator.simulate_season(
            team_states=team_states,
            remaining_fixtures=remaining_fixtures,
            current_points=current_points,
            n_simulations=model_config.DEFAULT_SIMULATIONS,
            week=current_week,  # Pass week for preseason uncertainty decay
        )

        # 7.5 Apply survival calibration to title probabilities
        # This blends MC output with historical lead survival rates
        games_remaining = 38 - current_week
        team_thetas = {name: state.effective_theta_home for name, state in team_states.items()}
        season_outcomes = calibrate_title_probabilities(
            mc_results=season_outcomes,
            current_standings=current_points,
            games_remaining=games_remaining,
            team_thetas=team_thetas,
        )

        # 8. Generate next week's predictions
        next_week_fixtures = [f for f in remaining_fixtures if f.matchweek == current_week + 1]
        next_week_predictions = [
            predict_match(f, team_states) for f in next_week_fixtures
        ]

        # 9. Compile season predictions
        season_predictions = []
        for team_name, outcomes in season_outcomes.items():
            season_predictions.append(SeasonPrediction(
                team=team_name,
                position_probs=outcomes["position_probs"],
                title_prob=outcomes["title_prob"],
                top4_prob=outcomes["top4_prob"],
                top6_prob=outcomes["top6_prob"],
                relegation_prob=outcomes["relegation_prob"],
                expected_position=outcomes["expected_position"],
                position_std=outcomes["position_std"],
                expected_points=outcomes["expected_points"],
                points_std=outcomes["points_std"],
            ))

        logger.info("Weekly update complete")

        return {
            "week": current_week,
            "timestamp": datetime.utcnow().isoformat(),
            "standings": standings,
            "team_states": {name: state.to_dict() for name, state in team_states.items()},
            "explanations": [e.to_dict() for e in explanations],
            "next_week_predictions": [p.to_dict() for p in next_week_predictions],
            "season_predictions": [p.to_dict() for p in season_predictions],
            "updates_triggered": sum(1 for e in explanations if e.update_triggered),
        }

    def _organize_results_by_team(
        self,
        results: List[MatchResult],
        teams: List[str],
    ) -> Dict[str, List[Tuple[MatchResult, bool]]]:
        """Organize results by team, tracking home/away."""
        team_results = {team: [] for team in teams}

        for result in results:
            if result.home_team in team_results:
                team_results[result.home_team].append((result, True))
            if result.away_team in team_results:
                team_results[result.away_team].append((result, False))

        return team_results

    def _organize_fixtures_by_team(
        self,
        fixtures: List[Fixture],
        teams: List[str],
    ) -> Dict[str, List[Tuple[Fixture, bool]]]:
        """Organize fixtures by team, tracking home/away."""
        team_fixtures = {team: [] for team in teams}

        for fixture in fixtures:
            if fixture.home_team in team_fixtures:
                team_fixtures[fixture.home_team].append((fixture, True))
            if fixture.away_team in team_fixtures:
                team_fixtures[fixture.away_team].append((fixture, False))

        return team_fixtures

    def _generate_match_predictions(
        self,
        fixtures: List[Fixture],
        team_states: Dict[str, TeamState],
    ) -> List[MatchPrediction]:
        """Generate predictions for a list of fixtures."""
        predictions = []
        for fixture in fixtures:
            try:
                pred = predict_match(fixture, team_states)
                predictions.append(pred)
            except KeyError as e:
                logger.warning(f"Skipping fixture {fixture.match_id}: unknown team {e}")

        return predictions


def load_initial_team_states(data_dir: str = "data") -> Dict[str, TeamState]:
    """
    Load initial team states from CSV files.

    Uses:
    - team_parameters.csv: stickiness, sigma, gravity
    - MOTSON_GPCM_Theta_Alpha.csv: initial theta values
    """
    import os

    params_path = os.path.join(data_dir, "team_parameters.csv")
    theta_path = os.path.join(data_dir, "MOTSON_GPCM_Theta_Alpha.csv")

    # Load team parameters
    params_df = pd.read_csv(params_path)

    # Load theta values (optional)
    theta_df = None
    if os.path.exists(theta_path):
        theta_df = pd.read_csv(theta_path)

    return initialize_team_states(params_df, theta_df)
