"""
MOTSON v2 - Monte Carlo Season Simulation

Simulates the remaining season thousands of times to generate:
- Position probability distributions
- Title/Top4/Relegation probabilities
- Expected points with uncertainty

This is where we get the "10,000 futures" for each team.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

from ..models.team_state import TeamState, Fixture, SeasonPrediction
from ..models.bayesian_engine import predict_match_probs
from ..config import model_config

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    Monte Carlo simulation for EPL season outcomes.

    Fast and embarrassingly parallel - we can run 10,000+ simulations
    in under a second for a typical remaining-season scenario.
    """

    def __init__(self, cfg=model_config):
        self.cfg = cfg

    def simulate_season(
        self,
        team_states: Dict[str, TeamState],
        remaining_fixtures: List[Fixture],
        current_points: Dict[str, int],
        n_simulations: int = 10000,
        seed: Optional[int] = None,
    ) -> Dict[str, Dict]:
        """
        Run Monte Carlo simulation of remaining season.

        Args:
            team_states: Current state for each team
            remaining_fixtures: Fixtures yet to be played
            current_points: Current points for each team
            n_simulations: Number of simulations to run
            seed: Random seed for reproducibility

        Returns:
            Dictionary with per-team outcome statistics
        """
        rng = np.random.default_rng(seed)

        teams = list(team_states.keys())
        n_teams = len(teams)
        n_fixtures = len(remaining_fixtures)

        logger.info(f"Running {n_simulations} simulations for {n_fixtures} remaining fixtures")

        # Pre-compute match probabilities for efficiency
        # Shape: (n_fixtures, 3) for [away_win, draw, home_win]
        match_probs = np.zeros((n_fixtures, 3))
        fixture_teams = []  # (home_idx, away_idx) for each fixture

        team_to_idx = {team: i for i, team in enumerate(teams)}

        for i, fixture in enumerate(remaining_fixtures):
            home_state = team_states[fixture.home_team]
            away_state = team_states[fixture.away_team]

            h_prob, d_prob, a_prob, _ = predict_match_probs(
                home_state.effective_theta_home,
                away_state.effective_theta_away,
                home_state.sigma,
                away_state.sigma,
            )

            match_probs[i] = [a_prob, d_prob, h_prob]
            fixture_teams.append((
                team_to_idx[fixture.home_team],
                team_to_idx[fixture.away_team],
            ))

        # Convert current points to array
        starting_points = np.array([current_points.get(team, 0) for team in teams])

        # Run simulations (vectorized for speed)
        # Shape: (n_simulations, n_teams)
        final_points = np.tile(starting_points, (n_simulations, 1))

        # Simulate each match
        for fixture_idx in range(n_fixtures):
            home_idx, away_idx = fixture_teams[fixture_idx]
            probs = match_probs[fixture_idx]

            # Sample outcomes for all simulations at once
            outcomes = rng.choice(3, size=n_simulations, p=probs)

            # 0 = away win, 1 = draw, 2 = home win
            # Home points: 0 for away win, 1 for draw, 3 for home win
            # Away points: 3 for away win, 1 for draw, 0 for home win
            home_points = np.where(outcomes == 2, 3, np.where(outcomes == 1, 1, 0))
            away_points = np.where(outcomes == 0, 3, np.where(outcomes == 1, 1, 0))

            final_points[:, home_idx] += home_points
            final_points[:, away_idx] += away_points

        # Determine final positions for each simulation
        # We need to handle ties - use random tiebreaker for simplicity
        # In reality, would use goal difference, but we're not tracking goals here
        tiebreaker = rng.random((n_simulations, n_teams)) * 0.001
        final_points_with_tiebreak = final_points + tiebreaker

        # argsort gives indices that would sort the array
        # We want descending order (highest points first)
        # So we negate and sort
        sorted_indices = np.argsort(-final_points_with_tiebreak, axis=1)

        # Convert to positions (rank)
        positions = np.zeros_like(sorted_indices)
        for sim in range(n_simulations):
            for pos, team_idx in enumerate(sorted_indices[sim]):
                positions[sim, team_idx] = pos + 1  # 1-indexed

        # Compile statistics for each team
        results = {}
        for team_idx, team in enumerate(teams):
            team_positions = positions[:, team_idx]
            team_points = final_points[:, team_idx]

            # Position distribution
            pos_counts = np.bincount(team_positions.astype(int), minlength=n_teams + 1)[1:]
            pos_probs = pos_counts / n_simulations

            # Key probabilities
            title_prob = pos_probs[0] if len(pos_probs) > 0 else 0
            top4_prob = pos_probs[:4].sum() if len(pos_probs) >= 4 else pos_probs.sum()
            top6_prob = pos_probs[:6].sum() if len(pos_probs) >= 6 else pos_probs.sum()
            relegation_prob = pos_probs[-3:].sum() if len(pos_probs) >= 3 else 0

            # Expected position and points
            expected_position = team_positions.mean()
            position_std = team_positions.std()
            expected_points = team_points.mean()
            points_std = team_points.std()

            results[team] = {
                "position_probs": pos_probs.tolist(),
                "title_prob": float(title_prob),
                "top4_prob": float(top4_prob),
                "top6_prob": float(top6_prob),
                "relegation_prob": float(relegation_prob),
                "expected_position": float(expected_position),
                "position_std": float(position_std),
                "expected_points": float(expected_points),
                "points_std": float(points_std),
            }

        return results

    def simulate_counterfactual(
        self,
        team_states: Dict[str, TeamState],
        all_fixtures: List[Fixture],
        counterfactual_results: Dict[str, str],  # match_id -> "H", "D", or "A"
        n_simulations: int = 10000,
        seed: Optional[int] = None,
    ) -> Dict[str, Dict]:
        """
        Simulate a counterfactual scenario.

        "What if City had beaten Burnley in week 3?"

        Args:
            team_states: Current team states
            all_fixtures: All season fixtures
            counterfactual_results: Overridden results for specific matches
            n_simulations: Number of simulations
            seed: Random seed

        Returns:
            Same format as simulate_season
        """
        # Compute current points up to now, applying counterfactuals
        current_points = {team: 0 for team in team_states}

        played_match_ids = set()
        for fixture in all_fixtures:
            if fixture.status == "FINISHED":
                played_match_ids.add(fixture.match_id)

                # Check if this is a counterfactual
                if fixture.match_id in counterfactual_results:
                    result = counterfactual_results[fixture.match_id]
                else:
                    # Use actual result
                    if fixture.home_goals > fixture.away_goals:
                        result = "H"
                    elif fixture.home_goals < fixture.away_goals:
                        result = "A"
                    else:
                        result = "D"

                # Award points
                if result == "H":
                    current_points[fixture.home_team] += 3
                elif result == "A":
                    current_points[fixture.away_team] += 3
                else:
                    current_points[fixture.home_team] += 1
                    current_points[fixture.away_team] += 1

        # Get remaining fixtures
        remaining = [f for f in all_fixtures if f.match_id not in played_match_ids]

        # Run simulation
        return self.simulate_season(
            team_states=team_states,
            remaining_fixtures=remaining,
            current_points=current_points,
            n_simulations=n_simulations,
            seed=seed,
        )


def quick_simulate(
    team_states: Dict[str, TeamState],
    remaining_fixtures: List[Fixture],
    current_points: Dict[str, int],
) -> Dict[str, Dict]:
    """Quick simulation with fewer iterations for testing."""
    simulator = MonteCarloSimulator()
    return simulator.simulate_season(
        team_states=team_states,
        remaining_fixtures=remaining_fixtures,
        current_points=current_points,
        n_simulations=model_config.QUICK_SIMULATIONS,
    )
