"""
MOTSON Bayesian Blending Engine

Combines 5-year gravity priors with current season IRT estimates using
principled Bayesian updating, with a floor on gravity weight (min 50%).

Philosophy: "Track distributions, not point estimates. Blend historical
gravity with current-season momentum using principled Bayesian updating."
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

from .irt_model import fit_irt_model, IRTParameters, gap_to_probabilities
from .irt_state import IRTTeamState, IRTMatchPrediction

logger = logging.getLogger(__name__)


# Default prior standard deviation for 5-year estimates
# This controls how much the prior influences the blend
# Larger = less influence from prior
DEFAULT_PRIOR_STD = 0.35

# Minimum weight on gravity prior (never less than 50%)
MIN_GRAVITY_WEIGHT = 0.5


def bayesian_blend(
    prior_mean: float,
    prior_std: float,
    likelihood_mean: float,
    likelihood_std: float,
    min_prior_weight: float = MIN_GRAVITY_WEIGHT,
) -> Tuple[float, float, float, float]:
    """
    Combine prior (gravity) and likelihood (current season) using Bayesian updating.

    Posterior = weighted average where weights are inverse variances.

    With a floor on prior weight to ensure gravity is always at least 50%.

    Args:
        prior_mean: Prior mean (from 5-year estimation)
        prior_std: Prior standard deviation
        likelihood_mean: Current season estimate
        likelihood_std: Standard error from current season
        min_prior_weight: Minimum weight for prior (default 0.5)

    Returns:
        (posterior_mean, posterior_std, gravity_weight, momentum_weight)
    """
    prior_var = prior_std ** 2
    likelihood_var = likelihood_std ** 2

    # Calculate natural weights (inverse variance weighting)
    prior_precision = 1 / prior_var
    likelihood_precision = 1 / likelihood_var
    total_precision = prior_precision + likelihood_precision

    natural_prior_weight = prior_precision / total_precision
    natural_likelihood_weight = likelihood_precision / total_precision

    # Apply floor on prior weight
    if natural_prior_weight < min_prior_weight:
        gravity_weight = min_prior_weight
        momentum_weight = 1 - min_prior_weight
    else:
        gravity_weight = natural_prior_weight
        momentum_weight = natural_likelihood_weight

    # Blended estimate
    posterior_mean = gravity_weight * prior_mean + momentum_weight * likelihood_mean

    # Posterior variance (standard Bayesian formula)
    posterior_var = 1 / total_precision
    posterior_std = np.sqrt(posterior_var)

    return posterior_mean, posterior_std, gravity_weight, momentum_weight


def load_five_year_priors(priors_path: str = "data/five_year_priors.json") -> Dict:
    """
    Load 5-year priors from JSON file.

    Returns dict with team -> {theta, theta_se, b_home, b_home_se, b_away, b_away_se}
    """
    path = Path(priors_path)
    if not path.exists():
        # Try relative to project root
        path = Path(__file__).parent.parent.parent / priors_path

    if not path.exists():
        raise FileNotFoundError(f"Five-year priors not found: {priors_path}")

    with open(path) as f:
        data = json.load(f)

    return data


class BayesianBlender:
    """
    Blends 5-year gravity priors with current season IRT estimates.

    Usage:
        blender = BayesianBlender()
        blender.load_priors()
        states = blender.initialize_season_states(current_epl_teams, promoted_teams)

        # Each week:
        states = blender.update_week(week_number, season_matches, states)
    """

    def __init__(self, priors_path: str = "data/five_year_priors.json"):
        self.priors_path = priors_path
        self.priors = None
        self.archetype_mapping = {}

    def load_priors(self):
        """Load 5-year priors from file."""
        data = load_five_year_priors(self.priors_path)
        self.priors = data["teams"]
        self.archetype_mapping = data.get("archetype_mapping", {})
        logger.info(f"Loaded priors for {len(self.priors)} teams/archetypes")

    def get_prior_for_team(self, team: str) -> Dict:
        """
        Get prior parameters for a team.

        For promoted teams, returns the archetype prior.
        """
        if team in self.priors:
            return self.priors[team]

        # Check if this is a promoted team that needs archetype
        # Default to middle archetype for unknown teams
        if team in self.archetype_mapping:
            archetype = self.archetype_mapping[team]
            if archetype in self.priors:
                return self.priors[archetype]

        # Check known promoted team lists
        from backend.config import EPL_TEAMS_2025_26

        # Default archetype for unknown promoted teams
        if "Archetype_Promoted_2" in self.priors:
            return self.priors["Archetype_Promoted_2"]

        # Ultimate fallback: average team
        return {
            "theta": 0.0,
            "theta_se": DEFAULT_PRIOR_STD,
            "b_home": 0.0,
            "b_home_se": DEFAULT_PRIOR_STD,
            "b_away": -0.15,
            "b_away_se": DEFAULT_PRIOR_STD,
        }

    def initialize_season_states(
        self,
        teams: List[str],
        promoted_teams: List[str] = None,
    ) -> Dict[str, IRTTeamState]:
        """
        Initialize team states at the start of a season.

        At season start, all estimates come from priors (no season data yet).

        Args:
            teams: List of team names in the league
            promoted_teams: List of teams that were promoted this season

        Returns:
            Dict of team -> IRTTeamState
        """
        if self.priors is None:
            self.load_priors()

        promoted_teams = promoted_teams or []
        states = {}

        for team in teams:
            prior = self.get_prior_for_team(team)

            # At season start, posterior = prior (no season data)
            states[team] = IRTTeamState(
                team=team,
                # Blended posterior (= prior at start)
                theta=prior["theta"],
                theta_se=prior["theta_se"],
                b_home=prior["b_home"],
                b_home_se=prior["b_home_se"],
                b_away=prior["b_away"],
                b_away_se=prior["b_away_se"],
                # 5-year priors
                theta_prior=prior["theta"],
                theta_prior_se=prior["theta_se"],
                b_home_prior=prior["b_home"],
                b_home_prior_se=prior["b_home_se"],
                b_away_prior=prior["b_away"],
                b_away_prior_se=prior["b_away_se"],
                # Season estimates (none yet)
                theta_season=0.0,
                theta_season_se=1.0,  # Very uncertain
                b_home_season=0.0,
                b_home_season_se=1.0,
                b_away_season=0.0,
                b_away_season_se=1.0,
                # Weights
                gravity_weight=1.0,  # 100% prior at start
                momentum_weight=0.0,
                # Tracking
                matches_played=0,
                expected_points_season=0.0,
                actual_points_season=0,
                last_updated=datetime.utcnow(),
                is_promoted=team in promoted_teams,
            )

        logger.info(f"Initialized {len(states)} team states for season")
        return states

    def update_week(
        self,
        week: int,
        season_matches: List[Dict],
        current_states: Dict[str, IRTTeamState],
        n_warmup: int = 300,
        n_samples: int = 500,
    ) -> Dict[str, IRTTeamState]:
        """
        Update team states after a week of matches.

        Runs fresh IRT on ALL season matches (not incremental), then blends
        with 5-year priors.

        Args:
            week: Current week number
            season_matches: ALL matches this season so far (not just this week)
                           Each match: {home_team, away_team, outcome (0/1/2)}
            current_states: Current team states
            n_warmup: MCMC warmup samples
            n_samples: MCMC posterior samples

        Returns:
            Updated team states
        """
        teams = list(current_states.keys())
        team_to_idx = {t: i for i, t in enumerate(teams)}

        # Count matches per team
        team_matches = {t: 0 for t in teams}
        team_points = {t: 0 for t in teams}
        for m in season_matches:
            home = m["home_team"]
            away = m["away_team"]
            outcome = m["outcome"]

            team_matches[home] = team_matches.get(home, 0) + 1
            team_matches[away] = team_matches.get(away, 0) + 1

            if outcome == 2:  # Home win
                team_points[home] = team_points.get(home, 0) + 3
            elif outcome == 0:  # Away win
                team_points[away] = team_points.get(away, 0) + 3
            else:  # Draw
                team_points[home] = team_points.get(home, 0) + 1
                team_points[away] = team_points.get(away, 0) + 1

        if not season_matches:
            logger.info("No season matches yet, returning initial states")
            return current_states

        # Run fresh IRT on season data
        logger.info(f"Running IRT on {len(season_matches)} season matches (week {week})...")

        # Use priors as starting point for IRT
        prior_means = {
            team: current_states[team].theta_prior
            for team in teams
        }

        try:
            irt_params = fit_irt_model(
                season_matches,
                team_to_idx,
                theta_prior_mean=prior_means,
                theta_prior_std=0.5,  # Allow movement from prior
                n_warmup=n_warmup,
                n_samples=n_samples,
            )
        except Exception as e:
            logger.error(f"IRT fitting failed: {e}")
            return current_states

        # Blend IRT results with priors
        updated_states = {}

        for team in teams:
            state = current_states[team]

            # Get season IRT estimates
            theta_season = irt_params.theta[team]
            theta_season_se = irt_params.theta_se[team]
            b_home_season = irt_params.b_home[team]
            b_home_season_se = irt_params.b_home_se[team]
            b_away_season = irt_params.b_away[team]
            b_away_season_se = irt_params.b_away_se[team]

            # Bayesian blend for each parameter
            theta_post, theta_post_se, grav_w, mom_w = bayesian_blend(
                state.theta_prior, state.theta_prior_se,
                theta_season, theta_season_se,
            )

            b_home_post, b_home_post_se, _, _ = bayesian_blend(
                state.b_home_prior, state.b_home_prior_se,
                b_home_season, b_home_season_se,
            )

            b_away_post, b_away_post_se, _, _ = bayesian_blend(
                state.b_away_prior, state.b_away_prior_se,
                b_away_season, b_away_season_se,
            )

            # Calculate expected points for this team
            expected_pts = self._calculate_expected_points(
                team, season_matches, current_states
            )

            updated_states[team] = IRTTeamState(
                team=team,
                # Blended posterior
                theta=theta_post,
                theta_se=theta_post_se,
                b_home=b_home_post,
                b_home_se=b_home_post_se,
                b_away=b_away_post,
                b_away_se=b_away_post_se,
                # Priors (unchanged)
                theta_prior=state.theta_prior,
                theta_prior_se=state.theta_prior_se,
                b_home_prior=state.b_home_prior,
                b_home_prior_se=state.b_home_prior_se,
                b_away_prior=state.b_away_prior,
                b_away_prior_se=state.b_away_prior_se,
                # Season estimates
                theta_season=theta_season,
                theta_season_se=theta_season_se,
                b_home_season=b_home_season,
                b_home_season_se=b_home_season_se,
                b_away_season=b_away_season,
                b_away_season_se=b_away_season_se,
                # Weights
                gravity_weight=grav_w,
                momentum_weight=mom_w,
                # Tracking
                matches_played=team_matches.get(team, 0),
                expected_points_season=expected_pts,
                actual_points_season=team_points.get(team, 0),
                last_updated=datetime.utcnow(),
                is_promoted=state.is_promoted,
            )

        logger.info(f"Updated {len(updated_states)} team states for week {week}")
        return updated_states

    def _calculate_expected_points(
        self,
        team: str,
        matches: List[Dict],
        states: Dict[str, IRTTeamState],
    ) -> float:
        """Calculate expected points for a team based on pre-match predictions."""
        expected = 0.0

        for m in matches:
            home = m["home_team"]
            away = m["away_team"]

            if team not in [home, away]:
                continue

            # Calculate prediction using states BEFORE the match
            # (In practice, we're using current states as approximation)
            home_state = states.get(home)
            away_state = states.get(away)

            if not home_state or not away_state:
                continue

            # Calculate gap
            m_home = home_state.theta - away_state.b_away
            m_away = away_state.theta - home_state.b_home
            gap = m_home - m_away

            h_prob, d_prob, a_prob = gap_to_probabilities(gap)

            if team == home:
                expected += 3 * h_prob + 1 * d_prob
            else:
                expected += 3 * a_prob + 1 * d_prob

        return expected

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        states: Dict[str, IRTTeamState],
    ) -> IRTMatchPrediction:
        """
        Predict match outcome probabilities.

        Args:
            home_team: Home team name
            away_team: Away team name
            states: Current team states

        Returns:
            IRTMatchPrediction with probabilities and gap breakdown
        """
        home_state = states[home_team]
        away_state = states[away_team]

        # Calculate attack margins
        m_home = home_state.theta - away_state.b_away
        m_away = away_state.theta - home_state.b_home

        # Gap from home team perspective
        gap = m_home - m_away

        # Convert to probabilities
        h_prob, d_prob, a_prob = gap_to_probabilities(gap)

        # Confidence based on combined SEs
        combined_se = np.sqrt(
            home_state.theta_se**2 +
            away_state.theta_se**2 +
            home_state.b_home_se**2 +
            away_state.b_away_se**2
        )
        # Map SE to confidence: lower SE = higher confidence
        confidence = max(0.1, 1.0 - combined_se / 1.2)

        return IRTMatchPrediction(
            match_id=f"{home_team}_vs_{away_team}",
            matchweek=0,  # To be filled by caller
            home_team=home_team,
            away_team=away_team,
            home_win_prob=h_prob,
            draw_prob=d_prob,
            away_win_prob=a_prob,
            gap=gap,
            m_home=m_home,
            m_away=m_away,
            home_theta=home_state.theta,
            home_b_home=home_state.b_home,
            away_theta=away_state.theta,
            away_b_away=away_state.b_away,
            confidence=confidence,
        )
