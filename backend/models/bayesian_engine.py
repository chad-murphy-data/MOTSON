"""
MOTSON v2 - Bayesian(-ish) Prediction Engine

Core philosophy: "Track distributions, not point estimates.
Update on cumulative calibration, not individual surprises."

This implements the approximate Bayesian approach:
- Cumulative z-score calibration for updates
- Stickiness-scaled learning rates
- Gravity decay over season
- Proper uncertainty propagation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

from .team_state import (
    TeamState,
    MatchPrediction,
    MatchResult,
    Fixture,
    UpdateExplanation,
)
from ..config import model_config, ANALYST_ADJUSTMENTS


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def predict_match_probs(
    home_theta: float,
    away_theta: float,
    home_sigma: float = 0.0,
    away_sigma: float = 0.0,
    cfg=model_config,
) -> Tuple[float, float, float, float]:
    """
    Predict match outcome probabilities.

    Args:
        home_theta: Home team's effective theta (including analyst adj)
        away_theta: Away team's effective theta (including analyst adj)
        home_sigma: Home team's uncertainty (for confidence calc)
        away_sigma: Away team's uncertainty (for confidence calc)
        cfg: Model configuration

    Returns:
        (home_win_prob, draw_prob, away_win_prob, confidence)
    """
    # Delta = home advantage
    delta = home_theta + cfg.HOME_ADVANTAGE - away_theta

    # Logits for [away_win, draw, home_win]
    # Draw probability decreases with mismatch (but not too fast!)
    z_away = -cfg.K_SCALE * delta
    z_draw = cfg.B0_DRAW - cfg.C_MISMATCH * abs(delta)
    z_home = cfg.K_SCALE * delta

    probs = softmax(np.array([z_away, z_draw, z_home]))

    # Confidence based on combined uncertainty
    # Lower sigma = higher confidence
    combined_sigma = np.sqrt(home_sigma**2 + away_sigma**2)
    # Map to 0-1: sigma=0 -> conf=1, sigma=0.6 -> conf~0.5
    confidence = max(0.1, 1.0 - combined_sigma / 0.6)

    return probs[2], probs[1], probs[0], confidence


def predict_match(
    fixture: Fixture,
    team_states: Dict[str, TeamState],
) -> MatchPrediction:
    """
    Generate prediction for a single fixture.
    """
    home_state = team_states[fixture.home_team]
    away_state = team_states[fixture.away_team]

    home_theta = home_state.effective_theta_home
    away_theta = away_state.effective_theta_away

    h_prob, d_prob, a_prob, conf = predict_match_probs(
        home_theta=home_theta,
        away_theta=away_theta,
        home_sigma=home_state.sigma,
        away_sigma=away_state.sigma,
    )

    return MatchPrediction(
        match_id=fixture.match_id,
        matchweek=fixture.matchweek,
        home_team=fixture.home_team,
        away_team=fixture.away_team,
        home_win_prob=h_prob,
        draw_prob=d_prob,
        away_win_prob=a_prob,
        confidence=conf,
        home_theta=home_theta,
        away_theta=away_theta,
        delta=home_theta + model_config.HOME_ADVANTAGE - away_theta,
    )


def calculate_expected_points(
    prediction: MatchPrediction,
    is_home: bool,
) -> float:
    """
    Calculate expected points for a team from a match prediction.

    Win = 3 pts, Draw = 1 pt, Loss = 0 pts
    E[pts] = 3*P(win) + 1*P(draw)
    """
    if is_home:
        return 3 * prediction.home_win_prob + 1 * prediction.draw_prob
    else:
        return 3 * prediction.away_win_prob + 1 * prediction.draw_prob


def position_to_theta(position: float) -> float:
    """
    Convert league position (1-20) to approximate theta scale.

    Position 1 (champion) ~ theta +1.0
    Position 10 (mid-table) ~ theta 0.0
    Position 20 (bottom) ~ theta -1.0
    """
    # Linear mapping: position 1 -> +1.0, position 20 -> -1.0
    return 1.0 - (position - 1) * (2.0 / 19.0)


def theta_to_position(theta: float) -> float:
    """Inverse of position_to_theta."""
    return 1.0 + (1.0 - theta) * (19.0 / 2.0)


class BayesianEngine:
    """
    The core prediction and update engine.

    Implements cumulative calibration updates:
    - Track expected vs actual points over the season
    - Only update theta when systematically off (|z| > threshold)
    - Scale updates by stickiness (stable teams move less)
    - Apply gravity pull toward historical position (decays over season)
    """

    def __init__(self, cfg=model_config):
        self.cfg = cfg

    def weekly_update(
        self,
        team: TeamState,
        week: int,
        match_predictions: List[Tuple[MatchPrediction, bool]],  # (prediction, is_home)
        match_results: List[Tuple[MatchResult, bool]],  # (result, is_home)
    ) -> Tuple[TeamState, UpdateExplanation]:
        """
        Update team state based on cumulative calibration.

        This is THE KEY ALGORITHM.

        We don't update theta for individual surprises.
        We update when the CUMULATIVE season performance
        is systematically above/below expectations.

        Args:
            team: Current team state
            week: Current matchweek number
            match_predictions: All predictions for this team this season
            match_results: All results for this team this season

        Returns:
            Updated team state and explanation of changes
        """
        cfg = self.cfg

        # 1. Calculate total expected points this season
        expected_pts = sum(
            calculate_expected_points(pred, is_home)
            for pred, is_home in match_predictions
        )

        # 2. Calculate actual points
        actual_pts = sum(
            result.home_points if is_home else result.away_points
            for result, is_home in match_results
        )

        n_matches = len(match_results)
        if n_matches == 0:
            return team, UpdateExplanation(
                team=team.team,
                week=week,
                actual_points=0,
                expected_points=0,
                z_score=0,
                update_triggered=False,
                reason="No matches played yet",
            )

        # 3. Calculate z-score: how many SEs from expectation?
        # SE grows with sqrt(n_matches)
        se = math.sqrt(n_matches * cfg.POINTS_VARIANCE_PER_MATCH)
        if se == 0:
            z_score = 0.0
        else:
            z_score = (actual_pts - expected_pts) / se

        # Update team tracking
        team.expected_points_season = expected_pts
        team.actual_points_season = actual_pts
        team.matches_played = n_matches
        team.cumulative_z_score = z_score

        # 4. Check if update is warranted
        if abs(z_score) < cfg.UPDATE_THRESHOLD:
            # Within expected variance - model is calibrated
            # But apply a small "drift" update proportional to z-score
            # This creates realistic week-to-week wobble
            old_theta = team.theta_home
            old_sigma = team.sigma

            # Small drift: 20% of what a full update would be, scaled by z-score
            drift_factor = 0.20
            drift = (
                z_score  # Direction and magnitude from z
                * drift_factor
                * cfg.LEARNING_RATE
                * (team.sigma / cfg.BASELINE_SIGMA)
                * (1.0 / team.stickiness)
            )

            team.theta_home += drift
            team.theta_away += drift
            team.sigma *= 0.98  # Tiny sigma shrink

            return team, UpdateExplanation(
                team=team.team,
                week=week,
                actual_points=actual_pts,
                expected_points=expected_pts,
                z_score=z_score,
                update_triggered=False,
                reason=f"Small drift (|z|={abs(z_score):.2f} < {cfg.UPDATE_THRESHOLD})",
                theta_change=drift,
                sigma_change=team.sigma - old_sigma,
                new_theta=team.theta_home,
                new_sigma=team.sigma,
            )

        # 5. Systematic over/under performance - update theta!
        excess_z = np.sign(z_score) * (abs(z_score) - cfg.UPDATE_THRESHOLD)

        # Update magnitude scales with:
        # - Excess Z (how far off)
        # - Sigma (more uncertain teams move more)
        # - Inverse stickiness (volatile teams move more)
        theta_change = (
            excess_z
            * cfg.LEARNING_RATE
            * (team.sigma / cfg.BASELINE_SIGMA)
            * (1.0 / team.stickiness)
        )

        old_theta = team.theta_home
        old_sigma = team.sigma

        team.theta_home += theta_change
        team.theta_away += theta_change
        team.sigma *= 0.95  # Shrink uncertainty (we learned something)

        # 6. Apply gravity pull (decays over season)
        gravity_strength = team.gravity_weight * math.exp(-week / cfg.GRAVITY_DECAY_WEEKS)
        gravity_theta = position_to_theta(team.gravity_mean)

        gravity_pull = gravity_strength * (gravity_theta - team.theta_home) * 0.05
        team.theta_home += gravity_pull
        team.theta_away += gravity_pull

        # Determine reason
        if z_score > 0:
            reason = f"Over-performing: {actual_pts} pts vs {expected_pts:.1f} expected (z={z_score:.2f})"
        else:
            reason = f"Under-performing: {actual_pts} pts vs {expected_pts:.1f} expected (z={z_score:.2f})"

        return team, UpdateExplanation(
            team=team.team,
            week=week,
            actual_points=actual_pts,
            expected_points=expected_pts,
            z_score=z_score,
            update_triggered=True,
            reason=reason,
            theta_change=theta_change,
            sigma_change=team.sigma - old_sigma,
            gravity_pull=gravity_pull,
            new_theta=team.theta_home,
            new_sigma=team.sigma,
        )

    def simulate_match(
        self,
        home_theta: float,
        away_theta: float,
        rng: np.random.Generator,
    ) -> Tuple[int, int]:
        """
        Simulate a single match outcome.

        Returns (home_points, away_points).
        """
        h_prob, d_prob, a_prob, _ = predict_match_probs(home_theta, away_theta)

        r = rng.random()
        if r < h_prob:
            return 3, 0  # Home win
        elif r < h_prob + d_prob:
            return 1, 1  # Draw
        else:
            return 0, 3  # Away win

    def simulate_remaining_season(
        self,
        team_states: Dict[str, TeamState],
        remaining_fixtures: List[Fixture],
        current_points: Dict[str, int],
        n_simulations: int = 10000,
        seed: Optional[int] = None,
    ) -> Dict[str, Dict]:
        """
        Monte Carlo simulation of remaining season.

        Returns per-team statistics:
        - Position distribution
        - Title/Top4/Relegation probabilities
        - Expected final points
        """
        rng = np.random.default_rng(seed)

        teams = list(team_states.keys())
        n_teams = len(teams)

        # Track outcomes across simulations
        position_counts = {team: np.zeros(n_teams) for team in teams}
        points_totals = {team: [] for team in teams}

        for sim in range(n_simulations):
            # Start with current points
            sim_points = current_points.copy()

            # Simulate remaining matches
            for fixture in remaining_fixtures:
                home_state = team_states[fixture.home_team]
                away_state = team_states[fixture.away_team]

                home_pts, away_pts = self.simulate_match(
                    home_state.effective_theta_home,
                    away_state.effective_theta_away,
                    rng,
                )

                sim_points[fixture.home_team] += home_pts
                sim_points[fixture.away_team] += away_pts

            # Determine final positions (handle ties by goal difference proxy - random for now)
            sorted_teams = sorted(
                teams,
                key=lambda t: (sim_points[t], rng.random()),  # Random tiebreaker
                reverse=True,
            )

            # Record positions and points
            for pos, team in enumerate(sorted_teams):
                position_counts[team][pos] += 1
                points_totals[team].append(sim_points[team])

        # Compile results
        results = {}
        for team in teams:
            pos_probs = position_counts[team] / n_simulations
            pts_array = np.array(points_totals[team])

            results[team] = {
                "position_probs": pos_probs.tolist(),
                "title_prob": pos_probs[0],
                "top4_prob": sum(pos_probs[:4]),
                "top6_prob": sum(pos_probs[:6]),
                "relegation_prob": sum(pos_probs[-3:]),
                "expected_position": sum((i + 1) * p for i, p in enumerate(pos_probs)),
                "position_std": np.sqrt(sum((i + 1 - sum((j + 1) * pos_probs[j] for j in range(n_teams)))**2 * p for i, p in enumerate(pos_probs))),
                "expected_points": pts_array.mean(),
                "points_std": pts_array.std(),
            }

        return results


def normalize_gpcm_thetas(theta_df, target_std: float = 0.5):
    """
    Normalize GPCM thetas to have mean=0 and target standard deviation.

    This ensures comparability across different years' GPCM outputs,
    which may have different scales depending on the teams in the dataset.

    Args:
        theta_df: DataFrame with Team and Theta columns
        target_std: Target standard deviation (default 0.5)

    Returns:
        Dict mapping team -> normalized theta
    """
    if theta_df is None or len(theta_df) == 0:
        return {}

    thetas = theta_df["Theta"].values
    mean = thetas.mean()
    std = thetas.std()

    if std < 0.01:  # Avoid division by very small std
        std = 1.0

    # Z-score normalize then scale to target std
    normalized = {}
    for _, row in theta_df.iterrows():
        z = (row["Theta"] - mean) / std
        normalized[row["Team"]] = z * target_std

    return normalized


def initialize_team_states(
    team_params_df,  # DataFrame with team parameters
    theta_df=None,   # Optional: initial theta values
) -> Dict[str, TeamState]:
    """
    Initialize team states from parameter data.

    Args:
        team_params_df: DataFrame with columns:
            Team, stickiness, initial_sigma, gravity_mean, n_seasons
        theta_df: Optional DataFrame with initial theta values

    Returns:
        Dictionary of team -> TeamState
    """
    states = {}

    # Normalize GPCM thetas for consistent scale across years
    normalized_thetas = normalize_gpcm_thetas(
        theta_df,
        target_std=model_config.THETA_TARGET_STD
    )

    for _, row in team_params_df.iterrows():
        team = row["Team"]
        n_seasons = row.get("n_seasons", 5)  # Default to 5 if not specified

        # Get normalized GPCM theta if available
        gpcm_theta = normalized_thetas.get(team)

        # Gravity-based theta (from historical position)
        gravity_theta = position_to_theta(row["gravity_mean"])

        # For promoted/recently-returned teams (n_seasons <= 2),
        # use the promoted team default theta instead of stale GPCM data
        # Their historical GPCM may be from years ago before relegation
        if n_seasons <= 2:
            # Promoted teams start at the "promoted team average" (~16th place)
            # This reflects where newly promoted teams typically finish
            theta_home = model_config.PROMOTED_TEAM_THETA
        elif gpcm_theta is not None:
            # Established teams: use GPCM directly
            theta_home = gpcm_theta
        else:
            # No GPCM data: use gravity
            theta_home = gravity_theta

        # Apply analyst adjustment
        analyst_adj = ANALYST_ADJUSTMENTS.get(team, 0.0)

        states[team] = TeamState(
            team=team,
            theta_home=theta_home,
            theta_away=theta_home - model_config.HOME_AWAY_OFFSET,
            sigma=row["initial_sigma"],
            stickiness=row["stickiness"],
            gravity_mean=row["gravity_mean"],
            gravity_weight=model_config.INITIAL_GRAVITY_WEIGHT,
            analyst_adj=analyst_adj,
        )

    return states
