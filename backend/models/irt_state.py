"""
MOTSON Bayesian IRT State Model

New state model based on the Bayesian IRT approach with:
- theta: ability to win matches
- b_home: difficulty to beat at home
- b_away: difficulty to beat away
- Standard errors for all parameters
- Bayesian blending of 5-year priors with current season estimates
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict
import json


@dataclass
class IRTTeamState:
    """
    Complete IRT state for a single team.

    Parameters:
    - theta: Ability to win matches (how good the team is at "answering" opponents)
    - b_home: Difficulty to beat at home (how hard it is for opponents to win here)
    - b_away: Difficulty to beat away (how hard it is for opponents to win against this team on road)

    All parameters are on an unbounded scale centered at 0 for an average EPL team.
    Typical range: [-1.5, +1.5]

    Each parameter has:
    - A 5-year prior (gravity) from historical data
    - A current season estimate from fresh weekly IRT
    - A blended posterior combining both
    - A standard error capturing uncertainty
    """

    team: str

    # === BLENDED POSTERIOR (what we use for predictions) ===
    theta: float  # Blended theta
    theta_se: float  # Posterior standard error
    b_home: float  # Blended b_home
    b_home_se: float  # Posterior standard error
    b_away: float  # Blended b_away
    b_away_se: float  # Posterior standard error

    # === 5-YEAR GRAVITY PRIORS ===
    theta_prior: float  # 5-year theta
    theta_prior_se: float  # Prior standard error
    b_home_prior: float  # 5-year b_home
    b_home_prior_se: float
    b_away_prior: float  # 5-year b_away
    b_away_prior_se: float

    # === CURRENT SEASON ESTIMATES ===
    theta_season: float = 0.0  # This season's theta estimate
    theta_season_se: float = 1.0  # Season estimate SE (large early season)
    b_home_season: float = 0.0
    b_home_season_se: float = 1.0
    b_away_season: float = 0.0
    b_away_season_se: float = 1.0

    # === BLENDING WEIGHTS ===
    gravity_weight: float = 0.5  # Weight on prior (always >= 0.5)
    momentum_weight: float = 0.5  # Weight on current season

    # === SEASON TRACKING ===
    matches_played: int = 0
    expected_points_season: float = 0.0
    actual_points_season: int = 0

    # === METADATA ===
    last_updated: Optional[datetime] = None
    is_promoted: bool = False  # True if team was promoted this season

    @property
    def points_per_game(self) -> float:
        """Actual points per game this season."""
        if self.matches_played == 0:
            return 0.0
        return self.actual_points_season / self.matches_played

    @property
    def expected_ppg(self) -> float:
        """Expected points per game this season."""
        if self.matches_played == 0:
            return 0.0
        return self.expected_points_season / self.matches_played

    @property
    def performance_vs_expected(self) -> float:
        """How much better/worse than expected (points)."""
        return self.actual_points_season - self.expected_points_season

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "team": self.team,
            # Blended posterior
            "theta": round(self.theta, 4),
            "theta_se": round(self.theta_se, 4),
            "b_home": round(self.b_home, 4),
            "b_home_se": round(self.b_home_se, 4),
            "b_away": round(self.b_away, 4),
            "b_away_se": round(self.b_away_se, 4),
            # 5-year priors
            "theta_prior": round(self.theta_prior, 4),
            "theta_prior_se": round(self.theta_prior_se, 4),
            "b_home_prior": round(self.b_home_prior, 4),
            "b_home_prior_se": round(self.b_home_prior_se, 4),
            "b_away_prior": round(self.b_away_prior, 4),
            "b_away_prior_se": round(self.b_away_prior_se, 4),
            # Current season
            "theta_season": round(self.theta_season, 4),
            "theta_season_se": round(self.theta_season_se, 4),
            "b_home_season": round(self.b_home_season, 4),
            "b_home_season_se": round(self.b_home_season_se, 4),
            "b_away_season": round(self.b_away_season, 4),
            "b_away_season_se": round(self.b_away_season_se, 4),
            # Weights
            "gravity_weight": round(self.gravity_weight, 4),
            "momentum_weight": round(self.momentum_weight, 4),
            # Season tracking
            "matches_played": self.matches_played,
            "expected_points_season": round(self.expected_points_season, 2),
            "actual_points_season": self.actual_points_season,
            "points_per_game": round(self.points_per_game, 2),
            "expected_ppg": round(self.expected_ppg, 2),
            "performance_vs_expected": round(self.performance_vs_expected, 2),
            # Metadata
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "is_promoted": self.is_promoted,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IRTTeamState":
        """Deserialize from dictionary."""
        return cls(
            team=data["team"],
            theta=data["theta"],
            theta_se=data["theta_se"],
            b_home=data["b_home"],
            b_home_se=data["b_home_se"],
            b_away=data["b_away"],
            b_away_se=data["b_away_se"],
            theta_prior=data.get("theta_prior", data["theta"]),
            theta_prior_se=data.get("theta_prior_se", 0.3),
            b_home_prior=data.get("b_home_prior", data["b_home"]),
            b_home_prior_se=data.get("b_home_prior_se", 0.3),
            b_away_prior=data.get("b_away_prior", data["b_away"]),
            b_away_prior_se=data.get("b_away_prior_se", 0.3),
            theta_season=data.get("theta_season", 0.0),
            theta_season_se=data.get("theta_season_se", 1.0),
            b_home_season=data.get("b_home_season", 0.0),
            b_home_season_se=data.get("b_home_season_se", 1.0),
            b_away_season=data.get("b_away_season", 0.0),
            b_away_season_se=data.get("b_away_season_se", 1.0),
            gravity_weight=data.get("gravity_weight", 0.5),
            momentum_weight=data.get("momentum_weight", 0.5),
            matches_played=data.get("matches_played", 0),
            expected_points_season=data.get("expected_points_season", 0.0),
            actual_points_season=data.get("actual_points_season", 0),
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else None,
            is_promoted=data.get("is_promoted", False),
        )


@dataclass
class IRTMatchPrediction:
    """
    Match prediction using the IRT model.

    Prediction is based on:
        gap = (theta_home - b_away_opponent) - (theta_away - b_home_opponent)

    Where positive gap favors the home team.
    """

    match_id: str
    matchweek: int
    home_team: str
    away_team: str

    # Probabilities from gap
    home_win_prob: float
    draw_prob: float
    away_win_prob: float

    # Gap components for interpretability
    gap: float  # Overall gap (positive = home favored)
    m_home: float  # Home attack margin = theta_home - b_away_opponent
    m_away: float  # Away attack margin = theta_away - b_home_opponent

    # Team parameters used
    home_theta: float
    home_b_home: float
    away_theta: float
    away_b_away: float

    # Confidence (based on combined SEs)
    confidence: float

    def to_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "matchweek": self.matchweek,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_win_prob": round(self.home_win_prob, 3),
            "draw_prob": round(self.draw_prob, 3),
            "away_win_prob": round(self.away_win_prob, 3),
            "gap": round(self.gap, 3),
            "m_home": round(self.m_home, 3),
            "m_away": round(self.m_away, 3),
            "home_theta": round(self.home_theta, 3),
            "home_b_home": round(self.home_b_home, 3),
            "away_theta": round(self.away_theta, 3),
            "away_b_away": round(self.away_b_away, 3),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class WeeklySnapshot:
    """
    Snapshot of all team states at a specific week.
    Used for tracking evolution over the season.
    """

    week: int
    timestamp: datetime
    team_states: Dict[str, dict]  # team -> IRTTeamState.to_dict()

    def to_dict(self) -> dict:
        return {
            "week": self.week,
            "timestamp": self.timestamp.isoformat(),
            "team_states": self.team_states,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WeeklySnapshot":
        return cls(
            week=data["week"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            team_states=data["team_states"],
        )
