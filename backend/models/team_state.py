"""
MOTSON v2 - Team State Model

Tracks each team's latent strength (theta) with uncertainty (sigma),
plus historical context (stickiness, gravity) for Bayesian-ish updates.

Key insight: "Track distributions, not point estimates."
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
import json


@dataclass
class TeamState:
    """
    Complete state for a single team.

    Theta represents latent "strength" on an unbounded scale where:
    - 0 is roughly average EPL team
    - +1 is elite (Man City, Liverpool)
    - -1 is relegation quality

    Sigma represents our uncertainty about true theta.
    Tight sigma (0.15) = very confident (Man City always top 3)
    Wide sigma (0.40) = uncertain (promoted teams, volatile squads)
    """

    team: str

    # Core strength parameters
    theta_home: float           # Home strength (latent ability)
    theta_away: float           # Away strength (typically theta_home - 0.15)
    sigma: float                # Current uncertainty about true theta

    # Historical context
    stickiness: float           # Historical stability (0.3-0.97, higher = harder to move)
    gravity_mean: float         # Historical "belongs here" position (1-20 scale)
    gravity_weight: float       # Current gravity pull strength (decays over season)

    # Analyst override
    analyst_adj: float = 0.0    # Manual thumb-on-scale adjustment

    # Season tracking
    expected_points_season: float = 0.0
    actual_points_season: int = 0
    matches_played: int = 0
    cumulative_z_score: float = 0.0

    # Metadata
    last_updated: Optional[datetime] = None
    update_history: List[dict] = field(default_factory=list)

    @property
    def effective_theta_home(self) -> float:
        """Theta including analyst adjustment."""
        return self.theta_home + self.analyst_adj

    @property
    def effective_theta_away(self) -> float:
        """Away theta including analyst adjustment."""
        return self.theta_away + self.analyst_adj

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

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "team": self.team,
            "theta_home": round(self.theta_home, 4),
            "theta_away": round(self.theta_away, 4),
            "effective_theta_home": round(self.effective_theta_home, 4),
            "effective_theta_away": round(self.effective_theta_away, 4),
            "sigma": round(self.sigma, 4),
            "stickiness": round(self.stickiness, 3),
            "gravity_mean": round(self.gravity_mean, 1),
            "gravity_weight": round(self.gravity_weight, 3),
            "analyst_adj": round(self.analyst_adj, 3),
            "expected_points_season": round(self.expected_points_season, 2),
            "actual_points_season": self.actual_points_season,
            "matches_played": self.matches_played,
            "cumulative_z_score": round(self.cumulative_z_score, 3),
            "points_per_game": round(self.points_per_game, 2),
            "expected_ppg": round(self.expected_ppg, 2),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TeamState":
        """Deserialize from dictionary."""
        return cls(
            team=data["team"],
            theta_home=data["theta_home"],
            theta_away=data["theta_away"],
            sigma=data["sigma"],
            stickiness=data["stickiness"],
            gravity_mean=data["gravity_mean"],
            gravity_weight=data["gravity_weight"],
            analyst_adj=data.get("analyst_adj", 0.0),
            expected_points_season=data.get("expected_points_season", 0.0),
            actual_points_season=data.get("actual_points_season", 0),
            matches_played=data.get("matches_played", 0),
            cumulative_z_score=data.get("cumulative_z_score", 0.0),
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else None,
            update_history=data.get("update_history", []),
        )


@dataclass
class MatchResult:
    """A completed match result."""

    match_id: str
    matchweek: int
    date: datetime
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int

    @property
    def result(self) -> str:
        """H (home win), D (draw), or A (away win)."""
        if self.home_goals > self.away_goals:
            return "H"
        elif self.home_goals < self.away_goals:
            return "A"
        return "D"

    @property
    def home_points(self) -> int:
        """Points earned by home team."""
        if self.result == "H":
            return 3
        elif self.result == "D":
            return 1
        return 0

    @property
    def away_points(self) -> int:
        """Points earned by away team."""
        if self.result == "A":
            return 3
        elif self.result == "D":
            return 1
        return 0

    def to_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "matchweek": self.matchweek,
            "date": self.date.isoformat(),
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
            "result": self.result,
        }


@dataclass
class Fixture:
    """An upcoming (or completed) fixture."""

    match_id: str
    matchweek: int
    date: datetime
    home_team: str
    away_team: str
    status: str = "SCHEDULED"  # SCHEDULED, LIVE, FINISHED

    # Filled in after match
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None

    def to_match_result(self) -> Optional[MatchResult]:
        """Convert to MatchResult if finished."""
        if self.status != "FINISHED" or self.home_goals is None:
            return None
        return MatchResult(
            match_id=self.match_id,
            matchweek=self.matchweek,
            date=self.date,
            home_team=self.home_team,
            away_team=self.away_team,
            home_goals=self.home_goals,
            away_goals=self.away_goals,
        )

    def to_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "matchweek": self.matchweek,
            "date": self.date.isoformat(),
            "home_team": self.home_team,
            "away_team": self.away_team,
            "status": self.status,
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
        }


@dataclass
class MatchPrediction:
    """Prediction for an upcoming match."""

    match_id: str
    matchweek: int
    home_team: str
    away_team: str

    # Probabilities
    home_win_prob: float
    draw_prob: float
    away_win_prob: float

    # Confidence based on team sigmas
    confidence: float  # 0-1, higher = more confident

    # Context
    home_theta: float
    away_theta: float
    delta: float  # theta difference including home advantage

    def to_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "matchweek": self.matchweek,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_win_prob": round(self.home_win_prob, 3),
            "draw_prob": round(self.draw_prob, 3),
            "away_win_prob": round(self.away_win_prob, 3),
            "confidence": round(self.confidence, 2),
            "home_theta": round(self.home_theta, 3),
            "away_theta": round(self.away_theta, 3),
            "delta": round(self.delta, 3),
        }


@dataclass
class SeasonPrediction:
    """Season outcome probabilities for a team."""

    team: str

    # Position probabilities (1-20)
    position_probs: List[float]  # position_probs[0] = P(finish 1st)

    # Key outcome probabilities
    title_prob: float            # P(1st)
    top4_prob: float             # P(1st-4th) - Champions League
    top6_prob: float             # P(1st-6th) - Europa
    relegation_prob: float       # P(18th-20th)

    # Expected finish
    expected_position: float
    position_std: float

    # Points projection
    expected_points: float
    points_std: float

    def to_dict(self) -> dict:
        return {
            "team": self.team,
            "position_probs": [round(p, 4) for p in self.position_probs],
            "title_prob": round(self.title_prob, 4),
            "top4_prob": round(self.top4_prob, 4),
            "top6_prob": round(self.top6_prob, 4),
            "relegation_prob": round(self.relegation_prob, 4),
            "expected_position": round(self.expected_position, 2),
            "position_std": round(self.position_std, 2),
            "expected_points": round(self.expected_points, 1),
            "points_std": round(self.points_std, 1),
        }


@dataclass
class UpdateExplanation:
    """
    Explains WHY a team's theta changed (or didn't).

    This is key for transparency - users should understand
    why predictions changed.
    """

    team: str
    week: int

    # What happened
    actual_points: int
    expected_points: float
    z_score: float

    # Decision
    update_triggered: bool
    reason: str

    # Changes (if update triggered)
    theta_change: float = 0.0
    sigma_change: float = 0.0
    gravity_pull: float = 0.0

    # New values
    new_theta: float = 0.0
    new_sigma: float = 0.0

    def to_dict(self) -> dict:
        return {
            "team": self.team,
            "week": self.week,
            "actual_points": self.actual_points,
            "expected_points": round(self.expected_points, 2),
            "z_score": round(self.z_score, 3),
            "update_triggered": self.update_triggered,
            "reason": self.reason,
            "theta_change": round(self.theta_change, 4),
            "sigma_change": round(self.sigma_change, 4),
            "gravity_pull": round(self.gravity_pull, 4),
            "new_theta": round(self.new_theta, 4),
            "new_sigma": round(self.new_sigma, 4),
        }
