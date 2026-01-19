"""
MOTSON v2 - Database Layer

SQLite-based persistence for:
- Team states (current and historical)
- Match results and fixtures
- Predictions history
- Update explanations
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import logging

from ..models.team_state import TeamState, MatchResult, Fixture, SeasonPrediction
from ..models.irt_state import IRTTeamState
from ..config import app_config

logger = logging.getLogger(__name__)


class Database:
    """SQLite database for MOTSON data persistence."""

    def __init__(self, db_path: str = "motson.db"):
        self.db_path = db_path
        self._ensure_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self):
        """Create tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Team states table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_states (
                team TEXT PRIMARY KEY,
                theta_home REAL,
                theta_away REAL,
                sigma REAL,
                stickiness REAL,
                gravity_mean REAL,
                gravity_weight REAL,
                analyst_adj REAL,
                expected_points_season REAL,
                actual_points_season INTEGER,
                matches_played INTEGER,
                cumulative_z_score REAL,
                last_updated TEXT,
                update_history TEXT
            )
        """)

        # Team state history (for trajectory visualization)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_state_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team TEXT,
                week INTEGER,
                theta_home REAL,
                theta_away REAL,
                sigma REAL,
                expected_points REAL,
                actual_points INTEGER,
                z_score REAL,
                timestamp TEXT,
                UNIQUE(team, week)
            )
        """)

        # Match results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS match_results (
                match_id TEXT PRIMARY KEY,
                matchweek INTEGER,
                date TEXT,
                home_team TEXT,
                away_team TEXT,
                home_goals INTEGER,
                away_goals INTEGER
            )
        """)

        # Fixtures (scheduled matches)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fixtures (
                match_id TEXT PRIMARY KEY,
                matchweek INTEGER,
                date TEXT,
                home_team TEXT,
                away_team TEXT,
                status TEXT
            )
        """)

        # Match predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS match_predictions (
                match_id TEXT PRIMARY KEY,
                matchweek INTEGER,
                home_team TEXT,
                away_team TEXT,
                home_win_prob REAL,
                draw_prob REAL,
                away_win_prob REAL,
                confidence REAL,
                predicted_at TEXT
            )
        """)

        # Season predictions (stored per week)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS season_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team TEXT,
                week INTEGER,
                position_probs TEXT,
                title_prob REAL,
                top4_prob REAL,
                top6_prob REAL,
                relegation_prob REAL,
                expected_position REAL,
                position_std REAL,
                expected_points REAL,
                points_std REAL,
                timestamp TEXT,
                UNIQUE(team, week)
            )
        """)

        # Update explanations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS update_explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team TEXT,
                week INTEGER,
                actual_points INTEGER,
                expected_points REAL,
                z_score REAL,
                update_triggered INTEGER,
                reason TEXT,
                theta_change REAL,
                sigma_change REAL,
                gravity_pull REAL,
                timestamp TEXT,
                UNIQUE(team, week)
            )
        """)

        # Metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """)

        # IRT Team States (new Bayesian IRT model)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS irt_team_states (
                team TEXT PRIMARY KEY,
                theta REAL,
                theta_se REAL,
                b_home REAL,
                b_home_se REAL,
                b_away REAL,
                b_away_se REAL,
                theta_prior REAL,
                theta_prior_se REAL,
                b_home_prior REAL,
                b_home_prior_se REAL,
                b_away_prior REAL,
                b_away_prior_se REAL,
                theta_season REAL,
                theta_season_se REAL,
                b_home_season REAL,
                b_home_season_se REAL,
                b_away_season REAL,
                b_away_season_se REAL,
                gravity_weight REAL,
                momentum_weight REAL,
                matches_played INTEGER,
                expected_points_season REAL,
                actual_points_season INTEGER,
                last_updated TEXT,
                is_promoted INTEGER
            )
        """)

        # IRT Team State History (for trajectory visualization)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS irt_team_state_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team TEXT,
                week INTEGER,
                theta REAL,
                theta_se REAL,
                b_home REAL,
                b_away REAL,
                theta_season REAL,
                theta_season_se REAL,
                gravity_weight REAL,
                expected_points REAL,
                actual_points INTEGER,
                timestamp TEXT,
                UNIQUE(team, week)
            )
        """)

        conn.commit()
        conn.close()

    # Team States

    def save_team_state(self, state: TeamState):
        """Save a team state."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO team_states
            (team, theta_home, theta_away, sigma, stickiness, gravity_mean,
             gravity_weight, analyst_adj, expected_points_season, actual_points_season,
             matches_played, cumulative_z_score, last_updated, update_history)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            state.team,
            state.theta_home,
            state.theta_away,
            state.sigma,
            state.stickiness,
            state.gravity_mean,
            state.gravity_weight,
            state.analyst_adj,
            state.expected_points_season,
            state.actual_points_season,
            state.matches_played,
            state.cumulative_z_score,
            state.last_updated.isoformat() if state.last_updated else None,
            json.dumps(state.update_history),
        ))

        conn.commit()
        conn.close()

    def save_team_states(self, states: Dict[str, TeamState]):
        """Save multiple team states."""
        for state in states.values():
            self.save_team_state(state)

    def get_team_state(self, team: str) -> Optional[TeamState]:
        """Get a team state."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM team_states WHERE team = ?", (team,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return TeamState(
            team=row["team"],
            theta_home=row["theta_home"],
            theta_away=row["theta_away"],
            sigma=row["sigma"],
            stickiness=row["stickiness"],
            gravity_mean=row["gravity_mean"],
            gravity_weight=row["gravity_weight"],
            analyst_adj=row["analyst_adj"],
            expected_points_season=row["expected_points_season"],
            actual_points_season=row["actual_points_season"],
            matches_played=row["matches_played"],
            cumulative_z_score=row["cumulative_z_score"],
            last_updated=datetime.fromisoformat(row["last_updated"]) if row["last_updated"] else None,
            update_history=json.loads(row["update_history"]) if row["update_history"] else [],
        )

    def get_all_team_states(self) -> Dict[str, TeamState]:
        """Get all team states."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT team FROM team_states")
        teams = [row["team"] for row in cursor.fetchall()]
        conn.close()

        return {team: self.get_team_state(team) for team in teams}

    def save_team_state_history(self, state: TeamState, week: int):
        """Save team state snapshot for history tracking."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO team_state_history
            (team, week, theta_home, theta_away, sigma, expected_points,
             actual_points, z_score, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            state.team,
            week,
            state.theta_home,
            state.theta_away,
            state.sigma,
            state.expected_points_season,
            state.actual_points_season,
            state.cumulative_z_score,
            datetime.utcnow().isoformat(),
        ))

        conn.commit()
        conn.close()

    def get_team_history(self, team: str) -> List[Dict]:
        """Get historical state snapshots for a team."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM team_state_history
            WHERE team = ?
            ORDER BY week
        """, (team,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_all_teams_history(self) -> List[Dict]:
        """Get historical state snapshots for all teams."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM team_state_history
            ORDER BY week, team
        """)

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_all_season_predictions_history(self) -> List[Dict]:
        """Get all historical season predictions for all teams and weeks."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM season_predictions
            ORDER BY week, team
        """)

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "team": row["team"],
                "week": row["week"],
                "position_probs": json.loads(row["position_probs"]),
                "title_prob": row["title_prob"],
                "top4_prob": row["top4_prob"],
                "top6_prob": row["top6_prob"],
                "relegation_prob": row["relegation_prob"],
                "expected_position": row["expected_position"],
                "position_std": row["position_std"],
                "expected_points": row["expected_points"],
                "points_std": row["points_std"],
            }
            for row in rows
        ]

    # Match Results

    def save_match_result(self, result: MatchResult):
        """Save a match result."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO match_results
            (match_id, matchweek, date, home_team, away_team, home_goals, away_goals)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            result.match_id,
            result.matchweek,
            result.date.isoformat(),
            result.home_team,
            result.away_team,
            result.home_goals,
            result.away_goals,
        ))

        conn.commit()
        conn.close()

    def save_match_results(self, results: List[MatchResult]):
        """Save multiple match results."""
        for result in results:
            self.save_match_result(result)

    def get_match_results(self, matchweek: Optional[int] = None) -> List[MatchResult]:
        """Get match results, optionally filtered by matchweek."""
        conn = self._get_connection()
        cursor = conn.cursor()

        if matchweek:
            cursor.execute("SELECT * FROM match_results WHERE matchweek = ?", (matchweek,))
        else:
            cursor.execute("SELECT * FROM match_results ORDER BY matchweek, date")

        rows = cursor.fetchall()
        conn.close()

        return [
            MatchResult(
                match_id=row["match_id"],
                matchweek=row["matchweek"],
                date=datetime.fromisoformat(row["date"]),
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_goals=row["home_goals"],
                away_goals=row["away_goals"],
            )
            for row in rows
        ]

    # Season Predictions

    def save_season_prediction(self, prediction: SeasonPrediction, week: int):
        """Save a season prediction."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO season_predictions
            (team, week, position_probs, title_prob, top4_prob, top6_prob,
             relegation_prob, expected_position, position_std, expected_points,
             points_std, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction.team,
            week,
            json.dumps(prediction.position_probs),
            prediction.title_prob,
            prediction.top4_prob,
            prediction.top6_prob,
            prediction.relegation_prob,
            prediction.expected_position,
            prediction.position_std,
            prediction.expected_points,
            prediction.points_std,
            datetime.utcnow().isoformat(),
        ))

        conn.commit()
        conn.close()

    def save_season_predictions(self, predictions: List[SeasonPrediction], week: int):
        """Save multiple season predictions."""
        for pred in predictions:
            self.save_season_prediction(pred, week)

    def get_season_predictions(self, week: Optional[int] = None) -> List[Dict]:
        """Get season predictions, optionally filtered by week."""
        conn = self._get_connection()
        cursor = conn.cursor()

        if week:
            cursor.execute("""
                SELECT * FROM season_predictions WHERE week = ?
            """, (week,))
        else:
            # Get latest predictions
            cursor.execute("""
                SELECT * FROM season_predictions sp
                WHERE week = (SELECT MAX(week) FROM season_predictions)
            """)

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "team": row["team"],
                "week": row["week"],
                "position_probs": json.loads(row["position_probs"]),
                "title_prob": row["title_prob"],
                "top4_prob": row["top4_prob"],
                "top6_prob": row["top6_prob"],
                "relegation_prob": row["relegation_prob"],
                "expected_position": row["expected_position"],
                "position_std": row["position_std"],
                "expected_points": row["expected_points"],
                "points_std": row["points_std"],
            }
            for row in rows
        ]

    # Fixtures

    def save_fixture(self, fixture):
        """Save a fixture."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO fixtures
            (match_id, matchweek, date, home_team, away_team, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            fixture.match_id,
            fixture.matchweek,
            fixture.date.isoformat() if hasattr(fixture.date, 'isoformat') else str(fixture.date),
            fixture.home_team,
            fixture.away_team,
            fixture.status,
        ))

        conn.commit()
        conn.close()

    def save_fixtures(self, fixtures: List):
        """Save multiple fixtures."""
        for fixture in fixtures:
            self.save_fixture(fixture)

    def get_fixtures(self, matchweek: Optional[int] = None) -> List[Dict]:
        """Get fixtures, optionally filtered by matchweek."""
        conn = self._get_connection()
        cursor = conn.cursor()

        if matchweek:
            cursor.execute("SELECT * FROM fixtures WHERE matchweek = ? ORDER BY date", (matchweek,))
        else:
            cursor.execute("SELECT * FROM fixtures ORDER BY matchweek, date")

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    # Match Predictions

    def save_match_prediction(self, prediction: Dict):
        """Save a match prediction."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO match_predictions
            (match_id, matchweek, home_team, away_team, home_win_prob,
             draw_prob, away_win_prob, confidence, predicted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction["match_id"],
            prediction["matchweek"],
            prediction["home_team"],
            prediction["away_team"],
            prediction["home_win_prob"],
            prediction["draw_prob"],
            prediction["away_win_prob"],
            prediction["confidence"],
            datetime.utcnow().isoformat(),
        ))

        conn.commit()
        conn.close()

    def save_match_predictions(self, predictions: List[Dict]):
        """Save multiple match predictions."""
        for pred in predictions:
            self.save_match_prediction(pred)

    def get_match_predictions(self, matchweek: Optional[int] = None) -> List[Dict]:
        """Get match predictions, optionally filtered by matchweek."""
        conn = self._get_connection()
        cursor = conn.cursor()

        if matchweek:
            cursor.execute("SELECT * FROM match_predictions WHERE matchweek = ? ORDER BY match_id", (matchweek,))
        else:
            cursor.execute("SELECT * FROM match_predictions ORDER BY matchweek, match_id")

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    # Standings Cache

    def save_standings(self, standings: List[Dict]):
        """Save current standings to metadata as JSON."""
        self.set_metadata("cached_standings", json.dumps(standings))

    def get_cached_standings(self) -> Optional[List[Dict]]:
        """Get cached standings from metadata."""
        data = self.get_metadata("cached_standings")
        if data:
            return json.loads(data)
        return None

    # Metadata

    def set_metadata(self, key: str, value: str):
        """Set a metadata value."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO metadata (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (key, value, datetime.utcnow().isoformat()))

        conn.commit()
        conn.close()

    def get_metadata(self, key: str) -> Optional[str]:
        """Get a metadata value."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()

        return row["value"] if row else None

    def get_current_week(self) -> int:
        """Get the current matchweek from metadata."""
        week = self.get_metadata("current_week")
        return int(week) if week else 0

    # IRT Team States (new Bayesian IRT model)

    def save_irt_team_state(self, state: IRTTeamState):
        """Save an IRT team state."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO irt_team_states
            (team, theta, theta_se, b_home, b_home_se, b_away, b_away_se,
             theta_prior, theta_prior_se, b_home_prior, b_home_prior_se,
             b_away_prior, b_away_prior_se, theta_season, theta_season_se,
             b_home_season, b_home_season_se, b_away_season, b_away_season_se,
             gravity_weight, momentum_weight, matches_played, expected_points_season,
             actual_points_season, last_updated, is_promoted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            state.team,
            state.theta, state.theta_se,
            state.b_home, state.b_home_se,
            state.b_away, state.b_away_se,
            state.theta_prior, state.theta_prior_se,
            state.b_home_prior, state.b_home_prior_se,
            state.b_away_prior, state.b_away_prior_se,
            state.theta_season, state.theta_season_se,
            state.b_home_season, state.b_home_season_se,
            state.b_away_season, state.b_away_season_se,
            state.gravity_weight, state.momentum_weight,
            state.matches_played, state.expected_points_season,
            state.actual_points_season,
            state.last_updated.isoformat() if state.last_updated else None,
            1 if state.is_promoted else 0,
        ))

        conn.commit()
        conn.close()

    def save_irt_team_states(self, states: Dict[str, IRTTeamState]):
        """Save multiple IRT team states."""
        for state in states.values():
            self.save_irt_team_state(state)

    def get_irt_team_state(self, team: str) -> Optional[IRTTeamState]:
        """Get an IRT team state."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM irt_team_states WHERE team = ?", (team,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return IRTTeamState(
            team=row["team"],
            theta=row["theta"],
            theta_se=row["theta_se"],
            b_home=row["b_home"],
            b_home_se=row["b_home_se"],
            b_away=row["b_away"],
            b_away_se=row["b_away_se"],
            theta_prior=row["theta_prior"],
            theta_prior_se=row["theta_prior_se"],
            b_home_prior=row["b_home_prior"],
            b_home_prior_se=row["b_home_prior_se"],
            b_away_prior=row["b_away_prior"],
            b_away_prior_se=row["b_away_prior_se"],
            theta_season=row["theta_season"],
            theta_season_se=row["theta_season_se"],
            b_home_season=row["b_home_season"],
            b_home_season_se=row["b_home_season_se"],
            b_away_season=row["b_away_season"],
            b_away_season_se=row["b_away_season_se"],
            gravity_weight=row["gravity_weight"],
            momentum_weight=row["momentum_weight"],
            matches_played=row["matches_played"],
            expected_points_season=row["expected_points_season"],
            actual_points_season=row["actual_points_season"],
            last_updated=datetime.fromisoformat(row["last_updated"]) if row["last_updated"] else None,
            is_promoted=bool(row["is_promoted"]),
        )

    def get_all_irt_team_states(self) -> Dict[str, IRTTeamState]:
        """Get all IRT team states."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT team FROM irt_team_states")
        teams = [row["team"] for row in cursor.fetchall()]
        conn.close()

        return {team: self.get_irt_team_state(team) for team in teams}

    def save_irt_team_state_history(self, state: IRTTeamState, week: int):
        """Save IRT team state snapshot for history tracking."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO irt_team_state_history
            (team, week, theta, theta_se, b_home, b_away, theta_season, theta_season_se,
             gravity_weight, expected_points, actual_points, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            state.team,
            week,
            state.theta,
            state.theta_se,
            state.b_home,
            state.b_away,
            state.theta_season,
            state.theta_season_se,
            state.gravity_weight,
            state.expected_points_season,
            state.actual_points_season,
            datetime.utcnow().isoformat(),
        ))

        conn.commit()
        conn.close()

    def get_irt_team_history(self, team: str) -> List[Dict]:
        """Get historical IRT state snapshots for a team."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM irt_team_state_history
            WHERE team = ?
            ORDER BY week
        """, (team,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_all_irt_teams_history(self) -> List[Dict]:
        """Get historical IRT state snapshots for all teams."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM irt_team_state_history
            ORDER BY week, team
        """)

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]


# Global database instance
_db: Optional[Database] = None

# Default database path - use data/motson.db relative to project root
_DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "motson.db"


def get_db() -> Database:
    """Get or create database instance."""
    global _db
    if _db is None:
        _db = Database(str(_DEFAULT_DB_PATH))
    return _db
