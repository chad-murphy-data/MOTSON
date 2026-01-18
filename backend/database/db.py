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


# Global database instance
_db: Optional[Database] = None


def get_db() -> Database:
    """Get or create database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db
