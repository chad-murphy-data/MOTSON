"""
MOTSON v2 - Football Data API Integration

Fetches match results and fixtures from football-data.org API.
Free tier: 10 requests/minute.
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging

from ..models.team_state import MatchResult, Fixture
from ..config import api_config, app_config, TEAM_NAME_MAP

logger = logging.getLogger(__name__)


class FootballDataAPI:
    """
    Client for football-data.org API.

    Handles rate limiting and team name normalization.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or api_config.FOOTBALL_DATA_API_KEY
        self.base_url = api_config.FOOTBALL_DATA_BASE_URL
        self.last_request_time = datetime.min
        self.request_count = 0

        if not self.api_key:
            logger.warning("No FOOTBALL_DATA_API_KEY set - API calls will fail")

    def _normalize_team_name(self, api_name: str) -> str:
        """Convert API team name to our standard format."""
        # Direct mapping
        if api_name in TEAM_NAME_MAP:
            return TEAM_NAME_MAP[api_name]

        # Try removing common suffixes
        clean_name = api_name.replace(" FC", "").replace(" AFC", "").strip()

        # Check mapping again
        if clean_name in TEAM_NAME_MAP:
            return TEAM_NAME_MAP[clean_name]

        # Handle specific patterns
        if "Manchester United" in api_name:
            return "Manchester Utd"
        if "Manchester City" in api_name:
            return "Manchester City"
        if "Newcastle" in api_name:
            return "Newcastle Utd"
        if "Nottingham" in api_name or "Nott" in api_name:
            return "Nott'ham Forest"
        if "Wolverhampton" in api_name or "Wolves" in api_name:
            return "Wolves"
        if "Bournemouth" in api_name:
            return "Bournemouth"
        if "Brighton" in api_name:
            return "Brighton"
        if "West Ham" in api_name:
            return "West Ham"
        if "Tottenham" in api_name:
            return "Tottenham"
        if "Sunderland" in api_name:
            return "Sunderland"
        if "Leeds" in api_name:
            return "Leeds United"

        # Return as-is if no mapping found
        return clean_name

    async def _rate_limit(self):
        """Enforce rate limiting."""
        now = datetime.now()

        # Reset counter if window has passed
        if (now - self.last_request_time).seconds >= api_config.RATE_LIMIT_WINDOW:
            self.request_count = 0
            self.last_request_time = now

        # Wait if at limit
        if self.request_count >= api_config.RATE_LIMIT_REQUESTS:
            wait_time = api_config.RATE_LIMIT_WINDOW - (now - self.last_request_time).seconds
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time}s")
                await asyncio.sleep(wait_time)
                self.request_count = 0
                self.last_request_time = datetime.now()

        self.request_count += 1

    async def _get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make authenticated GET request."""
        await self._rate_limit()

        url = f"{self.base_url}{endpoint}"
        headers = {"X-Auth-Token": self.api_key}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 429:
                    # Rate limited - wait and retry
                    retry_after = int(response.headers.get("X-RequestCounter-Reset", 60))
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    return await self._get(endpoint, params)

                if response.status != 200:
                    text = await response.text()
                    logger.error(f"API error {response.status}: {text}")
                    raise Exception(f"API error {response.status}: {text}")

                return await response.json()

    async def get_matches(
        self,
        season: int = None,
        status: Optional[str] = None,
        matchday: Optional[int] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> List[Dict]:
        """
        Fetch matches from API.

        Args:
            season: Season year (e.g., 2025 for 2025-26)
            status: SCHEDULED, LIVE, IN_PLAY, PAUSED, FINISHED, POSTPONED, SUSPENDED, CANCELLED
            matchday: Specific matchweek
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
        """
        season = season or app_config.CURRENT_SEASON

        params = {"season": season}
        if status:
            params["status"] = status
        if matchday:
            params["matchday"] = matchday
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to

        data = await self._get(f"/competitions/{app_config.COMPETITION_CODE}/matches", params)
        return data.get("matches", [])

    async def get_finished_matches(self, season: int = None) -> List[MatchResult]:
        """Get all finished matches for the season."""
        matches = await self.get_matches(season=season, status="FINISHED")

        results = []
        for m in matches:
            try:
                result = MatchResult(
                    match_id=str(m["id"]),
                    matchweek=m.get("matchday", 0),
                    date=datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")),
                    home_team=self._normalize_team_name(m["homeTeam"]["name"]),
                    away_team=self._normalize_team_name(m["awayTeam"]["name"]),
                    home_goals=m["score"]["fullTime"]["home"],
                    away_goals=m["score"]["fullTime"]["away"],
                )
                results.append(result)
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping match {m.get('id')}: {e}")

        # Sort by matchweek, then date
        results.sort(key=lambda r: (r.matchweek, r.date))
        return results

    async def get_scheduled_fixtures(self, season: int = None) -> List[Fixture]:
        """Get all scheduled (upcoming) fixtures."""
        matches = await self.get_matches(season=season, status="SCHEDULED")

        fixtures = []
        for m in matches:
            try:
                fixture = Fixture(
                    match_id=str(m["id"]),
                    matchweek=m.get("matchday", 0),
                    date=datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")),
                    home_team=self._normalize_team_name(m["homeTeam"]["name"]),
                    away_team=self._normalize_team_name(m["awayTeam"]["name"]),
                    status="SCHEDULED",
                )
                fixtures.append(fixture)
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping fixture {m.get('id')}: {e}")

        fixtures.sort(key=lambda f: (f.matchweek, f.date))
        return fixtures

    async def get_all_fixtures(self, season: int = None) -> List[Fixture]:
        """Get all fixtures (scheduled and finished) for the season."""
        matches = await self.get_matches(season=season)

        fixtures = []
        for m in matches:
            try:
                status = m.get("status", "SCHEDULED")
                fixture = Fixture(
                    match_id=str(m["id"]),
                    matchweek=m.get("matchday", 0),
                    date=datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")),
                    home_team=self._normalize_team_name(m["homeTeam"]["name"]),
                    away_team=self._normalize_team_name(m["awayTeam"]["name"]),
                    status=status,
                    home_goals=m["score"]["fullTime"]["home"] if status == "FINISHED" else None,
                    away_goals=m["score"]["fullTime"]["away"] if status == "FINISHED" else None,
                )
                fixtures.append(fixture)
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping fixture {m.get('id')}: {e}")

        fixtures.sort(key=lambda f: (f.matchweek, f.date))
        return fixtures

    async def get_current_matchweek(self, season: int = None) -> int:
        """Determine current matchweek based on finished matches."""
        results = await self.get_finished_matches(season)
        if not results:
            return 1

        # Current matchweek = highest completed matchweek
        return max(r.matchweek for r in results)

    async def get_standings(self, season: int = None) -> List[Dict]:
        """Get current league standings."""
        season = season or app_config.CURRENT_SEASON

        data = await self._get(
            f"/competitions/{app_config.COMPETITION_CODE}/standings",
            {"season": season}
        )

        standings = []
        for s in data.get("standings", []):
            if s.get("type") == "TOTAL":
                for entry in s.get("table", []):
                    standings.append({
                        "position": entry["position"],
                        "team": self._normalize_team_name(entry["team"]["name"]),
                        "played": entry["playedGames"],
                        "won": entry["won"],
                        "drawn": entry["draw"],
                        "lost": entry["lost"],
                        "goals_for": entry["goalsFor"],
                        "goals_against": entry["goalsAgainst"],
                        "goal_difference": entry["goalDifference"],
                        "points": entry["points"],
                    })

        return standings

    async def get_teams(self, season: int = None) -> List[Dict]:
        """Get all teams in the competition with coach info."""
        season = season or app_config.CURRENT_SEASON

        data = await self._get(
            f"/competitions/{app_config.COMPETITION_CODE}/teams",
            {"season": season}
        )

        teams = []
        for t in data.get("teams", []):
            coach = t.get("coach", {})
            teams.append({
                "team": self._normalize_team_name(t["name"]),
                "team_id": t.get("id"),
                "coach_name": coach.get("name") if coach else None,
                "coach_id": coach.get("id") if coach else None,
                "coach_nationality": coach.get("nationality") if coach else None,
            })

        return teams


# Synchronous wrapper for non-async contexts
def fetch_finished_matches(api_key: str = None, season: int = None) -> List[MatchResult]:
    """Synchronous wrapper for get_finished_matches."""
    api = FootballDataAPI(api_key)
    return asyncio.run(api.get_finished_matches(season))


def fetch_scheduled_fixtures(api_key: str = None, season: int = None) -> List[Fixture]:
    """Synchronous wrapper for get_scheduled_fixtures."""
    api = FootballDataAPI(api_key)
    return asyncio.run(api.get_scheduled_fixtures(season))


def fetch_standings(api_key: str = None, season: int = None) -> List[Dict]:
    """Synchronous wrapper for get_standings."""
    api = FootballDataAPI(api_key)
    return asyncio.run(api.get_standings(season))


def fetch_teams(api_key: str = None, season: int = None) -> List[Dict]:
    """Synchronous wrapper for get_teams."""
    api = FootballDataAPI(api_key)
    return asyncio.run(api.get_teams(season))
