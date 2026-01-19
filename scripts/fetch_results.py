#!/usr/bin/env python3
"""
Fetch latest match results from football-data.org API and update local database.
This script is used by GitHub Actions before running simulations.
"""

import asyncio
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services.data_fetcher import FootballDataAPI
from backend.database.db import get_db


async def main():
    print("Fetching latest results from football-data.org API...")

    api = FootballDataAPI()
    db = get_db()

    # Fetch finished matches
    results = await api.get_finished_matches()
    print(f"Found {len(results)} finished matches")

    # Save to database
    db.save_match_results(results)
    print(f"Saved {len(results)} results to database")

    # Get current matchweek
    max_week = max(r.matchweek for r in results) if results else 0
    print(f"Current matchweek: {max_week}")

    # Also fetch and save fixtures
    fixtures = await api.get_all_fixtures()
    db.save_fixtures(fixtures)
    print(f"Saved {len(fixtures)} fixtures to database")

    return max_week


if __name__ == "__main__":
    week = asyncio.run(main())
    print(f"\nDone! Latest matchweek with results: {week}")
