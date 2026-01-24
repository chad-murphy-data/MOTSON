#!/usr/bin/env python3
"""
Fetch external betting odds and Opta predictions for evaluation.

This script fetches data from:
1. football-data.co.uk - Historical betting odds (free)
2. The Analyst (theanalyst.com) - Opta weekly predictions

Run this script with network access to populate the evaluation cache,
then run evaluate_predictions.py to compare MOTSON predictions.

Usage:
    python scripts/fetch_external_odds.py [--season YYYY]
"""

import argparse
import csv
import io
import json
import re
import ssl
import urllib.request
from datetime import datetime
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR = DATA_DIR / "evaluation_cache"


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_url(url: str, timeout: int = 30) -> str:
    """Fetch URL content with SSL handling."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as response:
        return response.read().decode('utf-8', errors='replace')


def fetch_football_data_odds(season: int = 2025):
    """
    Fetch betting odds from football-data.co.uk.

    Season format: 2025 for 2025-26 season (stored as 2526 in URL).
    """
    # Convert season to URL format (2025 -> 2526)
    season_code = f"{season % 100:02d}{(season + 1) % 100:02d}"
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"

    print(f"Fetching betting odds from {url}...")

    try:
        content = fetch_url(url)

        # Save to cache
        ensure_cache_dir()
        cache_file = CACHE_DIR / "betting_odds.csv"
        with open(cache_file, 'w') as f:
            f.write(content)

        # Count rows
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        print(f"✓ Downloaded {len(rows)} matches to {cache_file}")

        return cache_file

    except Exception as e:
        print(f"✗ Error fetching odds: {e}")
        print("  Try accessing the URL manually in a browser:")
        print(f"  {url}")
        return None


def fetch_opta_predictions():
    """
    Fetch Opta predictions from The Analyst.

    Note: This is fragile as it depends on the HTML structure.
    """
    url = "https://theanalyst.com/articles/premier-league-match-predictions"

    print(f"Fetching Opta predictions from {url}...")

    try:
        content = fetch_url(url)

        # Save raw HTML for debugging
        ensure_cache_dir()
        html_file = CACHE_DIR / "opta_raw.html"
        with open(html_file, 'w') as f:
            f.write(content)

        print(f"  Saved raw HTML to {html_file}")

        # Try to parse predictions (this is fragile)
        predictions = parse_opta_html(content)

        if predictions:
            # Save parsed predictions
            cache_file = CACHE_DIR / "opta_predictions.json"
            with open(cache_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            print(f"✓ Parsed {len(predictions)} predictions to {cache_file}")
            return cache_file
        else:
            print("  Warning: Could not parse any predictions from HTML")
            print("  The page structure may have changed.")
            print("  You can manually create opta_predictions.json")
            return None

    except Exception as e:
        print(f"✗ Error fetching Opta predictions: {e}")
        return None


def parse_opta_html(html: str) -> dict:
    """
    Attempt to parse Opta predictions from HTML.

    Returns dict in format: {"TeamA|TeamB": {prediction data}}
    """
    predictions = {}

    # Look for JSON data embedded in the page
    json_pattern = r'"predictions":\s*(\[.*?\])'
    match = re.search(json_pattern, html, re.DOTALL)

    if match:
        try:
            data = json.loads(match.group(1))
            for item in data:
                home = item.get('homeTeam', {}).get('name', '')
                away = item.get('awayTeam', {}).get('name', '')
                if home and away:
                    key = f"{home}|{away}"
                    predictions[key] = {
                        'home_team': home,
                        'away_team': away,
                        'home_win_prob': item.get('homeWinProb', 0) / 100,
                        'draw_prob': item.get('drawProb', 0) / 100,
                        'away_win_prob': item.get('awayWinProb', 0) / 100,
                        'source': 'opta',
                        'matchweek': item.get('matchweek', 0),
                        'match_id': '',
                    }
        except json.JSONDecodeError:
            pass

    # Alternative: look for table data
    if not predictions:
        # Pattern: "TeamA 45.2% 28.1% TeamB 26.7%"
        pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(\d+\.?\d*)[%]?\s+(\d+\.?\d*)[%]?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(\d+\.?\d*)[%]?'

        for match in re.finditer(pattern, html):
            try:
                home_team = match.group(1)
                home_prob = float(match.group(2))
                draw_prob = float(match.group(3))
                away_team = match.group(4)
                away_prob = float(match.group(5))

                # Normalize
                if home_prob > 1:
                    home_prob /= 100
                    draw_prob /= 100
                    away_prob /= 100

                total = home_prob + draw_prob + away_prob
                if 0.9 <= total <= 1.1:
                    home_prob /= total
                    draw_prob /= total
                    away_prob /= total

                    key = f"{home_team}|{away_team}"
                    predictions[key] = {
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_win_prob': home_prob,
                        'draw_prob': draw_prob,
                        'away_win_prob': away_prob,
                        'source': 'opta',
                        'matchweek': 0,
                        'match_id': '',
                    }
            except (ValueError, IndexError):
                continue

    return predictions


def create_sample_betting_csv():
    """Create a sample betting odds CSV template."""
    ensure_cache_dir()
    sample_file = CACHE_DIR / "betting_odds_sample.csv"

    with open(sample_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
            'B365H', 'B365D', 'B365A', 'Notes'
        ])
        writer.writerow([
            '2025-08-16', 'Arsenal', 'Liverpool', '2', '1', 'H',
            '2.50', '3.40', '2.80', 'Example row - replace with real data'
        ])

    print(f"\nCreated sample CSV template: {sample_file}")
    print("Fill in with data from football-data.co.uk or other sources.")


def create_sample_opta_json():
    """Create a sample Opta predictions JSON template."""
    ensure_cache_dir()
    sample_file = CACHE_DIR / "opta_predictions_sample.json"

    sample_data = {
        "Arsenal|Liverpool": {
            "home_team": "Arsenal",
            "away_team": "Liverpool",
            "home_win_prob": 0.42,
            "draw_prob": 0.28,
            "away_win_prob": 0.30,
            "source": "opta",
            "matchweek": 1,
            "match_id": ""
        },
        "Note": "Replace with actual Opta predictions from theanalyst.com"
    }

    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)

    print(f"\nCreated sample JSON template: {sample_file}")
    print("Fill in with data from theanalyst.com/articles/premier-league-match-predictions")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch external betting odds and Opta predictions"
    )
    parser.add_argument(
        '--season', type=int, default=2025,
        help="Season year (e.g., 2025 for 2025-26)"
    )
    parser.add_argument(
        '--create-samples', action='store_true',
        help="Create sample CSV/JSON templates"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("MOTSON External Data Fetcher")
    print("=" * 50)
    print()

    if args.create_samples:
        create_sample_betting_csv()
        create_sample_opta_json()
        return

    # Try to fetch betting odds
    betting_file = fetch_football_data_odds(args.season)
    print()

    # Try to fetch Opta predictions
    opta_file = fetch_opta_predictions()
    print()

    # Summary
    print("=" * 50)
    print("Summary")
    print("=" * 50)

    if betting_file:
        print(f"✓ Betting odds: {betting_file}")
    else:
        print("✗ Betting odds: Not fetched")
        print("  Run: python scripts/fetch_external_odds.py --create-samples")
        print("  Then manually download from football-data.co.uk")

    if opta_file:
        print(f"✓ Opta predictions: {opta_file}")
    else:
        print("✗ Opta predictions: Not fetched")
        print("  Run: python scripts/fetch_external_odds.py --create-samples")
        print("  Then manually populate from theanalyst.com")

    print()
    print("Next step: python scripts/evaluate_predictions.py")


if __name__ == "__main__":
    main()
