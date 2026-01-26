#!/usr/bin/env python3
"""
Sync actual points from match_results to team state tables.

This script updates:
1. irt_team_states.actual_points_season and matches_played
2. team_state_history.actual_points for the current week
3. irt_team_state_history.actual_points for the current week
4. cached_standings in metadata table

Run this after fetching new match results to keep all tables in sync.
"""

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def main():
    project_root = Path(__file__).parent.parent
    db_path = project_root / 'data' / 'motson.db'

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Calculate full standings from match_results
    cursor.execute('SELECT home_team, away_team, home_goals, away_goals, matchweek FROM match_results')
    stats = defaultdict(lambda: {
        'played': 0, 'won': 0, 'drawn': 0, 'lost': 0,
        'goals_for': 0, 'goals_against': 0, 'points': 0
    })
    max_week = 0

    for home, away, hg, ag, week in cursor.fetchall():
        max_week = max(max_week, week)

        # Home team stats
        stats[home]['played'] += 1
        stats[home]['goals_for'] += hg
        stats[home]['goals_against'] += ag
        if hg > ag:
            stats[home]['won'] += 1
            stats[home]['points'] += 3
        elif hg < ag:
            stats[home]['lost'] += 1
        else:
            stats[home]['drawn'] += 1
            stats[home]['points'] += 1

        # Away team stats
        stats[away]['played'] += 1
        stats[away]['goals_for'] += ag
        stats[away]['goals_against'] += hg
        if ag > hg:
            stats[away]['won'] += 1
            stats[away]['points'] += 3
        elif ag < hg:
            stats[away]['lost'] += 1
        else:
            stats[away]['drawn'] += 1
            stats[away]['points'] += 1

    # Extract points and matches for state updates
    points = {team: s['points'] for team, s in stats.items()}
    matches = {team: s['played'] for team, s in stats.items()}

    print(f"Calculated standings from {sum(matches.values()) // 2} matches (Week {max_week})")

    # Update irt_team_states
    timestamp = datetime.now(timezone.utc).isoformat()
    updated = 0
    for team, pts in points.items():
        cursor.execute('''
            UPDATE irt_team_states
            SET actual_points_season = ?, matches_played = ?, last_updated = ?
            WHERE team = ?
        ''', (pts, matches[team], timestamp, team))
        if cursor.rowcount > 0:
            updated += 1
    print(f"Updated {updated} teams in irt_team_states")

    # Update team_state_history for current week
    updated = 0
    for team, pts in points.items():
        cursor.execute('''
            UPDATE team_state_history
            SET actual_points = ?, timestamp = ?
            WHERE team = ? AND week = ?
        ''', (pts, timestamp, team, max_week))
        if cursor.rowcount > 0:
            updated += 1
    print(f"Updated {updated} teams in team_state_history (week {max_week})")

    # Update irt_team_state_history for current week
    updated = 0
    for team, pts in points.items():
        cursor.execute('''
            UPDATE irt_team_state_history
            SET actual_points = ?
            WHERE team = ? AND week = ?
        ''', (pts, team, max_week))
        if cursor.rowcount > 0:
            updated += 1
    print(f"Updated {updated} teams in irt_team_state_history (week {max_week})")

    # Update cached_standings in metadata
    sorted_teams = sorted(
        stats.items(),
        key=lambda x: (-x[1]['points'], -(x[1]['goals_for'] - x[1]['goals_against']), -x[1]['goals_for'])
    )

    standings = []
    for pos, (team, s) in enumerate(sorted_teams, 1):
        standings.append({
            'position': pos,
            'team': team,
            'played': s['played'],
            'won': s['won'],
            'drawn': s['drawn'],
            'lost': s['lost'],
            'goals_for': s['goals_for'],
            'goals_against': s['goals_against'],
            'goal_difference': s['goals_for'] - s['goals_against'],
            'points': s['points']
        })

    cursor.execute('UPDATE metadata SET value = ? WHERE key = ?', (json.dumps(standings), 'cached_standings'))
    if cursor.rowcount > 0:
        print(f"Updated cached_standings in metadata")
    else:
        # Insert if not exists
        cursor.execute('INSERT INTO metadata (key, value) VALUES (?, ?)', ('cached_standings', json.dumps(standings)))
        print(f"Inserted cached_standings in metadata")

    conn.commit()
    conn.close()
    print("Done!")


if __name__ == "__main__":
    main()
