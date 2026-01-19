#!/usr/bin/env python3
"""
Sync 100M simulation results to the season_predictions table.
"""

import json
import sqlite3
from pathlib import Path

def main():
    project_root = Path(__file__).parent.parent

    # Load 100M results
    with open(project_root / 'data' / '100m_simulation_results.json') as f:
        data = json.load(f)

    week = data['week']
    teams = data['teams']

    conn = sqlite3.connect(project_root / 'data' / 'motson.db')
    cursor = conn.cursor()

    for team, stats in teams.items():
        # Format position probs as list of 20 values (positions 1-20)
        position_probs = stats.get('position_probs', {})
        position_probs_list = [position_probs.get(str(i), position_probs.get(i, 0)) / 100 for i in range(1, 21)]

        cursor.execute('''
            INSERT OR REPLACE INTO season_predictions
            (team, week, expected_points, expected_position, title_prob, top4_prob,
             relegation_prob, position_probs, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ''', (
            team,
            week,
            stats['expected_points'],
            stats['expected_position'],
            stats['p_title'] / 100,
            stats['p_top4'] / 100,
            stats['p_relegation'] / 100,
            json.dumps(position_probs_list)
        ))

    conn.commit()
    conn.close()
    print(f'Updated season_predictions for {len(teams)} teams at week {week}')

if __name__ == "__main__":
    main()
