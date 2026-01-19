#!/usr/bin/env python3
"""
MOTSON 100 Million Simulation Run

The Opta Troll Edition - 10,000 x 10,000 simulations for maximum precision.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.database.db import get_db
from backend.models.season_simulator import simulate_season
from backend.config import EPL_TEAMS_2025_26


def main():
    print("=" * 70)
    print("MOTSON 100,000,000 SIMULATION RUN")
    print("The Opta Troll Edition")
    print("=" * 70)
    print()

    db = get_db()

    # Get current IRT states
    irt_states = db.get_all_irt_team_states()
    print(f"Loaded IRT states for {len(irt_states)} teams")

    # Get fixtures
    fixtures = db.get_fixtures()
    remaining = [
        {'home_team': f['home_team'], 'away_team': f['away_team'], 'matchweek': f['matchweek']}
        for f in fixtures if f['matchweek'] > 22
    ]
    print(f"Remaining fixtures: {len(remaining)}")

    # Get current points at week 22
    results = db.get_match_results()
    current_points = {team: 0 for team in EPL_TEAMS_2025_26}
    for r in results:
        if r.matchweek <= 22:
            if r.home_goals > r.away_goals:
                current_points[r.home_team] += 3
            elif r.home_goals < r.away_goals:
                current_points[r.away_team] += 3
            else:
                current_points[r.home_team] += 1
                current_points[r.away_team] += 1

    print(f"Current points loaded for week 22")
    print()

    # Configuration
    n_batches = 100
    batch_size = 1_000_000
    total_sims = n_batches * batch_size

    print(f"Running {total_sims:,} simulations in {n_batches} batches of {batch_size:,}")
    print()

    # Accumulators
    team_counts = {
        team: {
            'title': 0,
            'top4': 0,
            'top6': 0,
            'top10': 0,
            'relegation': 0,
            'total_points': 0,
            'total_position': 0,
        }
        for team in EPL_TEAMS_2025_26
    }

    start_time = time.time()

    for batch in range(n_batches):
        batch_start = time.time()
        result = simulate_season(
            irt_states,
            remaining,
            current_points,
            n_simulations=batch_size,
            seed=42 + batch
        )

        for team, r in result.items():
            team_counts[team]['title'] += int(r.p_title * batch_size)
            team_counts[team]['top4'] += int(r.p_top4 * batch_size)
            team_counts[team]['top6'] += int(r.p_top6 * batch_size)
            team_counts[team]['top10'] += int(r.p_top10 * batch_size)
            team_counts[team]['relegation'] += int(r.p_relegation * batch_size)
            team_counts[team]['total_points'] += r.predicted_final_points * batch_size
            team_counts[team]['total_position'] += r.predicted_position * batch_size

        batch_elapsed = time.time() - batch_start
        total_elapsed = time.time() - start_time
        eta = (total_elapsed / (batch + 1)) * (n_batches - batch - 1)

        print(f"  Batch {batch + 1:>3}/{n_batches} done in {batch_elapsed:.1f}s | "
              f"Total: {total_elapsed/60:.1f}min | ETA: {eta/60:.1f}min")

    total_time = time.time() - start_time
    print()
    print(f"Completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print()

    # Calculate final results
    results_data = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_simulations": total_sims,
        "week": 22,
        "teams": {}
    }

    print("=" * 80)
    print("100,000,000 SIMULATION RESULTS")
    print("=" * 80)
    print()
    print(f"{'Team':20} {'p_title':>14} {'p_top4':>14} {'p_relegation':>14} {'Exp Pts':>10}")
    print("-" * 80)

    sorted_teams = sorted(EPL_TEAMS_2025_26, key=lambda t: -team_counts[t]['title'])

    for team in sorted_teams:
        counts = team_counts[team]
        p_title = counts['title'] / total_sims * 100
        p_top4 = counts['top4'] / total_sims * 100
        p_top6 = counts['top6'] / total_sims * 100
        p_top10 = counts['top10'] / total_sims * 100
        p_rel = counts['relegation'] / total_sims * 100
        exp_pts = counts['total_points'] / total_sims
        exp_pos = counts['total_position'] / total_sims

        results_data["teams"][team] = {
            "p_title": p_title,
            "p_top4": p_top4,
            "p_top6": p_top6,
            "p_top10": p_top10,
            "p_relegation": p_rel,
            "expected_points": exp_pts,
            "expected_position": exp_pos,
            "current_points": current_points.get(team, 0),
            "title_count": counts['title'],
            "relegation_count": counts['relegation'],
        }

        print(f"{team:20} {p_title:>13.5f}% {p_top4:>13.5f}% {p_rel:>13.5f}% {exp_pts:>10.2f}")

    # Save results
    output_path = project_root / "data" / "100m_simulation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print()
    print(f"Results saved to {output_path}")

    # Fun stats
    print()
    print("=" * 80)
    print("FUN STATS FOR THE OPTA TROLL")
    print("=" * 80)

    wolves_survival = total_sims - team_counts['Wolves']['relegation']
    print(f"Wolves survived relegation in {wolves_survival:,} out of {total_sims:,} simulations")
    print(f"That's a {wolves_survival/total_sims*100:.6f}% survival rate")

    chelsea_titles = team_counts['Chelsea']['title']
    print(f"Chelsea won the title in {chelsea_titles:,} simulations ({chelsea_titles/total_sims*100:.6f}%)")

    liverpool_titles = team_counts['Liverpool']['title']
    print(f"Liverpool won the title in {liverpool_titles:,} simulations ({liverpool_titles/total_sims*100:.5f}%)")


if __name__ == "__main__":
    main()
