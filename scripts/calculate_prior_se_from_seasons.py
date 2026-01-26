#!/usr/bin/env python3
"""
Calculate principled prior_se values by running IRT on each season independently
and measuring cross-season theta variance for each team.

This gives us:
- Teams with consistent performance → low SE (we're confident about their level)
- Teams with volatile performance → higher SE (more uncertainty)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict
import json

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.models.irt_model import fit_irt_model


# Team name mappings (football-data.co.uk uses different names)
TEAM_NAME_MAP = {
    "Man City": "Manchester City",
    "Man United": "Manchester Utd",
    "Spurs": "Tottenham",
    "Newcastle": "Newcastle Utd",
    "Nott'm Forest": "Nott'ham Forest",
    "Sheffield United": "Sheffield Utd",
    "West Brom": "West Bromwich",
    "Wolves": "Wolves",
}


def normalize_team_name(name):
    """Normalize team names to our standard format."""
    return TEAM_NAME_MAP.get(name, name)


def fetch_season_data(season_code):
    """Fetch a season's data from football-data.co.uk."""
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
    print(f"  Fetching {url}...")

    try:
        df = pd.read_csv(url)

        # Convert to match format
        matches = []
        for _, row in df.iterrows():
            if pd.isna(row.get('FTHG')) or pd.isna(row.get('FTAG')):
                continue

            home = normalize_team_name(row['HomeTeam'])
            away = normalize_team_name(row['AwayTeam'])
            hg, ag = int(row['FTHG']), int(row['FTAG'])

            if hg > ag:
                outcome = 2  # home win
            elif hg < ag:
                outcome = 0  # away win
            else:
                outcome = 1  # draw

            matches.append({
                'home_team': home,
                'away_team': away,
                'outcome': outcome
            })

        print(f"    Got {len(matches)} matches")
        return matches
    except Exception as e:
        print(f"    Error: {e}")
        return []


def fit_season_irt(matches):
    """Fit IRT model to a single season's matches."""
    # Build team index
    teams = set()
    for m in matches:
        teams.add(m['home_team'])
        teams.add(m['away_team'])

    team_to_idx = {t: i for i, t in enumerate(sorted(teams))}

    # Fit IRT
    params = fit_irt_model(
        matches,
        team_to_idx,
        theta_prior_mean=None,  # Uninformative
        theta_prior_std=0.75,
        n_warmup=300,
        n_samples=500,
    )

    return params.theta, params.theta_se


def main():
    # Seasons to analyze (most recent 5 full seasons)
    seasons = [
        ('2020-21', '2021'),
        ('2021-22', '2122'),
        ('2022-23', '2223'),
        ('2023-24', '2324'),
        ('2024-25', '2425'),
    ]

    # Store theta estimates per team per season
    team_thetas = defaultdict(list)
    team_season_se = defaultdict(list)

    print("=" * 60)
    print("CALCULATING PRIOR SE FROM CROSS-SEASON THETA VARIANCE")
    print("=" * 60)

    for season_name, season_code in seasons:
        print(f"\n--- {season_name} Season ---")
        matches = fetch_season_data(season_code)

        if not matches:
            print(f"  Skipping {season_name} (no data)")
            continue

        print(f"  Fitting IRT model...")
        thetas, ses = fit_season_irt(matches)

        # Store results
        for team, theta in thetas.items():
            team_thetas[team].append(theta)
            team_season_se[team].append(ses[team])

        # Print top/bottom teams for this season
        sorted_teams = sorted(thetas.items(), key=lambda x: -x[1])
        print(f"  Top 3: {sorted_teams[0][0]} ({sorted_teams[0][1]:.3f}), "
              f"{sorted_teams[1][0]} ({sorted_teams[1][1]:.3f}), "
              f"{sorted_teams[2][0]} ({sorted_teams[2][1]:.3f})")
        print(f"  Bottom 3: {sorted_teams[-3][0]} ({sorted_teams[-3][1]:.3f}), "
              f"{sorted_teams[-2][0]} ({sorted_teams[-2][1]:.3f}), "
              f"{sorted_teams[-1][0]} ({sorted_teams[-1][1]:.3f})")

    # Calculate cross-season statistics
    print("\n" + "=" * 60)
    print("CROSS-SEASON THETA STATISTICS")
    print("=" * 60)

    results = {}

    # For teams with multiple seasons, calculate SD of their z-scored thetas
    # Z-scoring within each season removes the arbitrary scale issue
    for team, thetas in team_thetas.items():
        n_seasons = len(thetas)

        if n_seasons >= 3:  # Need at least 3 seasons for meaningful variance
            # Calculate the SD of raw thetas (this captures true volatility)
            theta_sd = np.std(thetas)
            theta_mean = np.mean(thetas)

            # The prior SE should reflect:
            # 1. Cross-season volatility (theta_sd)
            # 2. Within-season estimation uncertainty (avg of season SEs)
            avg_season_se = np.mean(team_season_se[team])

            # Combined SE: sqrt(cross-season variance + avg within-season variance)
            # But we want to emphasize cross-season variance for the prior
            # Use a weighted combination favoring cross-season volatility
            combined_se = np.sqrt(theta_sd**2 + (avg_season_se**2) * 0.25)

            # Clamp to reasonable range [0.08, 0.25]
            combined_se = max(0.08, min(0.25, combined_se))

            results[team] = {
                'theta_mean': theta_mean,
                'theta_sd': theta_sd,
                'avg_season_se': avg_season_se,
                'combined_se': combined_se,
                'n_seasons': n_seasons,
                'thetas': thetas,
            }

    # Sort by theta_mean (strongest first)
    sorted_results = sorted(results.items(), key=lambda x: -x[1]['theta_mean'])

    print(f"\n{'Team':25} {'Mean θ':>8} {'SD':>8} {'Seasons':>8} {'Prior SE':>10}")
    print("-" * 65)

    for team, data in sorted_results:
        print(f"{team:25} {data['theta_mean']:+8.3f} {data['theta_sd']:8.3f} "
              f"{data['n_seasons']:8d} {data['combined_se']:10.3f}")

    # Save results
    output = {
        'calculated_at': pd.Timestamp.now().isoformat(),
        'method': 'cross-season theta variance',
        'teams': {
            team: {
                'theta_mean': float(data['theta_mean']),
                'theta_sd': float(data['theta_sd']),
                'recommended_se': float(data['combined_se']),
                'n_seasons': data['n_seasons'],
            }
            for team, data in results.items()
        }
    }

    output_path = project_root / 'data' / 'cross_season_theta_variance.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    ses = [d['combined_se'] for d in results.values()]
    print(f"Prior SE range: {min(ses):.3f} to {max(ses):.3f}")
    print(f"Prior SE mean: {np.mean(ses):.3f}")
    print(f"Prior SE median: {np.median(ses):.3f}")


if __name__ == "__main__":
    main()
