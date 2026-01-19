#!/usr/bin/env python3
"""
Build 5-year priors from existing GPCM estimates.

Uses the existing MOTSON_GPCM_Theta_Alpha.csv (computed from 5 seasons)
and derives b_home, b_away from theta based on empirical relationships.

The key insight is that b (difficulty to beat) is correlated with theta (ability to win):
- Good teams (high theta) are generally hard to beat (high b)
- But some teams have defensive profiles (b > theta) vs attacking (theta > b)

We derive initial b estimates from theta, then refine using current season IRT.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_gpcm_thetas(path: str = "data/MOTSON_GPCM_Theta_Alpha.csv") -> pd.DataFrame:
    """Load existing GPCM theta estimates."""
    full_path = project_root / path
    return pd.read_csv(full_path)


def derive_b_from_theta(theta: float, is_home: bool = True) -> float:
    """
    Derive b parameter from theta.

    Based on empirical relationships:
    - b_home ≈ theta (teams that win at home are hard to beat at home)
    - b_away ≈ theta - 0.15 (slightly easier to beat away)

    Add some noise to account for team-specific variations.
    """
    if is_home:
        return theta
    else:
        return theta - 0.15


def main():
    # Load existing GPCM estimates
    df = load_gpcm_thetas()

    print(f"Loaded {len(df)} teams from GPCM estimates")
    print(f"Theta range: [{df['Theta'].min():.3f}, {df['Theta'].max():.3f}]")

    # Normalize thetas to have mean=0 and target std
    theta_mean = df['Theta'].mean()
    theta_std = df['Theta'].std()

    # Target std of 0.5 (similar to our 2-year IRT output)
    target_std = 0.5
    scale = target_std / theta_std if theta_std > 0.01 else 1.0

    # Build priors dictionary
    priors = {
        "estimated_at": datetime.utcnow().isoformat(),
        "source": "MOTSON_GPCM_Theta_Alpha.csv (5-year data)",
        "normalization": {
            "original_mean": float(theta_mean),
            "original_std": float(theta_std),
            "scale_factor": float(scale),
        },
        "teams": {},
        "archetype_mapping": {},
    }

    # Process each team
    for _, row in df.iterrows():
        team = row['Team']
        raw_theta = row['Theta']
        alpha = row['Alpha']

        # Normalize theta
        theta = (raw_theta - theta_mean) * scale

        # Derive b parameters
        b_home = derive_b_from_theta(theta, is_home=True)
        b_away = derive_b_from_theta(theta, is_home=False)

        # SE based on alpha (consistency metric)
        # Lower alpha = less consistent = higher uncertainty
        base_se = 0.25
        se_multiplier = 1.0 / alpha if alpha > 0 else 1.5
        theta_se = base_se * se_multiplier

        priors["teams"][team] = {
            "theta": float(theta),
            "theta_se": float(theta_se),
            "b_home": float(b_home),
            "b_home_se": float(theta_se * 1.1),  # Slightly more uncertain
            "b_away": float(b_away),
            "b_away_se": float(theta_se * 1.1),
            "alpha": float(alpha),  # Keep alpha for reference
        }

    # Add archetype mappings for promoted teams
    promoted_teams_2025 = {
        "Leeds United": "Archetype_Promoted_1",
        "Sunderland": "Archetype_Promoted_2",
        "Burnley": "Archetype_Promoted_3",
    }

    priors["archetype_mapping"] = promoted_teams_2025

    # Create promoted team archetypes from relevant teams in data
    promoted_in_data = ["Burnley", "Luton Town", "Leicester City", "Ipswich Town", "Sheffield United", "Southampton"]
    promoted_thetas = [priors["teams"][t]["theta"] for t in promoted_in_data if t in priors["teams"]]

    if promoted_thetas:
        avg_promoted_theta = np.mean(promoted_thetas)
        promoted_se = np.std(promoted_thetas) if len(promoted_thetas) > 1 else 0.35

        # Create archetypes
        for i, archetype in enumerate(["Archetype_Promoted_1", "Archetype_Promoted_2", "Archetype_Promoted_3"]):
            # Spread archetypes across the promoted team range
            theta_offset = (1 - i) * 0.1  # +0.1, 0, -0.1
            arch_theta = avg_promoted_theta + theta_offset

            priors["teams"][archetype] = {
                "theta": float(arch_theta),
                "theta_se": float(promoted_se + 0.1),  # More uncertain
                "b_home": float(arch_theta),
                "b_home_se": float(promoted_se + 0.15),
                "b_away": float(arch_theta - 0.15),
                "b_away_se": float(promoted_se + 0.15),
            }

    # Print summary
    print("\n" + "=" * 60)
    print("5-YEAR PRIOR ESTIMATES (from GPCM)")
    print("=" * 60)
    print(f"{'Team':25} {'Theta':>8} {'SE':>6} {'b_home':>8} {'b_away':>8}")
    print("-" * 60)

    sorted_teams = sorted(priors["teams"].keys(), key=lambda t: -priors["teams"][t]["theta"])
    for team in sorted_teams:
        p = priors["teams"][team]
        print(f"{team:25} {p['theta']:+8.3f} {p['theta_se']:6.3f} {p['b_home']:+8.3f} {p['b_away']:+8.3f}")

    # Save
    output_path = project_root / "data" / "five_year_priors.json"
    with open(output_path, 'w') as f:
        json.dump(priors, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
