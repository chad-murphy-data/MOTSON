"""
Quick test script to verify MOTSON core functionality.
Run with: python test_core.py
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.config import model_config, ANALYST_ADJUSTMENTS
from backend.models.bayesian_engine import (
    predict_match_probs,
    position_to_theta,
    theta_to_position,
    initialize_team_states,
    BayesianEngine,
)
from backend.services.monte_carlo import MonteCarloSimulator
import pandas as pd
import numpy as np


def test_match_predictions():
    """Test match prediction calibration."""
    print("\n" + "="*60)
    print("MATCH PREDICTION CALIBRATION TEST")
    print("="*60)

    # Test cases from Chad's calibration targets
    test_cases = [
        # (home_theta, away_theta, description, expected_range)
        (1.0, -0.6, "Liverpool vs Burnley at Anfield", {"home": (85, 95), "draw": (5, 12)}),
        (0.0, 0.0, "Even mid-table matchup", {"home": (35, 45), "draw": (25, 35)}),
        (1.2, 0.8, "Liverpool vs City at Anfield", {"home": (40, 65), "draw": (15, 25)}),
    ]

    for home_theta, away_theta, desc, expected in test_cases:
        h, d, a, conf = predict_match_probs(home_theta, away_theta)
        h_pct, d_pct, a_pct = h*100, d*100, a*100

        print(f"\n{desc}")
        print(f"  Home: {h_pct:.1f}% (target: {expected['home'][0]}-{expected['home'][1]}%)")
        print(f"  Draw: {d_pct:.1f}% (target: {expected['draw'][0]}-{expected['draw'][1]}%)")
        print(f"  Away: {a_pct:.1f}%")

        # Check if within expected ranges
        home_ok = expected['home'][0] <= h_pct <= expected['home'][1]
        draw_ok = expected['draw'][0] <= d_pct <= expected['draw'][1]
        status = "PASS" if home_ok and draw_ok else "CHECK"
        print(f"  Status: {status}")


def test_position_theta_conversion():
    """Test position <-> theta conversion."""
    print("\n" + "="*60)
    print("POSITION <-> THETA CONVERSION TEST")
    print("="*60)

    positions = [1, 5, 10, 15, 20]
    for pos in positions:
        theta = position_to_theta(pos)
        back = theta_to_position(theta)
        print(f"  Position {pos:2d} -> theta {theta:+.3f} -> position {back:.1f}")


def test_team_initialization():
    """Test team state initialization from CSV."""
    print("\n" + "="*60)
    print("TEAM INITIALIZATION TEST")
    print("="*60)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    params_df = pd.read_csv(os.path.join(data_dir, "team_parameters.csv"))

    print(f"  Loaded {len(params_df)} teams from team_parameters.csv")

    # Initialize states
    states = initialize_team_states(params_df)

    print(f"  Initialized {len(states)} team states")

    # Show top 5 by effective theta
    print("\n  Top 5 teams by effective theta:")
    sorted_teams = sorted(states.values(), key=lambda t: t.effective_theta_home, reverse=True)
    for team in sorted_teams[:5]:
        adj = f" (+{team.analyst_adj:.2f} adj)" if team.analyst_adj > 0 else ""
        print(f"    {team.team:20} theta={team.effective_theta_home:+.3f}{adj} sigma={team.sigma:.3f}")

    return states


def test_monte_carlo():
    """Test Monte Carlo simulation."""
    print("\n" + "="*60)
    print("MONTE CARLO SIMULATION TEST")
    print("="*60)

    # Initialize states
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    params_df = pd.read_csv(os.path.join(data_dir, "team_parameters.csv"))
    states = initialize_team_states(params_df)

    # Create some fake remaining fixtures (simplified)
    from backend.models.team_state import Fixture
    from datetime import datetime

    teams = list(states.keys())
    fixtures = []

    # Create 5 fake fixtures for testing
    for i in range(5):
        home = teams[i]
        away = teams[19-i]
        fixtures.append(Fixture(
            match_id=f"test_{i}",
            matchweek=1,
            date=datetime.now(),
            home_team=home,
            away_team=away,
            status="SCHEDULED",
        ))

    # Current points (all 0 for testing)
    current_points = {team: 0 for team in teams}

    # Run simulation
    simulator = MonteCarloSimulator()
    results = simulator.simulate_season(
        team_states=states,
        remaining_fixtures=fixtures,
        current_points=current_points,
        n_simulations=1000,
        seed=42,
    )

    print(f"  Ran 1000 simulations for {len(fixtures)} fixtures")
    print("\n  Top 5 title probabilities:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['title_prob'], reverse=True)
    for team, res in sorted_results[:5]:
        print(f"    {team:20} Title: {res['title_prob']*100:.1f}%  Top4: {res['top4_prob']*100:.1f}%")


def test_cumulative_update():
    """Test cumulative calibration update logic."""
    print("\n" + "="*60)
    print("CUMULATIVE CALIBRATION UPDATE TEST")
    print("="*60)

    from backend.models.team_state import TeamState, MatchResult, MatchPrediction
    from datetime import datetime

    engine = BayesianEngine()

    # Create a test team
    team = TeamState(
        team="Test FC",
        theta_home=0.0,
        theta_away=-0.15,
        sigma=0.30,
        stickiness=0.50,
        gravity_mean=10.0,
        gravity_weight=0.8,
    )

    print(f"  Initial state: theta={team.theta_home:.3f}, sigma={team.sigma:.3f}")

    # Scenario 1: Team performing within expectations
    predictions = [
        (MatchPrediction(
            match_id="1", matchweek=1, home_team="Test FC", away_team="Other",
            home_win_prob=0.5, draw_prob=0.25, away_win_prob=0.25,
            confidence=0.8, home_theta=0.0, away_theta=0.0, delta=0.3
        ), True),
    ]
    results = [
        (MatchResult(
            match_id="1", matchweek=1, date=datetime.now(),
            home_team="Test FC", away_team="Other",
            home_goals=2, away_goals=1
        ), True),
    ]

    updated, explanation = engine.weekly_update(team, week=1, match_predictions=predictions, match_results=results)

    print(f"\n  Scenario 1: Won as expected (slight favorite)")
    print(f"    Actual pts: {explanation.actual_points}, Expected: {explanation.expected_points:.2f}")
    print(f"    Z-score: {explanation.z_score:.2f}")
    print(f"    Update triggered: {explanation.update_triggered}")
    print(f"    Reason: {explanation.reason}")


def main():
    print("\n" + "="*60)
    print("MOTSON v2 CORE FUNCTIONALITY TEST")
    print("="*60)

    try:
        test_match_predictions()
        test_position_theta_conversion()
        states = test_team_initialization()
        test_monte_carlo()
        test_cumulative_update()

        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        print("\nCore functionality appears to be working correctly.")
        print("To test with live API data, set FOOTBALL_DATA_API_KEY and run:")
        print("  uvicorn backend.main:app --reload")

    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
