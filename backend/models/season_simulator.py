#!/usr/bin/env python3
"""
MOTSON Season Simulator (Vectorized)

Simulates remaining season matches using IRT parameters to generate:
- Predicted final points distribution
- Table position distribution
- Title/Top4/Relegation probabilities

Uses fully vectorized NumPy operations for speed - 100k simulations in seconds.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from .irt_model import gap_to_probabilities, GAP_ANCHORS


@dataclass
class SimulationResult:
    """Results from season simulation for a single team."""
    team: str
    current_points: int

    # Point predictions
    predicted_final_points: float
    predicted_points_std: float
    points_5th_percentile: float
    points_95th_percentile: float

    # Position predictions
    predicted_position: float
    position_5th_percentile: int
    position_95th_percentile: int

    # Outcome probabilities
    p_title: float      # Win the league
    p_top4: float       # Champions League
    p_top6: float       # Europa spots
    p_top10: float      # Upper half
    p_relegation: float # Bottom 3

    # Distribution data (for charts)
    points_distribution: List[Tuple[int, float]]  # [(points, probability), ...]
    position_distribution: List[Tuple[int, float]]  # [(position, probability), ...]


def vectorized_gap_to_probs(gaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized version of gap_to_probabilities.

    Args:
        gaps: Array of gap values (n_fixtures,)

    Returns:
        Tuple of (p_home, p_draw, p_away) arrays, each shape (n_fixtures,)
    """
    # Build anchor arrays
    anchor_gaps = np.array([a[0] for a in GAP_ANCHORS])
    anchor_home = np.array([a[1] for a in GAP_ANCHORS])
    anchor_draw = np.array([a[2] for a in GAP_ANCHORS])
    anchor_away = np.array([a[3] for a in GAP_ANCHORS])

    # Interpolate for each gap value
    p_home = np.interp(gaps, anchor_gaps, anchor_home)
    p_draw = np.interp(gaps, anchor_gaps, anchor_draw)
    p_away = np.interp(gaps, anchor_gaps, anchor_away)

    return p_home, p_draw, p_away


def simulate_season(
    team_states: Dict[str, 'IRTTeamState'],
    remaining_fixtures: List[Dict],
    current_points: Dict[str, int],
    n_simulations: int = 1000,
    seed: Optional[int] = None
) -> Dict[str, SimulationResult]:
    """
    Simulate the remaining season multiple times (vectorized).

    Args:
        team_states: Current IRT state for each team
        remaining_fixtures: List of {home_team, away_team, matchweek} dicts
        current_points: Current points for each team
        n_simulations: Number of Monte Carlo simulations
        seed: Random seed for reproducibility

    Returns:
        Dict mapping team name to SimulationResult
    """
    rng = np.random.default_rng(seed)
    teams = list(team_states.keys())
    n_teams = len(teams)
    team_to_idx = {t: i for i, t in enumerate(teams)}

    # Filter fixtures to only include teams we have states for
    valid_fixtures = [
        f for f in remaining_fixtures
        if f["home_team"] in team_states and f["away_team"] in team_states
    ]
    n_fixtures = len(valid_fixtures)

    if n_fixtures == 0:
        # No remaining fixtures - return current standings
        sorted_teams = sorted(teams, key=lambda t: -current_points.get(t, 0))
        results = {}
        for pos, team in enumerate(sorted_teams, 1):
            pts = current_points.get(team, 0)
            results[team] = SimulationResult(
                team=team,
                current_points=pts,
                predicted_final_points=float(pts),
                predicted_points_std=0.0,
                points_5th_percentile=float(pts),
                points_95th_percentile=float(pts),
                predicted_position=float(pos),
                position_5th_percentile=pos,
                position_95th_percentile=pos,
                p_title=1.0 if pos == 1 else 0.0,
                p_top4=1.0 if pos <= 4 else 0.0,
                p_top6=1.0 if pos <= 6 else 0.0,
                p_top10=1.0 if pos <= 10 else 0.0,
                p_relegation=1.0 if pos >= 18 else 0.0,
                points_distribution=[(pts, 1.0)],
                position_distribution=[(pos, 1.0)],
            )
        return results

    # Build fixture arrays
    home_idx = np.array([team_to_idx[f["home_team"]] for f in valid_fixtures])
    away_idx = np.array([team_to_idx[f["away_team"]] for f in valid_fixtures])

    # Build parameter arrays
    thetas = np.array([team_states[t].theta for t in teams])
    b_homes = np.array([team_states[t].b_home for t in teams])
    b_aways = np.array([team_states[t].b_away for t in teams])

    # Calculate gaps for all fixtures (vectorized)
    home_thetas = thetas[home_idx]
    away_thetas = thetas[away_idx]
    home_b_homes = b_homes[home_idx]
    away_b_aways = b_aways[away_idx]

    m_home = home_thetas - away_b_aways
    m_away = away_thetas - home_b_homes
    gaps = m_home - m_away  # Shape: (n_fixtures,)

    # Get probabilities for all fixtures
    p_home, p_draw, p_away = vectorized_gap_to_probs(gaps)

    # Stack probabilities: shape (n_fixtures, 3)
    probs = np.stack([p_away, p_draw, p_home], axis=1)

    # Cumulative probabilities for sampling
    cum_probs = np.cumsum(probs, axis=1)

    # Generate random values for all fixtures × all simulations
    # Shape: (n_simulations, n_fixtures)
    random_vals = rng.random((n_simulations, n_fixtures))

    # Determine outcomes: 0=away win, 1=draw, 2=home win
    # Shape: (n_simulations, n_fixtures)
    outcomes = np.zeros((n_simulations, n_fixtures), dtype=np.int32)
    outcomes[random_vals > cum_probs[:, 0]] = 1  # At least draw
    outcomes[random_vals > cum_probs[:, 1]] = 2  # Home win

    # Calculate points earned per fixture
    # Home points: 3 if home win (2), 1 if draw (1), 0 if away win (0)
    home_points = np.where(outcomes == 2, 3, np.where(outcomes == 1, 1, 0))
    away_points = np.where(outcomes == 0, 3, np.where(outcomes == 1, 1, 0))

    # Initialize points array with current points
    # Shape: (n_simulations, n_teams)
    sim_points = np.zeros((n_simulations, n_teams), dtype=np.int32)
    for team, pts in current_points.items():
        if team in team_to_idx:
            sim_points[:, team_to_idx[team]] = pts

    # Accumulate points from each fixture using np.add.at
    # This handles the same team appearing in multiple fixtures
    for fix_idx in range(n_fixtures):
        h_idx = home_idx[fix_idx]
        a_idx = away_idx[fix_idx]
        sim_points[:, h_idx] += home_points[:, fix_idx]
        sim_points[:, a_idx] += away_points[:, fix_idx]

    # Calculate positions for each simulation
    # argsort gives indices that would sort ascending, so negate points
    # Shape: (n_simulations, n_teams)
    sorted_indices = np.argsort(-sim_points, axis=1)

    # Convert to positions (1-indexed)
    positions = np.zeros_like(sim_points)
    for sim in range(n_simulations):
        positions[sim, sorted_indices[sim]] = np.arange(1, n_teams + 1)

    # Build results for each team
    results = {}
    for team in teams:
        idx = team_to_idx[team]
        pts_array = sim_points[:, idx]
        pos_array = positions[:, idx]

        # Points statistics
        mean_pts = float(np.mean(pts_array))
        std_pts = float(np.std(pts_array))
        p5_pts = float(np.percentile(pts_array, 5))
        p95_pts = float(np.percentile(pts_array, 95))

        # Position statistics
        mean_pos = float(np.mean(pos_array))
        p5_pos = int(np.percentile(pos_array, 5))  # Best case
        p95_pos = int(np.percentile(pos_array, 95))  # Worst case

        # Outcome probabilities
        p_title = float(np.mean(pos_array == 1))
        p_top4 = float(np.mean(pos_array <= 4))
        p_top6 = float(np.mean(pos_array <= 6))
        p_top10 = float(np.mean(pos_array <= 10))
        p_relegation = float(np.mean(pos_array >= 18))

        # Points distribution
        pts_unique, pts_counts = np.unique(pts_array, return_counts=True)
        points_distribution = [
            (int(pts), float(count / n_simulations))
            for pts, count in zip(pts_unique, pts_counts)
        ]

        # Position distribution
        pos_unique, pos_counts = np.unique(pos_array, return_counts=True)
        position_distribution = [
            (int(pos), float(count / n_simulations))
            for pos, count in zip(pos_unique, pos_counts)
        ]

        results[team] = SimulationResult(
            team=team,
            current_points=current_points.get(team, 0),
            predicted_final_points=mean_pts,
            predicted_points_std=std_pts,
            points_5th_percentile=p5_pts,
            points_95th_percentile=p95_pts,
            predicted_position=mean_pos,
            position_5th_percentile=p5_pos,
            position_95th_percentile=p95_pos,
            p_title=p_title,
            p_top4=p_top4,
            p_top6=p_top6,
            p_top10=p_top10,
            p_relegation=p_relegation,
            points_distribution=points_distribution,
            position_distribution=position_distribution,
        )

    return results


def simulate_from_week(
    team_states: Dict[str, 'IRTTeamState'],
    all_fixtures: List[Dict],
    all_results: List[Dict],
    from_week: int,
    n_simulations: int = 1000,
    seed: Optional[int] = None
) -> Dict[str, SimulationResult]:
    """
    Simulate season from a specific week.

    Calculates current points from results through from_week,
    then simulates remaining fixtures.

    Args:
        team_states: IRT states as of from_week
        all_fixtures: All season fixtures
        all_results: All match results
        from_week: Week to simulate from
        n_simulations: Number of simulations
        seed: Random seed

    Returns:
        Simulation results for each team
    """
    # Calculate points through from_week
    current_points = defaultdict(int)
    for result in all_results:
        if result.get("matchweek", 0) <= from_week:
            home = result["home_team"]
            away = result["away_team"]
            home_goals = result.get("home_goals", 0)
            away_goals = result.get("away_goals", 0)

            if home_goals > away_goals:
                current_points[home] += 3
            elif home_goals < away_goals:
                current_points[away] += 3
            else:
                current_points[home] += 1
                current_points[away] += 1

    # Get remaining fixtures (after from_week)
    remaining_fixtures = [
        f for f in all_fixtures
        if f.get("matchweek", 0) > from_week
    ]

    return simulate_season(
        team_states=team_states,
        remaining_fixtures=remaining_fixtures,
        current_points=dict(current_points),
        n_simulations=n_simulations,
        seed=seed
    )
