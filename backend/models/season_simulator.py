#!/usr/bin/env python3
"""
MOTSON Season Simulator

Simulates remaining season matches using IRT parameters to generate:
- Predicted final points distribution
- Table position distribution
- Title/Top4/Relegation probabilities
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from .irt_model import gap_to_probabilities


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


def simulate_match(
    home_theta: float,
    home_b_home: float,
    away_theta: float,
    away_b_away: float,
    rng: np.random.Generator
) -> Tuple[int, int]:
    """
    Simulate a single match outcome.

    Returns:
        (home_points, away_points) - points earned by each team
    """
    # Calculate gap
    m_home = home_theta - away_b_away
    m_away = away_theta - home_b_home
    gap = m_home - m_away

    # Get probabilities
    p_home, p_draw, p_away = gap_to_probabilities(gap)

    # Sample outcome
    outcome = rng.choice([0, 1, 2], p=[p_away, p_draw, p_home])

    if outcome == 2:  # Home win
        return (3, 0)
    elif outcome == 1:  # Draw
        return (1, 1)
    else:  # Away win
        return (0, 3)


def simulate_season(
    team_states: Dict[str, 'IRTTeamState'],
    remaining_fixtures: List[Dict],
    current_points: Dict[str, int],
    n_simulations: int = 1000,
    seed: Optional[int] = None
) -> Dict[str, SimulationResult]:
    """
    Simulate the remaining season multiple times.

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

    # Storage for simulation outcomes
    final_points = {team: [] for team in teams}
    final_positions = {team: [] for team in teams}

    for sim in range(n_simulations):
        # Start with current points
        sim_points = dict(current_points)

        # Simulate each remaining match
        for fixture in remaining_fixtures:
            home = fixture["home_team"]
            away = fixture["away_team"]

            # Get team states
            home_state = team_states.get(home)
            away_state = team_states.get(away)

            if not home_state or not away_state:
                continue

            # Simulate match
            home_pts, away_pts = simulate_match(
                home_theta=home_state.theta,
                home_b_home=home_state.b_home,
                away_theta=away_state.theta,
                away_b_away=away_state.b_away,
                rng=rng
            )

            sim_points[home] = sim_points.get(home, 0) + home_pts
            sim_points[away] = sim_points.get(away, 0) + away_pts

        # Record final points
        for team in teams:
            final_points[team].append(sim_points.get(team, 0))

        # Calculate positions for this simulation
        sorted_teams = sorted(teams, key=lambda t: -sim_points.get(t, 0))
        for pos, team in enumerate(sorted_teams, 1):
            final_positions[team].append(pos)

    # Build results
    results = {}
    for team in teams:
        pts_array = np.array(final_points[team])
        pos_array = np.array(final_positions[team])

        # Points statistics
        mean_pts = float(np.mean(pts_array))
        std_pts = float(np.std(pts_array))
        p5_pts = float(np.percentile(pts_array, 5))
        p95_pts = float(np.percentile(pts_array, 95))

        # Position statistics
        mean_pos = float(np.mean(pos_array))
        p5_pos = int(np.percentile(pos_array, 5))  # Best case (lower is better)
        p95_pos = int(np.percentile(pos_array, 95))  # Worst case

        # Outcome probabilities
        p_title = float(np.mean(pos_array == 1))
        p_top4 = float(np.mean(pos_array <= 4))
        p_top6 = float(np.mean(pos_array <= 6))
        p_top10 = float(np.mean(pos_array <= 10))
        p_relegation = float(np.mean(pos_array >= 18))

        # Points distribution (histogram)
        pts_min, pts_max = int(pts_array.min()), int(pts_array.max())
        pts_counts = defaultdict(int)
        for p in pts_array:
            pts_counts[int(p)] += 1
        points_distribution = [
            (pts, count / n_simulations)
            for pts, count in sorted(pts_counts.items())
        ]

        # Position distribution
        pos_counts = defaultdict(int)
        for p in pos_array:
            pos_counts[int(p)] += 1
        position_distribution = [
            (pos, count / n_simulations)
            for pos, count in sorted(pos_counts.items())
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
