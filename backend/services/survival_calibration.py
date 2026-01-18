"""
MOTSON v2 - Lead Survival Calibration

Calibrates Monte Carlo title probabilities using historical lead survival data.

The core insight: Monte Carlo simulations assume independent match outcomes,
but history shows that title races have "texture" - leads are more volatile
than pure probability suggests. Arsenal's 91% at 7 points feels too high
because historically, 7-point leads with 16 games to go don't hold 91% of the time.

This module provides:
1. Loading historical survival rates from data/lead_survival.json
2. Blending MC probabilities with empirical survival rates
3. Theta-conditioned adjustments (strong teams hold leads better)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class SurvivalCalibrator:
    """
    Calibrates title probabilities using historical lead survival data.

    The calibration blends Monte Carlo output with empirical survival rates
    using a weighted average. The weight given to empirical data increases
    when the lead is small and games remaining is high (where MC tends to
    be overconfident).
    """

    def __init__(self, survival_data_path: Optional[Path] = None):
        """
        Initialize with survival lookup data.

        Args:
            survival_data_path: Path to lead_survival.json. If None, uses default.
        """
        self.survival_data_path = survival_data_path or (PROJECT_ROOT / "data" / "lead_survival.json")
        self.smoothed_lookup: Dict[str, float] = {}
        self.raw_data: Dict = {}
        self._loaded = False

    def load(self) -> bool:
        """Load survival data from file."""
        if self._loaded:
            return True

        if not self.survival_data_path.exists():
            logger.warning(f"Survival data not found at {self.survival_data_path}")
            logger.info("Run scripts/build_survival_curves.py to generate it")
            return False

        try:
            with open(self.survival_data_path) as f:
                data = json.load(f)

            self.smoothed_lookup = data.get("smoothed_lookup", {})
            self.raw_data = data.get("raw_survival_table", {})
            self._loaded = True

            logger.info(f"Loaded survival data from {self.survival_data_path}")
            logger.info(f"  {len(self.smoothed_lookup)} lookup entries")
            return True

        except Exception as e:
            logger.error(f"Failed to load survival data: {e}")
            return False

    def get_survival_rate(self, lead_margin: int, games_remaining: int) -> Optional[float]:
        """
        Get historical survival rate for a given lead/games situation.

        Args:
            lead_margin: Points ahead of 2nd place (can be negative)
            games_remaining: Games left for the leader

        Returns:
            Survival rate (0-1) or None if no data
        """
        if not self._loaded and not self.load():
            return None

        key = f"{lead_margin}_{games_remaining}"
        return self.smoothed_lookup.get(key)

    def calibrate_title_prob(
        self,
        mc_title_prob: float,
        lead_margin: int,
        games_remaining: int,
        theta_advantage: float = 0.0,
        blend_weight: float = 0.5,
    ) -> float:
        """
        Calibrate Monte Carlo title probability using survival data.

        The calibration uses a weighted blend:
            calibrated = blend * empirical + (1 - blend) * mc

        The blend weight is adjusted based on:
        - Games remaining (more weight to empirical early in season)
        - Lead size (more weight to empirical for small leads)
        - Theta advantage (less weight to empirical for dominant teams)

        Args:
            mc_title_prob: Raw Monte Carlo title probability
            lead_margin: Points ahead of 2nd place
            games_remaining: Games left in season
            theta_advantage: Leader's theta minus 2nd place theta (strength gap)
            blend_weight: Base weight for empirical data (0-1)

        Returns:
            Calibrated title probability
        """
        survival_rate = self.get_survival_rate(lead_margin, games_remaining)

        if survival_rate is None:
            # No survival data - fall back to MC
            return mc_title_prob

        # Adjust blend weight based on context
        adjusted_weight = self._compute_adaptive_weight(
            base_weight=blend_weight,
            lead_margin=lead_margin,
            games_remaining=games_remaining,
            theta_advantage=theta_advantage,
        )

        # Blend probabilities
        calibrated = adjusted_weight * survival_rate + (1 - adjusted_weight) * mc_title_prob

        # Ensure valid probability
        calibrated = max(0.0, min(1.0, calibrated))

        logger.debug(
            f"Calibration: lead={lead_margin}, games={games_remaining}, "
            f"mc={mc_title_prob:.3f}, empirical={survival_rate:.3f}, "
            f"weight={adjusted_weight:.3f}, final={calibrated:.3f}"
        )

        return calibrated

    def _compute_adaptive_weight(
        self,
        base_weight: float,
        lead_margin: int,
        games_remaining: int,
        theta_advantage: float,
    ) -> float:
        """
        Compute adaptive blend weight based on context.

        The weight given to empirical data should be:
        - Higher early in season (where MC overestimates certainty)
        - Lower for very small leads (where empirical might be noisy)
        - Slightly lower for teams with large theta advantages (they're different from historical average)

        We want substantial weight to empirical data because the user observed:
        "7 points with 16 games to go feels more like 7 points with 10 games to go"
        This suggests MC is overconfident about title probabilities.
        """
        # Games factor: more weight to empirical when many games left
        # At 16+ games: 1.0, at 5 games: ~0.7, at 1 game: ~0.5
        games_factor = 0.5 + 0.5 * min(1.0, games_remaining / 16)

        # Lead factor: full weight for moderate leads (5-10 pts)
        # Slightly reduced for very small leads (noisy) or very large leads
        if abs(lead_margin) < 3:
            lead_factor = 0.85  # Small leads - slightly less empirical weight
        elif abs(lead_margin) > 12:
            lead_factor = 0.75  # Huge leads - slightly less empirical (MC already captures dominance)
        else:
            lead_factor = 1.0  # Sweet spot - trust empirical fully

        # Theta factor: strong teams beat historical averages slightly
        # But we don't want to discount empirical too much - "bottling" is real
        # Theta advantage of 0.2+ reduces empirical weight by up to 20%
        theta_factor = 1.0 - min(0.20, max(0, theta_advantage) / 1.0)

        # Combine factors - use geometric mean to keep factors meaningful
        adjusted = base_weight * games_factor * lead_factor * theta_factor

        # Higher minimum to ensure empirical always matters
        return max(0.30, min(0.85, adjusted))

    def calibrate_all_teams(
        self,
        mc_results: Dict[str, Dict],
        current_standings: Dict[str, int],  # team -> points
        games_remaining: int,
        team_thetas: Dict[str, float],
    ) -> Dict[str, Dict]:
        """
        Calibrate title probabilities for all teams.

        Args:
            mc_results: Raw Monte Carlo results (from simulate_season)
            current_standings: Current points for each team
            games_remaining: Games left in season (for leader)
            team_thetas: Current theta values for each team

        Returns:
            mc_results with calibrated title_prob values
        """
        if not self._loaded and not self.load():
            logger.warning("No survival data - returning uncalibrated results")
            return mc_results

        # Find current leader and 2nd place
        sorted_teams = sorted(current_standings.items(), key=lambda x: -x[1])

        if len(sorted_teams) < 2:
            return mc_results

        leader, leader_points = sorted_teams[0]
        second, second_points = sorted_teams[1]
        lead_margin = leader_points - second_points

        # Get theta advantage
        leader_theta = team_thetas.get(leader, 0)
        second_theta = team_thetas.get(second, 0)
        theta_advantage = leader_theta - second_theta

        # Calibrate each team's title probability
        calibrated_results = {}
        for team, team_data in mc_results.items():
            team_data_copy = dict(team_data)

            # Calculate this team's lead margin
            team_points = current_standings.get(team, 0)
            team_lead = team_points - second_points if team == leader else team_points - leader_points

            # Get this team's theta advantage over the field
            team_theta = team_thetas.get(team, 0)
            if team == leader:
                team_theta_adv = leader_theta - second_theta
            else:
                team_theta_adv = team_theta - leader_theta

            # Calibrate title probability
            mc_title = team_data["title_prob"]
            calibrated_title = self.calibrate_title_prob(
                mc_title_prob=mc_title,
                lead_margin=team_lead,
                games_remaining=games_remaining,
                theta_advantage=team_theta_adv,
            )

            team_data_copy["title_prob"] = calibrated_title
            team_data_copy["title_prob_uncalibrated"] = mc_title

            calibrated_results[team] = team_data_copy

        # Log calibration summary for leader
        if leader in calibrated_results:
            mc_prob = mc_results[leader]["title_prob"]
            cal_prob = calibrated_results[leader]["title_prob"]
            logger.info(
                f"Title calibration: {leader} lead={lead_margin}pts, "
                f"games={games_remaining}, mc={mc_prob:.1%}, calibrated={cal_prob:.1%}"
            )

        return calibrated_results


# Module-level instance for convenience
_calibrator = None


def get_calibrator() -> SurvivalCalibrator:
    """Get or create the global calibrator instance."""
    global _calibrator
    if _calibrator is None:
        _calibrator = SurvivalCalibrator()
    return _calibrator


def calibrate_title_probabilities(
    mc_results: Dict[str, Dict],
    current_standings: Dict[str, int],
    games_remaining: int,
    team_thetas: Dict[str, float],
) -> Dict[str, Dict]:
    """
    Convenience function to calibrate Monte Carlo results.

    Args:
        mc_results: Raw Monte Carlo simulation results
        current_standings: Current points for each team
        games_remaining: Games left in season
        team_thetas: Theta values for each team

    Returns:
        Calibrated results with adjusted title_prob
    """
    calibrator = get_calibrator()
    return calibrator.calibrate_all_teams(
        mc_results=mc_results,
        current_standings=current_standings,
        games_remaining=games_remaining,
        team_thetas=team_thetas,
    )
