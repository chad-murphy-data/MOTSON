"""
MOTSON v2 Configuration

Hyperparameters tuned based on Chad's calibration intuitions:
- Top club vs relegation at home: ~90% home win
- Even mid-table matchup: 40/30/30 (H/D/A)
- Big 6 clash: 15-20% draws
- Draws shouldn't collapse below ~10% even for mismatches
"""

import os
from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelConfig:
    """Core model hyperparameters."""

    # Match prediction
    HOME_ADVANTAGE: float = 0.30       # Theta boost for home team
    B0_DRAW: float = 0.05              # Base draw probability (raised from -0.15)
    C_MISMATCH: float = 0.50           # Draw reduction for theta mismatch (lowered from 0.7)
    K_SCALE: float = 1.0               # Win/loss logit scaling

    # Weekly updates (cumulative calibration)
    UPDATE_THRESHOLD: float = 1.0      # Z-score needed to trigger theta update
    LEARNING_RATE: float = 0.10        # Base theta change per excess Z
    BASELINE_SIGMA: float = 0.30       # Reference sigma for scaling
    POINTS_VARIANCE_PER_MATCH: float = 1.5  # For SE calculation

    # Gravity (historical pull)
    GRAVITY_DECAY_WEEKS: float = 20.0  # Half-life of gravity pull
    INITIAL_GRAVITY_WEIGHT: float = 0.8  # Starting gravity strength

    # Monte Carlo simulation
    DEFAULT_SIMULATIONS: int = 10001   # Number of season simulations (one more than Opta's supercomputer)
    QUICK_SIMULATIONS: int = 1001      # For counterfactuals (still better than Opta)

    # Home/away theta offset
    HOME_AWAY_OFFSET: float = 0.15     # Away theta = home theta - this

    # Preseason uncertainty - we're less certain before matches are played
    # This adds uncertainty that decays as the season progresses
    PRESEASON_SIGMA_BOOST: float = 0.20   # Extra sigma added at week 0
    SIGMA_DECAY_WEEKS: float = 10.0       # Half-life of preseason uncertainty decay


@dataclass
class APIConfig:
    """External API configuration."""

    FOOTBALL_DATA_BASE_URL: str = "https://api.football-data.org/v4"
    FOOTBALL_DATA_API_KEY: str = os.environ.get("FOOTBALL_DATA_API_KEY", "")

    # Rate limiting (free tier: 10 requests/minute)
    RATE_LIMIT_REQUESTS: int = 10
    RATE_LIMIT_WINDOW: int = 60  # seconds


@dataclass
class AppConfig:
    """Application configuration."""

    DATABASE_URL: str = os.environ.get("DATABASE_URL", "sqlite:///motson.db")
    DEBUG: bool = os.environ.get("DEBUG", "false").lower() == "true"

    # Current season
    CURRENT_SEASON: int = 2025
    COMPETITION_CODE: str = "PL"  # Premier League

    # Update schedule
    AUTO_UPDATE_ENABLED: bool = True
    UPDATE_DAY: str = "Monday"  # Day to run weekly updates
    UPDATE_HOUR: int = 6  # Hour (UTC) to run updates


# Analyst adjustments (Chad's expert overrides)
# Note: Arsenal boost removed - now reflected in recalculated GPCM thetas
ANALYST_ADJUSTMENTS: Dict[str, float] = {
    "Newcastle Utd": 0.10,  # Upward trajectory, Saudi investment
    "Manchester Utd": -0.10,  # Chaos factor, decline
    "Tottenham": -0.05,     # Spursy coefficient
}


# Team name mappings (API names -> display names)
TEAM_NAME_MAP: Dict[str, str] = {
    "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion FC": "Brighton",
    "Manchester United FC": "Manchester Utd",
    "Newcastle United FC": "Newcastle Utd",
    "Nottingham Forest FC": "Nott'ham Forest",
    "Wolverhampton Wanderers FC": "Wolves",
    "West Ham United FC": "West Ham",
    "Tottenham Hotspur FC": "Tottenham",
    "AFC Sunderland": "Sunderland",
    # Add more as needed from API responses
}


# Current 2025-26 EPL teams
EPL_TEAMS_2025_26 = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Burnley",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Leeds United",
    "Liverpool",
    "Manchester City",
    "Manchester Utd",
    "Newcastle Utd",
    "Nott'ham Forest",
    "Sunderland",
    "Tottenham",
    "West Ham",
    "Wolves",
]


# Instantiate default configs
model_config = ModelConfig()
api_config = APIConfig()
app_config = AppConfig()
