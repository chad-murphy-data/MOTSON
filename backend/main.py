"""
MOTSON v2 - FastAPI Application

EPL Prediction Dashboard API

"Track distributions, not point estimates.
Update on cumulative calibration, not individual surprises."
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd

from .config import app_config, model_config, ANALYST_ADJUSTMENTS, EPL_TEAMS_2025_26
from .models.team_state import TeamState, SeasonPrediction
from .models.bayesian_engine import BayesianEngine, predict_match, initialize_team_states
from .models.irt_state import IRTTeamState
from .models.irt_model import gap_to_probabilities
from .models.bayesian_blender import BayesianBlender
from .models.season_simulator import simulate_season, SimulationResult
from .services.data_fetcher import FootballDataAPI
from .services.weekly_update import WeeklyUpdatePipeline, load_initial_team_states
from .services.monte_carlo import MonteCarloSimulator
from .database.db import Database, get_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global state
team_states: Dict[str, TeamState] = {}
last_update_result: Dict = {}


def initialize_from_csv():
    """Initialize team states from CSV data files."""
    global team_states

    # Try to load from database first
    db = get_db()
    db_states = db.get_all_team_states()

    if db_states:
        logger.info(f"Loaded {len(db_states)} team states from database")
        team_states = db_states
        return

    # Otherwise, initialize from CSV files
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    if os.path.exists(os.path.join(data_dir, "team_parameters.csv")):
        logger.info("Initializing team states from CSV files")
        team_states = load_initial_team_states(data_dir)
        logger.info(f"Initialized {len(team_states)} teams")

        # Save to database
        db.save_team_states(team_states)
    else:
        logger.warning("No team_parameters.csv found - starting with empty state")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("MOTSON v2 starting up...")
    initialize_from_csv()
    yield
    # Shutdown
    logger.info("MOTSON v2 shutting down...")


# Create FastAPI app
app = FastAPI(
    title="MOTSON v2",
    description="Model Of Team Strength Outcome Network - EPL Predictions",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API

class StandingsEntry(BaseModel):
    position: int
    team: str
    played: int
    won: int
    drawn: int
    lost: int
    goals_for: int
    goals_against: int
    goal_difference: int
    points: int


class TeamStateResponse(BaseModel):
    team: str
    theta_home: float
    theta_away: float
    effective_theta_home: float
    effective_theta_away: float
    sigma: float
    stickiness: float
    gravity_mean: float
    analyst_adj: float
    expected_points_season: float
    actual_points_season: int
    matches_played: int
    cumulative_z_score: float
    points_per_game: float
    expected_ppg: float


class MatchPredictionResponse(BaseModel):
    match_id: str
    matchweek: int
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    confidence: float


class SeasonPredictionResponse(BaseModel):
    team: str
    title_prob: float
    top4_prob: float
    top6_prob: float
    relegation_prob: float
    expected_position: float
    position_std: float
    expected_points: float
    points_std: float


class CounterfactualRequest(BaseModel):
    match_id: str
    result: str  # "H", "D", or "A"


class UpdateResponse(BaseModel):
    week: int
    timestamp: str
    updates_triggered: int
    teams_updated: List[str]


# Health check

@app.get("/health")
async def health_check():
    """API health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "teams_loaded": len(team_states),
        "version": "2.0.0",
    }


# Standings endpoints

@app.get("/standings/current")
async def get_current_standings():
    """Get current actual league standings - from database cache first, API fallback."""
    db = get_db()

    # Try database cache first
    cached = db.get_cached_standings()
    if cached:
        return {"standings": cached, "source": "cached"}

    # Fallback to API
    api = FootballDataAPI()
    try:
        standings = await api.get_standings()
        db.save_standings(standings)  # Cache for next time
        return {"standings": standings, "source": "football-data.org"}
    except Exception as e:
        logger.error(f"Failed to fetch standings: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/standings/predicted")
async def get_predicted_standings():
    """Get predicted final standings based on Monte Carlo simulation.

    Uses 100M simulation data when available for higher precision.
    """
    # Try 100M data first
    sim_100m = _load_100m_data()
    if sim_100m and sim_100m.get("teams"):
        teams_data = sim_100m["teams"]
        predictions = [
            {
                "team": team,
                "title_prob": data["p_title"] / 100,
                "top4_prob": data["p_top4"] / 100,
                "top6_prob": data["p_top6"] / 100,
                "relegation_prob": data["p_relegation"] / 100,
                "expected_position": data["expected_position"],
                "expected_points": data["expected_points"],
                "current_points": data.get("current_points", 0),
            }
            for team, data in teams_data.items()
        ]
        sorted_preds = sorted(predictions, key=lambda x: x["expected_position"])
        return {
            "standings": sorted_preds,
            "as_of_week": sim_100m.get("week", 0),
            "source": "100m_simulation",
        }

    # Fall back to database
    db = get_db()
    predictions = db.get_season_predictions()

    if not predictions:
        raise HTTPException(status_code=404, detail="No predictions available - run update first")

    # Sort by expected position
    sorted_preds = sorted(predictions, key=lambda x: x["expected_position"])

    return {
        "standings": sorted_preds,
        "as_of_week": predictions[0]["week"] if predictions else 0,
        "source": "database",
    }


# Team endpoints

@app.get("/teams")
async def list_teams():
    """List all teams with current state summary."""
    return {
        "teams": [
            {
                "team": name,
                "theta": round(state.effective_theta_home, 3),
                "sigma": round(state.sigma, 3),
                "stickiness": round(state.stickiness, 3),
            }
            for name, state in sorted(
                team_states.items(),
                key=lambda x: x[1].effective_theta_home,
                reverse=True
            )
        ]
    }


@app.get("/team/{team_name}")
async def get_team_detail(team_name: str):
    """Get detailed team state."""
    # Handle URL encoding
    team_name = team_name.replace("%27", "'")

    if team_name not in team_states:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found")

    state = team_states[team_name]
    return state.to_dict()


@app.get("/team/{team_name}/history")
async def get_team_history(team_name: str):
    """Get team's theta/sigma trajectory over the season."""
    team_name = team_name.replace("%27", "'")

    if team_name not in team_states:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found")

    db = get_db()
    history = db.get_team_history(team_name)

    return {"team": team_name, "history": history}


# Match predictions

@app.get("/predictions/week/{week}")
async def get_week_predictions(week: int):
    """Get match predictions for a specific week - from database first, API fallback."""
    db = get_db()

    # Try database first
    cached_predictions = db.get_match_predictions(week)
    if cached_predictions:
        return {
            "week": week,
            "predictions": cached_predictions,
            "count": len(cached_predictions),
            "source": "cached",
        }

    # Fallback to API
    api = FootballDataAPI()
    try:
        all_fixtures = await api.get_all_fixtures()
        week_fixtures = [f for f in all_fixtures if f.matchweek == week]

        if not week_fixtures:
            raise HTTPException(status_code=404, detail=f"No fixtures found for week {week}")

        predictions = []
        for fixture in week_fixtures:
            if fixture.home_team in team_states and fixture.away_team in team_states:
                pred = predict_match(fixture, team_states)
                predictions.append(pred.to_dict())

        # Cache for next time
        db.save_match_predictions(predictions)

        return {
            "week": week,
            "predictions": predictions,
            "count": len(predictions),
            "source": "football-data.org",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/next")
async def get_next_week_predictions():
    """Get predictions for the next unplayed matchweek."""
    db = get_db()

    # Try to get current week from database first
    current_week = db.get_current_week()

    if current_week > 0:
        return await get_week_predictions(current_week + 1)

    # Fallback to API
    api = FootballDataAPI()
    try:
        current_week = await api.get_current_matchweek()
        return await get_week_predictions(current_week + 1)
    except Exception as e:
        logger.error(f"Failed to get next week predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Season outcome probabilities

def _load_100m_data():
    """Load 100M simulation results if available."""
    from pathlib import Path
    results_path = Path(__file__).parent.parent / "data" / "100m_simulation_results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


@app.get("/probabilities/season")
async def get_season_probabilities():
    """Get title/top4/relegation probabilities for all teams.

    Uses 100M simulation data when available for higher precision.
    """
    # Try 100M data first
    sim_100m = _load_100m_data()
    if sim_100m and sim_100m.get("teams"):
        teams_data = sim_100m["teams"]
        predictions = [
            {
                "team": team,
                "title_prob": data["p_title"] / 100,  # Convert from percentage
                "top4_prob": data["p_top4"] / 100,
                "top6_prob": data["p_top6"] / 100,
                "relegation_prob": data["p_relegation"] / 100,
                "expected_position": data["expected_position"],
                "expected_points": data["expected_points"],
                "current_points": data.get("current_points", 0),
            }
            for team, data in teams_data.items()
        ]
        return {
            "predictions": predictions,
            "as_of_week": sim_100m.get("week", 0),
            "source": "100m_simulation",
            "total_simulations": sim_100m.get("total_simulations"),
        }

    # Fall back to database
    db = get_db()
    predictions = db.get_season_predictions()

    if not predictions:
        raise HTTPException(status_code=404, detail="No predictions available - run update first")

    return {
        "predictions": predictions,
        "as_of_week": predictions[0]["week"] if predictions else 0,
        "source": "database",
    }


@app.get("/probabilities/title")
async def get_title_race():
    """Get title race probabilities, sorted by likelihood.

    Uses 100M simulation data when available for higher precision.
    """
    # Try 100M data first
    sim_100m = _load_100m_data()
    if sim_100m and sim_100m.get("teams"):
        teams_data = sim_100m["teams"]
        title_probs = sorted(
            [{"team": team, "probability": data["p_title"] / 100} for team, data in teams_data.items()],
            key=lambda x: x["probability"],
            reverse=True,
        )
        return {"title_race": title_probs[:10], "source": "100m_simulation"}

    # Fall back to database
    db = get_db()
    predictions = db.get_season_predictions()

    if not predictions:
        raise HTTPException(status_code=404, detail="No predictions available")

    title_probs = sorted(
        [{"team": p["team"], "probability": p["title_prob"]} for p in predictions],
        key=lambda x: x["probability"],
        reverse=True,
    )

    return {"title_race": title_probs[:10], "source": "database"}


@app.get("/probabilities/relegation")
async def get_relegation_battle():
    """Get relegation battle probabilities, sorted by risk.

    Uses 100M simulation data when available for higher precision.
    """
    # Try 100M data first
    sim_100m = _load_100m_data()
    if sim_100m and sim_100m.get("teams"):
        teams_data = sim_100m["teams"]
        relegation_probs = sorted(
            [{"team": team, "probability": data["p_relegation"] / 100} for team, data in teams_data.items()],
            key=lambda x: x["probability"],
            reverse=True,
        )
        return {"relegation_battle": relegation_probs[:10], "source": "100m_simulation"}

    # Fall back to database
    db = get_db()
    predictions = db.get_season_predictions()

    if not predictions:
        raise HTTPException(status_code=404, detail="No predictions available")

    relegation_probs = sorted(
        [{"team": p["team"], "probability": p["relegation_prob"]} for p in predictions],
        key=lambda x: x["probability"],
        reverse=True,
    )

    return {"relegation_battle": relegation_probs[:10], "source": "database"}


# Historical data endpoints

@app.get("/history/points")
async def get_historical_points():
    """Get historical predicted vs actual points for all teams over the season."""
    db = get_db()
    history = db.get_all_teams_history()

    if not history:
        raise HTTPException(status_code=404, detail="No historical data available - run update first")

    # Group by team
    teams_data = {}
    for record in history:
        team = record["team"]
        if team not in teams_data:
            teams_data[team] = []
        teams_data[team].append({
            "week": record["week"],
            "expected_points": record["expected_points"],
            "actual_points": record["actual_points"],
        })

    return {
        "history": teams_data,
        "weeks": sorted(set(r["week"] for r in history)),
    }


@app.get("/history/strength")
async def get_historical_strength():
    """Get historical team strength (theta) trajectories for all teams."""
    db = get_db()

    # Use IRT history which has the proper theta values
    history = db.get_all_irt_teams_history()

    if not history:
        raise HTTPException(status_code=404, detail="No historical data available - run update first")

    # Group by team
    teams_data = {}
    for record in history:
        team = record["team"]
        if team not in teams_data:
            teams_data[team] = []
        teams_data[team].append({
            "week": record["week"],
            "theta_avg": record["theta"],  # IRT theta is already the combined strength
        })

    return {
        "history": teams_data,
        "weeks": sorted(set(r["week"] for r in history)),
    }


@app.get("/history/positions")
async def get_historical_position_probs():
    """Get historical position probability distributions for heat map visualization."""
    db = get_db()
    predictions = db.get_all_season_predictions_history()

    if not predictions:
        raise HTTPException(status_code=404, detail="No historical predictions available - run update first")

    # Get the latest week's predictions for the heat map
    latest_week = max(p["week"] for p in predictions)
    latest_predictions = [p for p in predictions if p["week"] == latest_week]

    # Sort by expected position
    sorted_preds = sorted(latest_predictions, key=lambda x: x["expected_position"])

    return {
        "week": latest_week,
        "predictions": [
            {
                "team": p["team"],
                "position_probs": p["position_probs"],
                "expected_position": p["expected_position"],
            }
            for p in sorted_preds
        ],
    }


@app.get("/history/title-race")
async def get_historical_title_race():
    """Get historical title probability trajectories for all teams over the season."""
    db = get_db()
    predictions = db.get_all_season_predictions_history()

    if not predictions:
        raise HTTPException(status_code=404, detail="No historical predictions available - run update first")

    # Group by team
    teams_data = {}
    for record in predictions:
        team = record["team"]
        if team not in teams_data:
            teams_data[team] = []
        teams_data[team].append({
            "week": record["week"],
            "title_prob": record["title_prob"],
            "top4_prob": record["top4_prob"],
            "relegation_prob": record["relegation_prob"],
            "expected_position": record["expected_position"],
        })

    # Sort each team's data by week
    for team in teams_data:
        teams_data[team] = sorted(teams_data[team], key=lambda x: x["week"])

    return {
        "history": teams_data,
        "weeks": sorted(set(r["week"] for r in predictions)),
    }


# Counterfactual simulation

@app.post("/counterfactual")
async def run_counterfactual(scenarios: List[CounterfactualRequest]):
    """
    Run a counterfactual simulation.

    "What if City had beaten Burnley in week 3?"
    """
    from .models.team_state import Fixture

    db = get_db()
    simulator = MonteCarloSimulator()

    # Get fixtures and match results
    cached_fixtures = db.get_fixtures()
    match_results = db.get_match_results()

    # Create a lookup for match results by match_id
    results_lookup = {str(r.match_id): r for r in match_results}

    if cached_fixtures:
        # Convert cached fixtures to Fixture objects, merging in goal data from results
        all_fixtures = []
        for f in cached_fixtures:
            match_id = f["match_id"]
            result = results_lookup.get(str(match_id))

            fixture = Fixture(
                match_id=match_id,
                matchweek=f["matchweek"],
                date=datetime.fromisoformat(f["date"]) if isinstance(f["date"], str) else f["date"],
                home_team=f["home_team"],
                away_team=f["away_team"],
                status=f["status"],
                # Add goal data from match results if available
                home_goals=result.home_goals if result else None,
                away_goals=result.away_goals if result else None,
            )
            all_fixtures.append(fixture)
    else:
        # Fallback to API
        api = FootballDataAPI()
        try:
            all_fixtures = await api.get_all_fixtures()
            db.save_fixtures(all_fixtures)
        except Exception as e:
            logger.error(f"Failed to get fixtures: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    try:
        # Build counterfactual results dict
        cf_results = {s.match_id: s.result for s in scenarios}

        # Run simulation
        results = simulator.simulate_counterfactual(
            team_states=team_states,
            all_fixtures=all_fixtures,
            counterfactual_results=cf_results,
            n_simulations=model_config.QUICK_SIMULATIONS,
        )

        return {
            "scenarios": [s.dict() for s in scenarios],
            "results": results,
        }
    except Exception as e:
        logger.error(f"Counterfactual simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Admin endpoints

@app.post("/admin/update")
async def trigger_update(background_tasks: BackgroundTasks):
    """Trigger a weekly update (fetches latest results and updates predictions)."""
    global team_states, last_update_result

    try:
        pipeline = WeeklyUpdatePipeline()
        result = await pipeline.run_update(team_states)

        # Save to database
        db = get_db()
        db.save_team_states(team_states)

        # Save state history
        for team_name, state in team_states.items():
            db.save_team_state_history(state, result["week"])

        # Save season predictions
        for pred_dict in result["season_predictions"]:
            pred = SeasonPrediction(
                team=pred_dict["team"],
                position_probs=pred_dict["position_probs"],
                title_prob=pred_dict["title_prob"],
                top4_prob=pred_dict["top4_prob"],
                top6_prob=pred_dict["top6_prob"],
                relegation_prob=pred_dict["relegation_prob"],
                expected_position=pred_dict["expected_position"],
                position_std=pred_dict["position_std"],
                expected_points=pred_dict["expected_points"],
                points_std=pred_dict["points_std"],
            )
            db.save_season_prediction(pred, result["week"])

        db.set_metadata("current_week", str(result["week"]))
        db.set_metadata("last_update", datetime.utcnow().isoformat())

        # Also cache fixtures, match results, predictions, and standings
        try:
            api = FootballDataAPI()
            all_fixtures = await api.get_all_fixtures()
            db.save_fixtures(all_fixtures)

            # Save finished match results for counterfactual page
            finished_matches = await api.get_finished_matches()
            db.save_match_results(finished_matches)

            # Generate and save match predictions for upcoming fixtures
            upcoming_fixtures = [f for f in all_fixtures if f.status != "FINISHED"]
            match_preds = []
            for fixture in upcoming_fixtures:
                if fixture.home_team in team_states and fixture.away_team in team_states:
                    pred = predict_match(fixture, team_states)
                    match_preds.append(pred.to_dict())
            db.save_match_predictions(match_preds)

            # Cache standings
            standings = await api.get_standings()
            db.save_standings(standings)

            logger.info(f"Cached {len(all_fixtures)} fixtures, {len(finished_matches)} results, {len(match_preds)} predictions, standings")
        except Exception as cache_error:
            logger.warning(f"Failed to cache fixtures/standings (non-critical): {cache_error}")

        last_update_result = result

        return {
            "success": True,
            "week": result["week"],
            "updates_triggered": result["updates_triggered"],
            "teams_updated": [
                e["team"] for e in result["explanations"]
                if e["update_triggered"]
            ],
        }
    except Exception as e:
        logger.error(f"Update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/explanations")
async def get_update_explanations():
    """Get explanations for the most recent update."""
    if not last_update_result:
        raise HTTPException(status_code=404, detail="No update has been run yet")

    return {
        "week": last_update_result.get("week"),
        "explanations": last_update_result.get("explanations", []),
    }


@app.get("/admin/last-update")
async def get_last_update_info():
    """Get information about the last update (for dashboard timestamp display)."""
    import json
    from pathlib import Path

    # Try to read from last_update.json file first (written by scheduled_update.py)
    last_update_path = Path(__file__).parent.parent / "data" / "last_update.json"

    if last_update_path.exists():
        with open(last_update_path, "r") as f:
            return json.load(f)

    # Fall back to database metadata
    db = get_db()
    current_week = db.get_metadata("current_week")
    last_update = db.get_metadata("last_update")

    if current_week or last_update:
        return {
            "last_matchweek": int(current_week) if current_week else 0,
            "last_update_timestamp": last_update,
            "teams_updated": [],
            "simulations_run": model_config.DEFAULT_SIMULATIONS,
        }

    raise HTTPException(status_code=404, detail="No update has been run yet")


@app.get("/admin/config")
async def get_model_config():
    """Get current model configuration."""
    return {
        "model": {
            "HOME_ADVANTAGE": model_config.HOME_ADVANTAGE,
            "B0_DRAW": model_config.B0_DRAW,
            "C_MISMATCH": model_config.C_MISMATCH,
            "UPDATE_THRESHOLD": model_config.UPDATE_THRESHOLD,
            "LEARNING_RATE": model_config.LEARNING_RATE,
            "GRAVITY_DECAY_WEEKS": model_config.GRAVITY_DECAY_WEEKS,
            "DEFAULT_SIMULATIONS": model_config.DEFAULT_SIMULATIONS,
        },
        "analyst_adjustments": ANALYST_ADJUSTMENTS,
        "current_season": app_config.CURRENT_SEASON,
    }


@app.get("/admin/debug")
async def debug_api_data():
    """Debug endpoint to see what the football-data.org API is returning."""
    api = FootballDataAPI()

    try:
        all_fixtures = await api.get_all_fixtures()
        finished = [f for f in all_fixtures if f.status == "FINISHED"]
        remaining = [f for f in all_fixtures if f.status != "FINISHED"]

        # Count by status
        status_counts = {}
        for f in all_fixtures:
            status_counts[f.status] = status_counts.get(f.status, 0) + 1

        return {
            "configured_season": app_config.CURRENT_SEASON,
            "total_fixtures": len(all_fixtures),
            "finished_count": len(finished),
            "remaining_count": len(remaining),
            "status_counts": status_counts,
            "all_statuses": list(set(f.status for f in all_fixtures)),
            "matchweeks_finished": sorted(set(f.matchweek for f in finished)) if finished else [],
            "matchweeks_remaining": sorted(set(f.matchweek for f in remaining)) if remaining else [],
            "sample_finished": [f.to_dict() for f in finished[:3]] if finished else [],
            "sample_remaining": [f.to_dict() for f in remaining[:3]] if remaining else [],
        }
    except Exception as e:
        return {"error": str(e)}


# Results endpoint (for displaying match history)

@app.get("/results")
async def get_results(matchweek: Optional[int] = None):
    """Get match results, optionally filtered by matchweek."""
    db = get_db()

    # Try database first
    db_results = db.get_match_results(matchweek)
    if db_results:
        return {
            "results": [r.to_dict() for r in db_results],
            "count": len(db_results),
            "source": "cached",
        }

    # Fallback to API
    api = FootballDataAPI()
    try:
        results = await api.get_finished_matches()

        if matchweek:
            results = [r for r in results if r.matchweek == matchweek]

        return {
            "results": [r.to_dict() for r in results],
            "count": len(results),
            "source": "football-data.org",
        }
    except Exception as e:
        logger.error(f"Failed to fetch results: {e}")
        raise HTTPException(status_code=503, detail=str(e))


# ============================================================
# IRT Model Endpoints (Bayesian IRT with theta, b_home, b_away)
# ============================================================


class IRTTeamStateResponse(BaseModel):
    """IRT team state response model."""
    team: str
    theta: float
    theta_se: float
    b_home: float
    b_home_se: float
    b_away: float
    b_away_se: float
    theta_prior: float
    theta_season: float
    theta_season_se: float
    gravity_weight: float
    momentum_weight: float
    matches_played: int
    expected_points_season: float
    actual_points_season: int
    is_promoted: bool


class IRTMatchPredictionResponse(BaseModel):
    """IRT match prediction response model."""
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    gap: float
    m_home: float
    m_away: float
    confidence: float


@app.get("/irt/teams")
async def list_irt_teams():
    """List all teams with IRT state summary, sorted by theta."""
    db = get_db()
    states = db.get_all_irt_team_states()

    if not states:
        raise HTTPException(status_code=404, detail="No IRT states available - run backfill first")

    # Sort by theta
    sorted_teams = sorted(states.items(), key=lambda x: -x[1].theta)

    return {
        "teams": [
            {
                "team": state.team,
                "theta": round(state.theta, 3),
                "theta_se": round(state.theta_se, 3),
                "b_home": round(state.b_home, 3),
                "b_away": round(state.b_away, 3),
                "gravity_weight": round(state.gravity_weight, 2),
                "matches_played": state.matches_played,
                "actual_points": state.actual_points_season,
            }
            for _, state in sorted_teams
        ],
        "count": len(states),
    }


@app.get("/irt/team/{team_name}")
async def get_irt_team_detail(team_name: str):
    """Get detailed IRT state for a specific team."""
    team_name = team_name.replace("%27", "'")

    db = get_db()
    state = db.get_irt_team_state(team_name)

    if not state:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found")

    return state.to_dict()


@app.get("/irt/team/{team_name}/history")
async def get_irt_team_history(team_name: str):
    """Get IRT state history for a team (week-by-week trajectory)."""
    team_name = team_name.replace("%27", "'")

    db = get_db()
    history = db.get_irt_team_history(team_name)

    if not history:
        raise HTTPException(status_code=404, detail=f"No history for team '{team_name}'")

    return {
        "team": team_name,
        "history": history,
        "weeks": len(history),
    }


@app.get("/irt/history/all")
async def get_all_irt_history():
    """Get IRT state history for all teams (for trajectory charts)."""
    db = get_db()
    history = db.get_all_irt_teams_history()

    if not history:
        raise HTTPException(status_code=404, detail="No IRT history available")

    # Organize by week
    by_week = {}
    for record in history:
        week = record["week"]
        if week not in by_week:
            by_week[week] = []
        by_week[week].append(record)

    return {
        "history": history,
        "by_week": by_week,
        "total_records": len(history),
    }


@app.get("/irt/predict/{home_team}/{away_team}")
async def predict_match_irt(home_team: str, away_team: str):
    """
    Predict a match using the IRT model.

    Uses the formula:
        gap = (theta_home - b_away_opponent) - (theta_away - b_home_opponent)

    Where positive gap favors the home team.
    """
    home_team = home_team.replace("%27", "'")
    away_team = away_team.replace("%27", "'")

    db = get_db()
    home_state = db.get_irt_team_state(home_team)
    away_state = db.get_irt_team_state(away_team)

    if not home_state:
        raise HTTPException(status_code=404, detail=f"Home team '{home_team}' not found")
    if not away_state:
        raise HTTPException(status_code=404, detail=f"Away team '{away_team}' not found")

    # Calculate attack margins
    m_home = home_state.theta - away_state.b_away
    m_away = away_state.theta - home_state.b_home

    # Gap from home team perspective
    gap = m_home - m_away

    # Convert to probabilities
    h_prob, d_prob, a_prob = gap_to_probabilities(gap)

    # Confidence
    import numpy as np
    combined_se = np.sqrt(
        home_state.theta_se**2 + away_state.theta_se**2 +
        home_state.b_home_se**2 + away_state.b_away_se**2
    )
    confidence = max(0.1, 1.0 - combined_se / 1.2)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_prob": round(h_prob, 3),
        "draw_prob": round(d_prob, 3),
        "away_win_prob": round(a_prob, 3),
        "gap": round(gap, 3),
        "m_home": round(m_home, 3),
        "m_away": round(m_away, 3),
        "home_theta": round(home_state.theta, 3),
        "away_theta": round(away_state.theta, 3),
        "home_b_home": round(home_state.b_home, 3),
        "away_b_away": round(away_state.b_away, 3),
        "confidence": round(confidence, 2),
    }


@app.get("/irt/rankings")
async def get_irt_rankings():
    """Get team rankings by various IRT metrics."""
    db = get_db()
    states = db.get_all_irt_team_states()

    if not states:
        raise HTTPException(status_code=404, detail="No IRT states available")

    teams_list = list(states.values())

    # Sort by different metrics
    by_theta = sorted(teams_list, key=lambda x: -x.theta)
    by_b_home = sorted(teams_list, key=lambda x: -x.b_home)
    by_b_away = sorted(teams_list, key=lambda x: -x.b_away)

    return {
        "by_theta": [{"rank": i+1, "team": t.team, "theta": round(t.theta, 3)} for i, t in enumerate(by_theta)],
        "by_b_home": [{"rank": i+1, "team": t.team, "b_home": round(t.b_home, 3)} for i, t in enumerate(by_b_home)],
        "by_b_away": [{"rank": i+1, "team": t.team, "b_away": round(t.b_away, 3)} for i, t in enumerate(by_b_away)],
    }


# ============================================================
# Season Simulation Endpoints
# ============================================================


class IRTCounterfactualRequest(BaseModel):
    """Request model for IRT counterfactual simulation."""
    home_team: str
    away_team: str
    result: str  # "H" = home win, "D" = draw, "A" = away win


@app.get("/irt/simulation/current")
async def get_current_simulation():
    """Get current season simulation results from IRT model."""
    db = get_db()

    # Get IRT states
    irt_states = db.get_all_irt_team_states()
    if not irt_states:
        raise HTTPException(status_code=404, detail="No IRT states available - run backfill first")

    # Get current week
    current_week = db.get_current_week()

    # Get fixtures and results
    fixtures = db.get_fixtures()
    results = db.get_match_results()

    # Calculate current points
    current_points = {team: 0 for team in EPL_TEAMS_2025_26}
    for r in results:
        if r.matchweek <= current_week:
            if r.home_goals > r.away_goals:
                current_points[r.home_team] += 3
            elif r.home_goals < r.away_goals:
                current_points[r.away_team] += 3
            else:
                current_points[r.home_team] += 1
                current_points[r.away_team] += 1

    # Get remaining fixtures
    remaining_fixtures = [
        {"home_team": f["home_team"], "away_team": f["away_team"], "matchweek": f["matchweek"]}
        for f in fixtures if f["matchweek"] > current_week
    ]

    # Run simulation (10k for fast response, high precision)
    sim_results = simulate_season(
        team_states=irt_states,
        remaining_fixtures=remaining_fixtures,
        current_points=current_points,
        n_simulations=10000,
        seed=42
    )

    # Format response
    sorted_results = sorted(sim_results.values(), key=lambda x: x.predicted_position)

    return {
        "week": current_week,
        "simulations": 10000,
        "teams": [
            {
                "team": r.team,
                "current_points": r.current_points,
                "predicted_final_points": round(r.predicted_final_points, 1),
                "points_5th": round(r.points_5th_percentile, 0),
                "points_95th": round(r.points_95th_percentile, 0),
                "predicted_position": round(r.predicted_position, 1),
                "position_5th": r.position_5th_percentile,
                "position_95th": r.position_95th_percentile,
                "p_title": round(r.p_title * 100, 2),
                "p_top4": round(r.p_top4 * 100, 2),
                "p_top6": round(r.p_top6 * 100, 2),
                "p_relegation": round(r.p_relegation * 100, 2),
            }
            for r in sorted_results
        ],
    }


@app.get("/irt/simulation/100m")
async def get_100m_simulation():
    """Get the pre-computed 100M simulation results (Opta Troll Edition)."""
    import json
    from pathlib import Path

    results_path = Path(__file__).parent.parent / "data" / "100m_simulation_results.json"

    if not results_path.exists():
        raise HTTPException(status_code=404, detail="100M simulation results not found")

    with open(results_path, 'r') as f:
        data = json.load(f)

    # Format response with sorted teams
    sorted_teams = sorted(
        data["teams"].items(),
        key=lambda x: x[1].get("expected_position", 20)
    )

    return {
        "generated_at": data.get("generated_at"),
        "total_simulations": data.get("total_simulations"),
        "week": data.get("week"),
        "teams": [
            {
                "team": team,
                "p_title": round(stats["p_title"], 5),
                "p_top4": round(stats["p_top4"], 5),
                "p_top6": round(stats["p_top6"], 5),
                "p_top10": round(stats.get("p_top10", 0), 5),
                "p_relegation": round(stats["p_relegation"], 5),
                "expected_points": round(stats["expected_points"], 2),
                "expected_position": round(stats["expected_position"], 2),
                "current_points": stats.get("current_points", 0),
                "title_count": stats.get("title_count", 0),
                "relegation_count": stats.get("relegation_count", 0),
            }
            for team, stats in sorted_teams
        ],
    }


@app.post("/irt/counterfactual")
async def run_irt_counterfactual(
    scenarios: List[IRTCounterfactualRequest],
    n_simulations: int = Query(default=10000, ge=1000, le=100000)
):
    """
    Run an IRT-based counterfactual simulation.

    Example: "What if Wolves beat Arsenal this weekend?"

    Returns season outcome probabilities with and without the specified results.
    """
    db = get_db()

    # Get IRT states
    irt_states = db.get_all_irt_team_states()
    if not irt_states:
        raise HTTPException(status_code=404, detail="No IRT states available")

    # Get current week
    current_week = db.get_current_week()

    # Get fixtures and results
    fixtures = db.get_fixtures()
    results = db.get_match_results()

    # Calculate current points
    current_points = {team: 0 for team in EPL_TEAMS_2025_26}
    for r in results:
        if r.matchweek <= current_week:
            if r.home_goals > r.away_goals:
                current_points[r.home_team] += 3
            elif r.home_goals < r.away_goals:
                current_points[r.away_team] += 3
            else:
                current_points[r.home_team] += 1
                current_points[r.away_team] += 1

    # Get all remaining fixtures
    remaining_fixtures = [
        {"home_team": f["home_team"], "away_team": f["away_team"], "matchweek": f["matchweek"]}
        for f in fixtures if f["matchweek"] > current_week
    ]

    # Run baseline simulation (without counterfactual)
    baseline = simulate_season(
        team_states=irt_states,
        remaining_fixtures=remaining_fixtures,
        current_points=current_points.copy(),
        n_simulations=n_simulations,
        seed=42
    )

    # Apply counterfactual scenarios
    cf_points = current_points.copy()
    cf_fixtures = []
    applied_scenarios = []

    for scenario in scenarios:
        home = scenario.home_team.replace("%27", "'")
        away = scenario.away_team.replace("%27", "'")

        # Validate teams
        if home not in irt_states:
            raise HTTPException(status_code=400, detail=f"Unknown team: {home}")
        if away not in irt_states:
            raise HTTPException(status_code=400, detail=f"Unknown team: {away}")

        # Find and remove the fixture from remaining
        fixture_found = False
        for f in remaining_fixtures:
            if f["home_team"] == home and f["away_team"] == away:
                cf_fixtures.append(f)
                fixture_found = True
                break

        if not fixture_found:
            raise HTTPException(
                status_code=400,
                detail=f"Fixture {home} vs {away} not found in remaining fixtures"
            )

        # Apply result to points
        result = scenario.result.upper()
        if result == "H":
            cf_points[home] += 3
            applied_scenarios.append(f"{home} beat {away}")
        elif result == "A":
            cf_points[away] += 3
            applied_scenarios.append(f"{away} beat {home} (away)")
        elif result == "D":
            cf_points[home] += 1
            cf_points[away] += 1
            applied_scenarios.append(f"{home} drew with {away}")
        else:
            raise HTTPException(status_code=400, detail=f"Invalid result: {result}. Use H, D, or A")

    # Remove applied fixtures from remaining
    cf_remaining = [
        f for f in remaining_fixtures
        if not any(f["home_team"] == cf["home_team"] and f["away_team"] == cf["away_team"]
                   for cf in cf_fixtures)
    ]

    # Run counterfactual simulation
    counterfactual = simulate_season(
        team_states=irt_states,
        remaining_fixtures=cf_remaining,
        current_points=cf_points,
        n_simulations=n_simulations,
        seed=42
    )

    # Calculate deltas for affected teams
    affected_teams = set()
    for scenario in scenarios:
        affected_teams.add(scenario.home_team.replace("%27", "'"))
        affected_teams.add(scenario.away_team.replace("%27", "'"))

    deltas = []
    for team in affected_teams:
        if team in baseline and team in counterfactual:
            b = baseline[team]
            c = counterfactual[team]
            deltas.append({
                "team": team,
                "p_title_baseline": round(b.p_title * 100, 3),
                "p_title_counterfactual": round(c.p_title * 100, 3),
                "p_title_delta": round((c.p_title - b.p_title) * 100, 3),
                "p_top4_baseline": round(b.p_top4 * 100, 2),
                "p_top4_counterfactual": round(c.p_top4 * 100, 2),
                "p_top4_delta": round((c.p_top4 - b.p_top4) * 100, 2),
                "p_relegation_baseline": round(b.p_relegation * 100, 3),
                "p_relegation_counterfactual": round(c.p_relegation * 100, 3),
                "p_relegation_delta": round((c.p_relegation - b.p_relegation) * 100, 3),
                "points_baseline": round(b.predicted_final_points, 1),
                "points_counterfactual": round(c.predicted_final_points, 1),
                "points_delta": round(c.predicted_final_points - b.predicted_final_points, 1),
            })

    # Full results for all teams (sorted by predicted position in counterfactual)
    sorted_cf = sorted(counterfactual.values(), key=lambda x: x.predicted_position)

    return {
        "scenarios_applied": applied_scenarios,
        "n_simulations": n_simulations,
        "week": current_week,
        "deltas": deltas,
        "baseline": {
            team: {
                "p_title": round(r.p_title * 100, 3),
                "p_top4": round(r.p_top4 * 100, 2),
                "p_relegation": round(r.p_relegation * 100, 3),
                "predicted_points": round(r.predicted_final_points, 1),
                "predicted_position": round(r.predicted_position, 1),
            }
            for team, r in baseline.items()
        },
        "counterfactual": {
            team: {
                "p_title": round(r.p_title * 100, 3),
                "p_top4": round(r.p_top4 * 100, 2),
                "p_relegation": round(r.p_relegation * 100, 3),
                "predicted_points": round(r.predicted_final_points, 1),
                "predicted_position": round(r.predicted_position, 1),
            }
            for team, r in counterfactual.items()
        },
    }


@app.get("/irt/simulation/distributions")
async def get_points_distributions(n_simulations: int = Query(default=100000, ge=10000, le=1000000)):
    """
    Get full points distributions for all teams with interesting thresholds.

    Returns the probability of reaching various point totals:
    - Arsenal: % chance of 90+ points, 95+ points, 100+ points (record territory)
    - Wolves: % chance of <20 points (worst ever), <15 points (unprecedented)
    - All teams: full distribution histograms for visualization
    """
    db = get_db()

    # Get IRT states
    irt_states = db.get_all_irt_team_states()
    if not irt_states:
        raise HTTPException(status_code=404, detail="No IRT states available")

    # Get current week
    current_week = db.get_current_week()

    # Get fixtures and results
    fixtures = db.get_fixtures()
    results = db.get_match_results()

    # Calculate current points
    current_points = {team: 0 for team in EPL_TEAMS_2025_26}
    for r in results:
        if r.matchweek <= current_week:
            if r.home_goals > r.away_goals:
                current_points[r.home_team] += 3
            elif r.home_goals < r.away_goals:
                current_points[r.away_team] += 3
            else:
                current_points[r.home_team] += 1
                current_points[r.away_team] += 1

    # Get remaining fixtures
    remaining_fixtures = [
        {"home_team": f["home_team"], "away_team": f["away_team"], "matchweek": f["matchweek"]}
        for f in fixtures if f["matchweek"] > current_week
    ]

    # Run simulation with high count for accurate distributions
    sim_results = simulate_season(
        team_states=irt_states,
        remaining_fixtures=remaining_fixtures,
        current_points=current_points,
        n_simulations=n_simulations,
        seed=42
    )

    # Historical EPL records for context
    records = {
        "highest_points_ever": 100,  # Man City 2017-18
        "centurion_threshold": 100,
        "record_low_points": 11,  # Derby 2007-08
        "bad_season_threshold": 25,
        "excellent_season_threshold": 90,
        "title_contention_threshold": 85,
    }

    # Build response with distributions and interesting thresholds
    teams_data = []

    for team, r in sorted(sim_results.items(), key=lambda x: x[1].predicted_position):
        # Calculate cumulative probabilities from distribution
        dist = dict(r.points_distribution)  # {points: probability}

        # Above thresholds (cumulative from threshold up)
        p_90_plus = sum(p for pts, p in dist.items() if pts >= 90) * 100
        p_95_plus = sum(p for pts, p in dist.items() if pts >= 95) * 100
        p_100_plus = sum(p for pts, p in dist.items() if pts >= 100) * 100

        # Below thresholds (cumulative from threshold down)
        p_under_20 = sum(p for pts, p in dist.items() if pts < 20) * 100
        p_under_15 = sum(p for pts, p in dist.items() if pts < 15) * 100
        p_under_11 = sum(p for pts, p in dist.items() if pts < 11) * 100  # Worse than Derby

        # Position distribution
        pos_dist = dict(r.position_distribution)
        p_dead_last = pos_dist.get(20, 0) * 100

        teams_data.append({
            "team": team,
            "current_points": r.current_points,
            "predicted_points": round(r.predicted_final_points, 1),
            "points_std": round(r.predicted_points_std, 1),
            "points_5th": round(r.points_5th_percentile, 0),
            "points_95th": round(r.points_95th_percentile, 0),
            "predicted_position": round(r.predicted_position, 1),

            # Upside thresholds
            "p_90_plus_points": round(p_90_plus, 4),
            "p_95_plus_points": round(p_95_plus, 4),
            "p_centurion": round(p_100_plus, 5),  # Extra precision for rare events

            # Downside thresholds
            "p_under_20_points": round(p_under_20, 4),
            "p_under_15_points": round(p_under_15, 4),
            "p_worse_than_derby": round(p_under_11, 6),  # Very rare event

            # Dead last
            "p_dead_last": round(p_dead_last, 3),

            # Full distributions for visualization
            "points_distribution": [
                {"points": int(pts), "probability": round(prob * 100, 4)}
                for pts, prob in sorted(r.points_distribution)
            ],
            "position_distribution": [
                {"position": int(pos), "probability": round(prob * 100, 3)}
                for pos, prob in sorted(r.position_distribution)
            ],
        })

    return {
        "week": current_week,
        "simulations": n_simulations,
        "records": records,
        "teams": teams_data,
    }


@app.get("/irt/simulation/fun-stats")
async def get_fun_stats(n_simulations: int = Query(default=100000, ge=10000, le=1000000)):
    """
    Fun stats for the Opta Troll - extreme scenario probabilities.

    Highlights rare events like:
    - Teams reaching centurion status (100+ points)
    - Teams having historically bad seasons
    - Surprise title/relegation scenarios
    """
    db = get_db()

    # Get IRT states
    irt_states = db.get_all_irt_team_states()
    if not irt_states:
        raise HTTPException(status_code=404, detail="No IRT states available")

    # Get current week
    current_week = db.get_current_week()

    # Get fixtures and results
    fixtures = db.get_fixtures()
    results = db.get_match_results()

    # Calculate current points
    current_points = {team: 0 for team in EPL_TEAMS_2025_26}
    for r in results:
        if r.matchweek <= current_week:
            if r.home_goals > r.away_goals:
                current_points[r.home_team] += 3
            elif r.home_goals < r.away_goals:
                current_points[r.away_team] += 3
            else:
                current_points[r.home_team] += 1
                current_points[r.away_team] += 1

    # Get remaining fixtures
    remaining_fixtures = [
        {"home_team": f["home_team"], "away_team": f["away_team"], "matchweek": f["matchweek"]}
        for f in fixtures if f["matchweek"] > current_week
    ]

    # Run simulation
    sim_results = simulate_season(
        team_states=irt_states,
        remaining_fixtures=remaining_fixtures,
        current_points=current_points,
        n_simulations=n_simulations,
        seed=42
    )

    fun_stats = []

    # Centurion watch (100+ points)
    for team, r in sim_results.items():
        dist = dict(r.points_distribution)
        p_100 = sum(p for pts, p in dist.items() if pts >= 100) * 100
        if p_100 > 0:
            fun_stats.append({
                "category": "centurion_watch",
                "team": team,
                "description": f"{team} reach 100+ points",
                "probability": round(p_100, 5),
                "simulations": int(p_100 * n_simulations / 100),
            })

    # Worse than Derby (record low 11 points)
    for team, r in sim_results.items():
        dist = dict(r.points_distribution)
        p_under_11 = sum(p for pts, p in dist.items() if pts < 11) * 100
        if p_under_11 > 0:
            fun_stats.append({
                "category": "worse_than_derby",
                "team": team,
                "description": f"{team} finish with fewer than 11 points (worse than Derby 07-08)",
                "probability": round(p_under_11, 6),
                "simulations": int(p_under_11 * n_simulations / 100),
            })

    # Under 20 points (historically bad)
    for team, r in sim_results.items():
        dist = dict(r.points_distribution)
        p_under_20 = sum(p for pts, p in dist.items() if pts < 20) * 100
        if p_under_20 > 1:  # Only show if >1% chance
            fun_stats.append({
                "category": "historically_bad",
                "team": team,
                "description": f"{team} finish with fewer than 20 points",
                "probability": round(p_under_20, 3),
                "simulations": int(p_under_20 * n_simulations / 100),
            })

    # 90+ point seasons
    for team, r in sim_results.items():
        dist = dict(r.points_distribution)
        p_90 = sum(p for pts, p in dist.items() if pts >= 90) * 100
        if p_90 > 1:  # Only show if >1% chance
            fun_stats.append({
                "category": "excellent_season",
                "team": team,
                "description": f"{team} reach 90+ points",
                "probability": round(p_90, 2),
                "simulations": int(p_90 * n_simulations / 100),
            })

    # Surprise titles (teams with <5% pre-season but now >0.1%)
    top_6_baseline = {"Arsenal", "Liverpool", "Manchester City", "Chelsea", "Manchester Utd", "Tottenham"}
    for team, r in sim_results.items():
        if team not in top_6_baseline and r.p_title > 0.001:  # >0.1%
            fun_stats.append({
                "category": "surprise_title",
                "team": team,
                "description": f"{team} win the Premier League",
                "probability": round(r.p_title * 100, 3),
                "simulations": int(r.p_title * n_simulations),
            })

    # Surprise relegation (top 6 teams with any relegation chance)
    for team in top_6_baseline:
        if team in sim_results:
            r = sim_results[team]
            if r.p_relegation > 0:
                fun_stats.append({
                    "category": "surprise_relegation",
                    "team": team,
                    "description": f"{team} get relegated",
                    "probability": round(r.p_relegation * 100, 5),
                    "simulations": int(r.p_relegation * n_simulations),
                })

    # Survival odds for bottom teams
    for team, r in sim_results.items():
        if r.p_relegation > 0.5:  # More likely to go down than stay up
            survival_rate = 1 - r.p_relegation
            fun_stats.append({
                "category": "survival_watch",
                "team": team,
                "description": f"{team} avoid relegation",
                "probability": round(survival_rate * 100, 3),
                "simulations": int(survival_rate * n_simulations),
            })

    # Sort by probability descending within category
    fun_stats.sort(key=lambda x: (-x["probability"]))

    return {
        "week": current_week,
        "simulations": n_simulations,
        "stats": fun_stats,
        "headlines": {
            "wolves_survival": next(
                (s for s in fun_stats if s["team"] == "Wolves" and s["category"] == "survival_watch"),
                None
            ),
            "arsenal_centurion": next(
                (s for s in fun_stats if s["team"] == "Arsenal" and s["category"] == "centurion_watch"),
                None
            ),
            "worst_ever_candidate": next(
                (s for s in fun_stats if s["category"] == "worse_than_derby"),
                None
            ),
        },
    }


@app.get("/irt/simulation/rivalries")
async def get_rivalry_comparisons(n_simulations: int = Query(default=100000, ge=10000, le=500000)):
    """
    Get head-to-head rivalry comparison probabilities.

    Who finishes higher: Everton vs Liverpool, United vs City, Arsenal vs Spurs, etc.
    """
    import numpy as np

    db = get_db()

    # Get IRT states
    irt_states = db.get_all_irt_team_states()
    if not irt_states:
        raise HTTPException(status_code=404, detail="No IRT states available")

    # Get current week
    current_week = db.get_current_week()

    # Get fixtures and results
    fixtures = db.get_fixtures()
    results = db.get_match_results()

    # Calculate current points
    current_points = {team: 0 for team in EPL_TEAMS_2025_26}
    for r in results:
        if r.matchweek <= current_week:
            if r.home_goals > r.away_goals:
                current_points[r.home_team] += 3
            elif r.home_goals < r.away_goals:
                current_points[r.away_team] += 3
            else:
                current_points[r.home_team] += 1
                current_points[r.away_team] += 1

    # Get remaining fixtures
    remaining_fixtures = [
        {"home_team": f["home_team"], "away_team": f["away_team"], "matchweek": f["matchweek"]}
        for f in fixtures if f["matchweek"] > current_week
    ]

    # Define classic rivalries
    rivalries = [
        ("Liverpool", "Everton", "Merseyside Derby"),
        ("Manchester City", "Manchester Utd", "Manchester Derby"),
        ("Arsenal", "Tottenham", "North London Derby"),
        ("Arsenal", "Chelsea", "London Rivalry"),
        ("Chelsea", "Tottenham", "London Rivalry"),
        ("Liverpool", "Manchester Utd", "North-West Derby"),
        ("Arsenal", "Liverpool", "Title Race"),
        ("Newcastle Utd", "Sunderland", "Tyne-Wear Derby"),
        ("Aston Villa", "West Ham", "Midlands vs London"),
        ("Leeds United", "Manchester Utd", "Roses Rivalry"),
    ]

    # Filter to teams that exist in current season
    valid_rivalries = [
        (t1, t2, name) for t1, t2, name in rivalries
        if t1 in irt_states and t2 in irt_states
    ]

    # Run simulation with seed for reproducibility
    rng = np.random.default_rng(42)
    teams = list(irt_states.keys())
    n_teams = len(teams)
    team_to_idx = {t: i for i, t in enumerate(teams)}

    # Filter fixtures to only include teams we have states for
    valid_fixtures = [
        f for f in remaining_fixtures
        if f["home_team"] in irt_states and f["away_team"] in irt_states
    ]
    n_fixtures = len(valid_fixtures)

    if n_fixtures == 0:
        # Season complete - use final standings
        sorted_teams = sorted(teams, key=lambda t: -current_points.get(t, 0))
        final_positions = {t: pos for pos, t in enumerate(sorted_teams, 1)}

        rivalry_results = []
        for t1, t2, name in valid_rivalries:
            pos1 = final_positions[t1]
            pos2 = final_positions[t2]
            rivalry_results.append({
                "team1": t1,
                "team2": t2,
                "rivalry_name": name,
                "team1_higher": 100.0 if pos1 < pos2 else 0.0,
                "team2_higher": 100.0 if pos2 < pos1 else 0.0,
                "same_position": 100.0 if pos1 == pos2 else 0.0,
                "team1_current_points": current_points[t1],
                "team2_current_points": current_points[t2],
            })
        return {"week": current_week, "rivalries": rivalry_results, "simulations": 0}

    # Build fixture arrays
    home_idx = np.array([team_to_idx[f["home_team"]] for f in valid_fixtures])
    away_idx = np.array([team_to_idx[f["away_team"]] for f in valid_fixtures])

    # Build parameter arrays
    thetas = np.array([irt_states[t].theta for t in teams])
    b_homes = np.array([irt_states[t].b_home for t in teams])
    b_aways = np.array([irt_states[t].b_away for t in teams])

    # Calculate gaps for all fixtures (vectorized)
    home_thetas = thetas[home_idx]
    away_thetas = thetas[away_idx]
    home_b_homes = b_homes[home_idx]
    away_b_aways = b_aways[away_idx]

    m_home = home_thetas - away_b_aways
    m_away = away_thetas - home_b_homes
    gaps = m_home - m_away

    # Import gap conversion
    from .models.season_simulator import vectorized_gap_to_probs

    # Get probabilities for all fixtures
    p_home, p_draw, p_away = vectorized_gap_to_probs(gaps)

    # Stack probabilities: shape (n_fixtures, 3)
    probs = np.stack([p_away, p_draw, p_home], axis=1)
    cum_probs = np.cumsum(probs, axis=1)

    # Generate random values for all fixtures × all simulations
    random_vals = rng.random((n_simulations, n_fixtures))

    # Determine outcomes: 0=away win, 1=draw, 2=home win
    outcomes = np.zeros((n_simulations, n_fixtures), dtype=np.int32)
    outcomes[random_vals > cum_probs[:, 0]] = 1  # At least draw
    outcomes[random_vals > cum_probs[:, 1]] = 2  # Home win

    # Calculate points earned per fixture
    home_points = np.where(outcomes == 2, 3, np.where(outcomes == 1, 1, 0))
    away_points = np.where(outcomes == 0, 3, np.where(outcomes == 1, 1, 0))

    # Initialize points array with current points
    sim_points = np.zeros((n_simulations, n_teams), dtype=np.int32)
    for team, pts in current_points.items():
        if team in team_to_idx:
            sim_points[:, team_to_idx[team]] = pts

    # Accumulate points from each fixture
    for fix_idx in range(n_fixtures):
        h_idx = home_idx[fix_idx]
        a_idx = away_idx[fix_idx]
        sim_points[:, h_idx] += home_points[:, fix_idx]
        sim_points[:, a_idx] += away_points[:, fix_idx]

    # Calculate positions for each simulation
    sorted_indices = np.argsort(-sim_points, axis=1)
    positions = np.zeros_like(sim_points)
    for sim in range(n_simulations):
        positions[sim, sorted_indices[sim]] = np.arange(1, n_teams + 1)

    # Calculate rivalry probabilities
    rivalry_results = []
    for t1, t2, name in valid_rivalries:
        idx1 = team_to_idx[t1]
        idx2 = team_to_idx[t2]

        pos1 = positions[:, idx1]
        pos2 = positions[:, idx2]

        t1_higher = float(np.mean(pos1 < pos2)) * 100
        t2_higher = float(np.mean(pos2 < pos1)) * 100
        same = float(np.mean(pos1 == pos2)) * 100

        # Also get points difference stats
        pts1 = sim_points[:, idx1]
        pts2 = sim_points[:, idx2]
        avg_pts_diff = float(np.mean(pts1 - pts2))

        rivalry_results.append({
            "team1": t1,
            "team2": t2,
            "rivalry_name": name,
            "team1_higher": round(t1_higher, 2),
            "team2_higher": round(t2_higher, 2),
            "same_position": round(same, 2),
            "team1_current_points": current_points[t1],
            "team2_current_points": current_points[t2],
            "avg_points_difference": round(avg_pts_diff, 1),
        })

    return {
        "week": current_week,
        "simulations": n_simulations,
        "rivalries": rivalry_results,
    }


@app.get("/irt/simulation/history")
async def get_simulation_history(team: Optional[str] = None):
    """
    Get historical simulation predictions week-by-week.

    If team is specified, returns that team's history.
    Otherwise returns summary for all teams.
    """
    db = get_db()

    if team:
        team = team.replace("%27", "'")
        history = db.get_irt_team_history(team)

        if not history:
            raise HTTPException(status_code=404, detail=f"No history for team '{team}'")

        return {
            "team": team,
            "history": [
                {
                    "week": h["week"],
                    "theta": round(h["theta"], 3),
                    "predicted_points": h.get("predicted_final_points"),
                    "predicted_position": h.get("predicted_position"),
                    "p_title": round(h.get("p_title", 0) * 100, 2) if h.get("p_title") else None,
                    "p_top4": round(h.get("p_top4", 0) * 100, 2) if h.get("p_top4") else None,
                    "p_relegation": round(h.get("p_relegation", 0) * 100, 2) if h.get("p_relegation") else None,
                }
                for h in history
            ],
        }

    # All teams summary
    history = db.get_all_irt_teams_history()

    if not history:
        raise HTTPException(status_code=404, detail="No IRT history available")

    # Group by team
    by_team = {}
    for h in history:
        t = h["team"]
        if t not in by_team:
            by_team[t] = []
        by_team[t].append({
            "week": h["week"],
            "theta": round(h["theta"], 3),
            "predicted_points": h.get("predicted_final_points"),
            "predicted_position": h.get("predicted_position"),
            "p_title": round(h.get("p_title", 0) * 100, 2) if h.get("p_title") else None,
            "p_top4": round(h.get("p_top4", 0) * 100, 2) if h.get("p_top4") else None,
            "p_relegation": round(h.get("p_relegation", 0) * 100, 2) if h.get("p_relegation") else None,
        })

    return {
        "teams": by_team,
        "weeks": sorted(set(h["week"] for h in history)),
    }


# Serve React frontend (in production)
# Try multiple possible locations for the frontend build
possible_frontend_paths = [
    os.path.join(os.path.dirname(__file__), "..", "frontend", "build"),  # Local dev
    "/app/frontend/build",  # Docker
    os.path.join(os.getcwd(), "frontend", "build"),  # CWD-relative
]

frontend_path = None
frontend_static_path = None

for path in possible_frontend_paths:
    static_path = os.path.join(path, "static")
    index_path = os.path.join(path, "index.html")
    if os.path.exists(path) and os.path.exists(index_path):
        frontend_path = path
        frontend_static_path = static_path if os.path.exists(static_path) else None
        logger.info(f"Found frontend build at: {path}")
        break

if frontend_path:
    # Mount static files if they exist
    if frontend_static_path:
        app.mount("/static", StaticFiles(directory=frontend_static_path), name="static")

    # Mount assets folder if it exists (Vite puts files there)
    assets_path = os.path.join(frontend_path, "assets")
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        """Serve React frontend for all non-API routes."""
        # Don't catch API routes
        if full_path.startswith("api/") or full_path in ["health", "docs", "openapi.json", "redoc"]:
            raise HTTPException(status_code=404)

        # Try to serve the file directly first (for favicon, etc.)
        file_path = os.path.join(frontend_path, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)

        # Otherwise serve index.html for client-side routing
        index_path = os.path.join(frontend_path, "index.html")
        return FileResponse(index_path)
else:
    logger.warning(f"Frontend build not found in any of: {possible_frontend_paths}")
    logger.info("Running in API-only mode")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
