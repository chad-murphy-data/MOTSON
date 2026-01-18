"""
MOTSON v2 - FastAPI Application

EPL Prediction Dashboard API

"Track distributions, not point estimates.
Update on cumulative calibration, not individual surprises."
"""

import os
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
    """Get current actual league standings from API."""
    api = FootballDataAPI()
    try:
        standings = await api.get_standings()
        return {"standings": standings, "source": "football-data.org"}
    except Exception as e:
        logger.error(f"Failed to fetch standings: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/standings/predicted")
async def get_predicted_standings():
    """Get predicted final standings based on Monte Carlo simulation."""
    db = get_db()
    predictions = db.get_season_predictions()

    if not predictions:
        raise HTTPException(status_code=404, detail="No predictions available - run update first")

    # Sort by expected position
    sorted_preds = sorted(predictions, key=lambda x: x["expected_position"])

    return {
        "standings": sorted_preds,
        "as_of_week": predictions[0]["week"] if predictions else 0,
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
    """Get match predictions for a specific week."""
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

        return {
            "week": week,
            "predictions": predictions,
            "count": len(predictions),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/next")
async def get_next_week_predictions():
    """Get predictions for the next unplayed matchweek."""
    api = FootballDataAPI()

    try:
        current_week = await api.get_current_matchweek()
        return await get_week_predictions(current_week + 1)
    except Exception as e:
        logger.error(f"Failed to get next week predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Season outcome probabilities

@app.get("/probabilities/season")
async def get_season_probabilities():
    """Get title/top4/relegation probabilities for all teams."""
    db = get_db()
    predictions = db.get_season_predictions()

    if not predictions:
        raise HTTPException(status_code=404, detail="No predictions available - run update first")

    return {
        "predictions": predictions,
        "as_of_week": predictions[0]["week"] if predictions else 0,
    }


@app.get("/probabilities/title")
async def get_title_race():
    """Get title race probabilities, sorted by likelihood."""
    db = get_db()
    predictions = db.get_season_predictions()

    if not predictions:
        raise HTTPException(status_code=404, detail="No predictions available")

    title_probs = sorted(
        [{"team": p["team"], "probability": p["title_prob"]} for p in predictions],
        key=lambda x: x["probability"],
        reverse=True,
    )

    return {"title_race": title_probs[:10]}  # Top 10 contenders


@app.get("/probabilities/relegation")
async def get_relegation_battle():
    """Get relegation battle probabilities, sorted by risk."""
    db = get_db()
    predictions = db.get_season_predictions()

    if not predictions:
        raise HTTPException(status_code=404, detail="No predictions available")

    relegation_probs = sorted(
        [{"team": p["team"], "probability": p["relegation_prob"]} for p in predictions],
        key=lambda x: x["probability"],
        reverse=True,
    )

    return {"relegation_battle": relegation_probs[:10]}


# Counterfactual simulation

@app.post("/counterfactual")
async def run_counterfactual(scenarios: List[CounterfactualRequest]):
    """
    Run a counterfactual simulation.

    "What if City had beaten Burnley in week 3?"
    """
    api = FootballDataAPI()
    simulator = MonteCarloSimulator()

    try:
        all_fixtures = await api.get_all_fixtures()

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
    api = FootballDataAPI()

    try:
        results = await api.get_finished_matches()

        if matchweek:
            results = [r for r in results if r.matchweek == matchweek]

        return {
            "results": [r.to_dict() for r in results],
            "count": len(results),
        }
    except Exception as e:
        logger.error(f"Failed to fetch results: {e}")
        raise HTTPException(status_code=503, detail=str(e))


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
