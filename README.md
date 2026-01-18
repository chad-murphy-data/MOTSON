# MOTSON v2

**M**odel **O**f **T**eam **S**trength **O**utcome **N**etwork

EPL prediction dashboard powered by Bayesian inference with cumulative calibration.

## Philosophy

> "Track distributions, not point estimates. Update on cumulative calibration, not individual surprises."

- **Stickiness**: Some teams (Man City) are historically stable; others (Burnley) are volatile
- **Gravity**: Historical pull toward where a team "belongs" (decays over the season)
- **Cumulative calibration**: Only update theta when systematically over/under-performing
- **Transparency**: Explain WHY predictions change

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Free API key from [football-data.org](https://www.football-data.org/client/register)

### Local Development

1. **Clone and setup backend:**
   ```bash
   cd motson-v2
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your FOOTBALL_DATA_API_KEY
   ```

3. **Start the backend:**
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

4. **In a new terminal, start the frontend:**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

5. **Open http://localhost:3000**

### Deploy to Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

Or manually:
1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set `FOOTBALL_DATA_API_KEY` environment variable
4. Deploy!

## Features

### Dashboard
- Current standings vs predicted final positions
- Title race probabilities
- Relegation battle tracker
- Next week's match predictions

### Team Deep Dive
- Theta (strength) trajectory over the season
- Cumulative calibration status
- Position probability distribution
- Model confidence and explanation

### Match Predictions
- Win/Draw/Loss probabilities for every match
- Historical accuracy tracking
- Confidence based on team uncertainty

### Season Outcomes
- Monte Carlo simulation (10,000 seasons)
- Title, Top 4, and relegation probabilities
- Position distribution charts

### Counterfactual Generator
- "What if City had beaten Burnley?"
- See butterfly effects on season outcomes
- Explore alternate realities

## Architecture

```
motson-v2/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Hyperparameters & settings
│   ├── models/
│   │   ├── team_state.py    # Data models
│   │   └── bayesian_engine.py  # Core prediction engine
│   ├── services/
│   │   ├── data_fetcher.py  # football-data.org API
│   │   ├── weekly_update.py # Update pipeline
│   │   └── monte_carlo.py   # Season simulation
│   └── database/
│       └── db.py            # SQLite persistence
├── frontend/
│   └── src/
│       ├── pages/           # Dashboard, Standings, etc.
│       └── components/      # Shared UI components
├── data/                    # Initial team parameters
└── render.yaml              # Render deployment config
```

## API Endpoints

```
GET  /health                 # API health check
GET  /standings/current      # Actual league standings
GET  /standings/predicted    # Model predictions
GET  /teams                  # List all teams
GET  /team/{name}            # Team detail
GET  /team/{name}/history    # Theta trajectory
GET  /predictions/week/{n}   # Match predictions
GET  /predictions/next       # Next week predictions
GET  /probabilities/season   # Season outcomes
POST /counterfactual         # What-if simulation
POST /admin/update           # Trigger weekly update
```

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `HOME_ADVANTAGE` | 0.30 | Theta boost for home team |
| `B0_DRAW` | 0.05 | Base draw probability |
| `UPDATE_THRESHOLD` | 1.0 | Z-score to trigger update |
| `LEARNING_RATE` | 0.10 | Base theta change per excess Z |
| `GRAVITY_DECAY_WEEKS` | 20 | Half-life of gravity pull |

## License

MIT
