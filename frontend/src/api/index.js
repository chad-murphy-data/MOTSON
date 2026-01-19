/**
 * MOTSON API Client
 */

// In production, VITE_API_URL points to the backend service
// In development, we use the Vite proxy to /api
const API_BASE = import.meta.env.VITE_API_URL || '';

async function fetchApi(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;

  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `API error: ${response.status}`);
  }

  return response.json();
}

// Health check
export const getHealth = () => fetchApi('/health');

// Standings
export const getCurrentStandings = () => fetchApi('/standings/current');
export const getPredictedStandings = () => fetchApi('/standings/predicted');

// Teams
export const getTeams = () => fetchApi('/teams');
export const getTeamDetail = (teamName) => fetchApi(`/team/${encodeURIComponent(teamName)}`);
export const getTeamHistory = (teamName) => fetchApi(`/team/${encodeURIComponent(teamName)}/history`);

// Predictions
export const getWeekPredictions = (week) => fetchApi(`/predictions/week/${week}`);
export const getNextWeekPredictions = () => fetchApi('/predictions/next');

// Season probabilities
export const getSeasonProbabilities = () => fetchApi('/probabilities/season');
export const getTitleRace = () => fetchApi('/probabilities/title');
export const getRelegationBattle = () => fetchApi('/probabilities/relegation');

// Historical data
export const getHistoricalPoints = () => fetchApi('/history/points');
export const getHistoricalStrength = () => fetchApi('/history/strength');
export const getHistoricalPositions = () => fetchApi('/history/positions');
export const getHistoricalTitleRace = () => fetchApi('/history/title-race');

// Results
export const getResults = (matchweek) =>
  fetchApi(`/results${matchweek ? `?matchweek=${matchweek}` : ''}`);

// Counterfactual
export const runCounterfactual = (scenarios) =>
  fetchApi('/counterfactual', {
    method: 'POST',
    body: JSON.stringify(scenarios),
  });

// Admin
export const triggerUpdate = () =>
  fetchApi('/admin/update', { method: 'POST' });
export const getUpdateExplanations = () => fetchApi('/admin/explanations');
export const getModelConfig = () => fetchApi('/admin/config');
export const getLastUpdate = () => fetchApi('/admin/last-update');

// IRT Model endpoints
export const getIRTTeams = () => fetchApi('/irt/teams');
export const getIRTTeamDetail = (teamName) => fetchApi(`/irt/team/${encodeURIComponent(teamName)}`);
export const getIRTTeamHistory = (teamName) => fetchApi(`/irt/team/${encodeURIComponent(teamName)}/history`);
export const getIRTAllHistory = () => fetchApi('/irt/history/all');
export const getIRTRankings = () => fetchApi('/irt/rankings');
export const getIRTMatchPrediction = (homeTeam, awayTeam) =>
  fetchApi(`/irt/predict/${encodeURIComponent(homeTeam)}/${encodeURIComponent(awayTeam)}`);

// IRT Season Simulation endpoints
export const getIRTSimulationCurrent = () => fetchApi('/irt/simulation/current');
export const getIRT100MSimulation = () => fetchApi('/irt/simulation/100m');
export const getIRTSimulationHistory = (team = null) =>
  fetchApi(`/irt/simulation/history${team ? `?team=${encodeURIComponent(team)}` : ''}`);
export const getIRTDistributions = (nSimulations = 100000) =>
  fetchApi(`/irt/simulation/distributions?n_simulations=${nSimulations}`);
export const getIRTFunStats = (nSimulations = 100000) =>
  fetchApi(`/irt/simulation/fun-stats?n_simulations=${nSimulations}`);
export const runIRTCounterfactual = (scenarios, nSimulations = 10000) =>
  fetchApi(`/irt/counterfactual?n_simulations=${nSimulations}`, {
    method: 'POST',
    body: JSON.stringify(scenarios),
  });
export const getIRTRivalries = (nSimulations = 100000) =>
  fetchApi(`/irt/simulation/rivalries?n_simulations=${nSimulations}`);
