/**
 * MOTSON API Client
 */

const API_BASE = import.meta.env.VITE_API_URL || '/api';

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
