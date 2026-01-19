import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { Calendar, ChevronLeft, ChevronRight, Clock } from 'lucide-react';

import { getWeekPredictions, getNextWeekPredictions, getResults } from '../api';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import ProbabilityBar from '../components/ProbabilityBar';

export default function Predictions() {
  const [selectedWeek, setSelectedWeek] = React.useState(1);

  // Get current week from first query
  const nextWeekQuery = useQuery({
    queryKey: ['predictions', 'next'],
    queryFn: getNextWeekPredictions,
  });

  // Set initial week to current matchweek once we know it
  React.useEffect(() => {
    if (nextWeekQuery.data?.week && selectedWeek === 1) {
      setSelectedWeek(nextWeekQuery.data.week);
    }
  }, [nextWeekQuery.data, selectedWeek]);

  const predictionsQuery = useQuery({
    queryKey: ['predictions', selectedWeek],
    queryFn: () => getWeekPredictions(selectedWeek),
    enabled: selectedWeek > 0,
  });

  const resultsQuery = useQuery({
    queryKey: ['results', selectedWeek],
    queryFn: () => getResults(selectedWeek),
    enabled: selectedWeek > 0,
  });

  const isLoading = predictionsQuery.isLoading;
  const predictions = predictionsQuery.data?.predictions || [];
  const results = resultsQuery.data?.results || [];

  // Create a map of results by home_team + away_team for quick lookup
  const resultsMap = new Map(
    results.map((r) => [`${r.home_team}-${r.away_team}`, r])
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Match Predictions</h1>
          <p className="text-slate-500 mt-1">
            Win/Draw/Loss probabilities for each match
          </p>
        </div>
      </div>

      {/* Week Selector */}
      <div className="flex items-center justify-center space-x-4">
        <button
          onClick={() => setSelectedWeek((w) => Math.max(1, w - 1))}
          disabled={selectedWeek <= 1}
          className="p-2 rounded-lg hover:bg-slate-100 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <ChevronLeft className="w-5 h-5" />
        </button>

        <div className="flex items-center space-x-2 px-4 py-2 bg-white rounded-lg border border-slate-200">
          <Calendar className="w-5 h-5 text-slate-400" />
          <span className="font-semibold text-slate-900">
            Matchweek {selectedWeek}
          </span>
        </div>

        <button
          onClick={() => setSelectedWeek((w) => Math.min(38, w + 1))}
          disabled={selectedWeek >= 38}
          className="p-2 rounded-lg hover:bg-slate-100 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <ChevronRight className="w-5 h-5" />
        </button>
      </div>

      {/* Quick Navigation */}
      <div className="flex flex-wrap justify-center gap-2">
        {[1, 5, 10, 15, 20, 25, 30, 35, 38].map((week) => (
          <button
            key={week}
            onClick={() => setSelectedWeek(week)}
            className={`px-3 py-1 text-sm rounded-full transition-colors ${
              selectedWeek === week
                ? 'bg-primary-500 text-white'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
            }`}
          >
            W{week}
          </button>
        ))}
      </div>

      {/* Predictions Grid */}
      {isLoading ? (
        <LoadingSpinner />
      ) : predictions.length === 0 ? (
        <div className="card">
          <div className="card-body text-center py-12">
            <Calendar className="w-12 h-12 text-slate-300 mx-auto mb-4" />
            <p className="text-slate-500">No predictions available for this week</p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {predictions.map((match) => {
            const result = resultsMap.get(`${match.home_team}-${match.away_team}`);
            return (
              <MatchCard
                key={match.match_id}
                match={match}
                result={result}
              />
            );
          })}
        </div>
      )}

      {/* Legend */}
      <div className="card">
        <div className="card-body">
          <h3 className="font-semibold text-slate-900 mb-3">Understanding Predictions</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="flex items-start space-x-3">
              <div className="w-4 h-4 bg-primary-500 rounded mt-0.5" />
              <div>
                <div className="font-medium text-slate-700">Home Win</div>
                <div className="text-slate-500">
                  Probability that the home team wins
                </div>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="w-4 h-4 bg-slate-400 rounded mt-0.5" />
              <div>
                <div className="font-medium text-slate-700">Draw</div>
                <div className="text-slate-500">
                  Probability of a draw
                </div>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <div className="w-4 h-4 bg-primary-300 rounded mt-0.5" />
              <div>
                <div className="font-medium text-slate-700">Away Win</div>
                <div className="text-slate-500">
                  Probability that the away team wins
                </div>
              </div>
            </div>
          </div>
          <p className="text-xs text-slate-400 mt-4">
            Confidence indicates model certainty based on team uncertainty (sigma).
            Lower confidence means we're less sure about both teams' true strength.
          </p>
        </div>
      </div>
    </div>
  );
}

function MatchCard({ match, result }) {
  const homeWin = match.home_win_prob * 100;
  const draw = match.draw_prob * 100;
  const awayWin = match.away_win_prob * 100;

  const favorite =
    homeWin > awayWin ? 'home' : awayWin > homeWin ? 'away' : 'draw';

  const hasResult = result !== undefined;
  const actualResult = hasResult
    ? result.home_goals > result.away_goals
      ? 'H'
      : result.home_goals < result.away_goals
      ? 'A'
      : 'D'
    : null;

  const predictionCorrect =
    hasResult &&
    ((actualResult === 'H' && homeWin > draw && homeWin > awayWin) ||
      (actualResult === 'D' && draw > homeWin && draw > awayWin) ||
      (actualResult === 'A' && awayWin > homeWin && awayWin > draw));

  return (
    <div
      className={`card overflow-hidden ${
        hasResult
          ? predictionCorrect
            ? 'ring-2 ring-emerald-200'
            : 'ring-2 ring-red-200'
          : ''
      }`}
    >
      <div className="card-body">
        {/* Teams */}
        <div className="flex justify-between items-center mb-4">
          <Link
            to={`/team/${encodeURIComponent(match.home_team)}`}
            className={`font-semibold hover:text-primary-600 ${
              favorite === 'home' ? 'text-slate-900' : 'text-slate-600'
            }`}
          >
            {match.home_team}
          </Link>
          <div className="text-center">
            {hasResult ? (
              <div className="text-lg font-bold text-slate-900">
                {result.home_goals} - {result.away_goals}
              </div>
            ) : (
              <span className="text-slate-400">vs</span>
            )}
          </div>
          <Link
            to={`/team/${encodeURIComponent(match.away_team)}`}
            className={`font-semibold hover:text-primary-600 text-right ${
              favorite === 'away' ? 'text-slate-900' : 'text-slate-600'
            }`}
          >
            {match.away_team}
          </Link>
        </div>

        {/* Probability Bar */}
        <div className="flex h-4 rounded-full overflow-hidden bg-slate-100">
          <div
            className={`transition-all ${
              hasResult && actualResult === 'H' ? 'bg-emerald-500' : 'bg-primary-500'
            }`}
            style={{ width: `${homeWin}%` }}
          />
          <div
            className={`transition-all ${
              hasResult && actualResult === 'D' ? 'bg-emerald-400' : 'bg-slate-400'
            }`}
            style={{ width: `${draw}%` }}
          />
          <div
            className={`transition-all ${
              hasResult && actualResult === 'A' ? 'bg-emerald-500' : 'bg-primary-300'
            }`}
            style={{ width: `${awayWin}%` }}
          />
        </div>

        {/* Percentages */}
        <div className="flex justify-between mt-2 text-sm">
          <div className="text-center">
            <div className={`font-semibold ${
              hasResult && actualResult === 'H' ? 'text-emerald-600' : 'text-primary-600'
            }`}>
              {homeWin.toFixed(0)}%
            </div>
            <div className="text-xs text-slate-400">Home</div>
          </div>
          <div className="text-center">
            <div className={`font-semibold ${
              hasResult && actualResult === 'D' ? 'text-emerald-600' : 'text-slate-500'
            }`}>
              {draw.toFixed(0)}%
            </div>
            <div className="text-xs text-slate-400">Draw</div>
          </div>
          <div className="text-center">
            <div className={`font-semibold ${
              hasResult && actualResult === 'A' ? 'text-emerald-600' : 'text-primary-400'
            }`}>
              {awayWin.toFixed(0)}%
            </div>
            <div className="text-xs text-slate-400">Away</div>
          </div>
        </div>

        {/* Confidence */}
        <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
          <span className="text-slate-400">
            Confidence: {(match.confidence * 100).toFixed(0)}%
          </span>
          <span className="text-slate-400">
            Delta: {match.delta?.toFixed(2)}
          </span>
        </div>

        {/* Result indicator */}
        {hasResult && (
          <div
            className={`mt-3 text-center text-xs font-medium py-1 rounded ${
              predictionCorrect
                ? 'bg-emerald-50 text-emerald-700'
                : 'bg-red-50 text-red-700'
            }`}
          >
            {predictionCorrect ? 'Prediction Correct' : 'Prediction Incorrect'}
          </div>
        )}
      </div>
    </div>
  );
}
