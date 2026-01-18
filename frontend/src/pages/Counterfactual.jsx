import React from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import {
  Sparkles,
  Play,
  RotateCcw,
  ChevronDown,
  AlertCircle,
} from 'lucide-react';

import { getResults, runCounterfactual, getSeasonProbabilities } from '../api';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';

export default function Counterfactual() {
  const [selectedMatches, setSelectedMatches] = React.useState([]);
  const [counterfactualResults, setCounterfactualResults] = React.useState(null);

  const resultsQuery = useQuery({
    queryKey: ['results'],
    queryFn: () => getResults(),
  });

  const baselineQuery = useQuery({
    queryKey: ['seasonProbabilities'],
    queryFn: getSeasonProbabilities,
  });

  const counterfactualMutation = useMutation({
    mutationFn: runCounterfactual,
    onSuccess: (data) => {
      setCounterfactualResults(data.results);
    },
  });

  const matches = resultsQuery.data?.results || [];
  const baseline = baselineQuery.data?.predictions || [];

  const handleMatchSelect = (matchId) => {
    setSelectedMatches((prev) => {
      const existing = prev.find((m) => m.match_id === matchId);
      if (existing) {
        return prev.filter((m) => m.match_id !== matchId);
      }
      return [...prev, { match_id: matchId, result: 'H' }];
    });
    setCounterfactualResults(null);
  };

  const handleResultChange = (matchId, result) => {
    setSelectedMatches((prev) =>
      prev.map((m) => (m.match_id === matchId ? { ...m, result } : m))
    );
    setCounterfactualResults(null);
  };

  const handleSimulate = () => {
    if (selectedMatches.length === 0) return;
    counterfactualMutation.mutate(selectedMatches);
  };

  const handleReset = () => {
    setSelectedMatches([]);
    setCounterfactualResults(null);
  };

  // Group matches by matchweek
  const matchesByWeek = React.useMemo(() => {
    const grouped = {};
    matches.forEach((match) => {
      if (!grouped[match.matchweek]) {
        grouped[match.matchweek] = [];
      }
      grouped[match.matchweek].push(match);
    });
    return grouped;
  }, [matches]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 flex items-center">
            <Sparkles className="w-7 h-7 text-amber-500 mr-2" />
            What If?
          </h1>
          <p className="text-slate-500 mt-1">
            Explore alternate realities - what if results had gone differently?
          </p>
        </div>
        <div className="mt-4 md:mt-0 flex items-center space-x-3">
          <button
            onClick={handleReset}
            disabled={selectedMatches.length === 0}
            className="flex items-center space-x-2 px-4 py-2 border border-slate-200 rounded-lg hover:bg-slate-50 disabled:opacity-50"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Reset</span>
          </button>
          <button
            onClick={handleSimulate}
            disabled={selectedMatches.length === 0 || counterfactualMutation.isPending}
            className="flex items-center space-x-2 px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 disabled:opacity-50"
          >
            <Play className="w-4 h-4" />
            <span>
              {counterfactualMutation.isPending ? 'Simulating...' : 'Simulate'}
            </span>
          </button>
        </div>
      </div>

      {/* Instructions */}
      <div className="card bg-amber-50 border-amber-200">
        <div className="card-body">
          <div className="flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-amber-800">
              <p className="font-medium">How to use:</p>
              <ol className="list-decimal list-inside mt-1 space-y-1">
                <li>Select matches from the list below to modify</li>
                <li>Choose an alternate result (Home win, Draw, or Away win)</li>
                <li>Click "Simulate" to see how the season would unfold</li>
              </ol>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Match Selection */}
        <div className="card">
          <div className="card-header">
            <h2 className="font-semibold text-slate-900">Select Matches to Modify</h2>
            <p className="text-sm text-slate-500">
              {selectedMatches.length} match{selectedMatches.length !== 1 ? 'es' : ''} selected
            </p>
          </div>
          <div className="card-body max-h-[600px] overflow-y-auto">
            {resultsQuery.isLoading ? (
              <LoadingSpinner />
            ) : (
              <div className="space-y-4">
                {Object.entries(matchesByWeek)
                  .sort(([a], [b]) => Number(b) - Number(a))
                  .map(([week, weekMatches]) => (
                    <MatchWeekSection
                      key={week}
                      week={Number(week)}
                      matches={weekMatches}
                      selectedMatches={selectedMatches}
                      onMatchSelect={handleMatchSelect}
                      onResultChange={handleResultChange}
                    />
                  ))}
              </div>
            )}
          </div>
        </div>

        {/* Results Comparison */}
        <div className="card">
          <div className="card-header">
            <h2 className="font-semibold text-slate-900">
              {counterfactualResults ? 'Alternate Reality' : 'Current Reality'}
            </h2>
            <p className="text-sm text-slate-500">
              {counterfactualResults
                ? 'Season outcomes with your changes'
                : 'Actual season projections'}
            </p>
          </div>
          <div className="card-body">
            {baselineQuery.isLoading ? (
              <LoadingSpinner />
            ) : (
              <ComparisonTable
                baseline={baseline}
                counterfactual={counterfactualResults}
              />
            )}
          </div>
        </div>
      </div>

      {/* Butterfly Effect */}
      {counterfactualResults && (
        <div className="card bg-gradient-to-br from-primary-50 to-white">
          <div className="card-body">
            <h3 className="font-semibold text-primary-900 mb-2">
              The Butterfly Effect
            </h3>
            <p className="text-sm text-primary-700">
              Changing {selectedMatches.length} match{selectedMatches.length !== 1 ? 'es' : ''}{' '}
              ripples through the rest of the season. The Monte Carlo simulation
              shows how different those early results could make the final standings.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function MatchWeekSection({
  week,
  matches,
  selectedMatches,
  onMatchSelect,
  onResultChange,
}) {
  const [isExpanded, setIsExpanded] = React.useState(week >= matches[0]?.matchweek - 2);

  return (
    <div className="border border-slate-200 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 bg-slate-50 flex items-center justify-between hover:bg-slate-100"
      >
        <span className="font-medium text-slate-700">Matchweek {week}</span>
        <ChevronDown
          className={`w-4 h-4 text-slate-400 transition-transform ${
            isExpanded ? 'rotate-180' : ''
          }`}
        />
      </button>
      {isExpanded && (
        <div className="divide-y divide-slate-100">
          {matches.map((match) => {
            const selected = selectedMatches.find((m) => m.match_id === match.match_id);
            const actualResult =
              match.home_goals > match.away_goals
                ? 'H'
                : match.home_goals < match.away_goals
                ? 'A'
                : 'D';

            return (
              <div
                key={match.match_id}
                className={`p-3 ${selected ? 'bg-primary-50' : ''}`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={!!selected}
                      onChange={() => onMatchSelect(match.match_id)}
                      className="w-4 h-4 text-primary-600 rounded"
                    />
                    <div>
                      <div className="text-sm font-medium text-slate-900">
                        {match.home_team} {match.home_goals} - {match.away_goals}{' '}
                        {match.away_team}
                      </div>
                      <div className="text-xs text-slate-400">
                        Actual: {actualResult === 'H' ? 'Home win' : actualResult === 'A' ? 'Away win' : 'Draw'}
                      </div>
                    </div>
                  </div>

                  {selected && (
                    <div className="flex items-center space-x-2">
                      <span className="text-xs text-slate-500">Change to:</span>
                      <select
                        value={selected.result}
                        onChange={(e) => onResultChange(match.match_id, e.target.value)}
                        className="text-sm border border-slate-200 rounded px-2 py-1"
                      >
                        <option value="H">Home win</option>
                        <option value="D">Draw</option>
                        <option value="A">Away win</option>
                      </select>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ComparisonTable({ baseline, counterfactual }) {
  // Create comparison data
  const data = React.useMemo(() => {
    const baselineMap = new Map(baseline.map((t) => [t.team, t]));

    return baseline
      .map((team) => {
        const cf = counterfactual?.[team.team];
        return {
          team: team.team,
          baselinePos: team.expected_position,
          baselineTitle: team.title_prob,
          baselineRel: team.relegation_prob,
          cfPos: cf?.expected_position,
          cfTitle: cf?.title_prob,
          cfRel: cf?.relegation_prob,
          posDiff: cf ? cf.expected_position - team.expected_position : 0,
          titleDiff: cf ? cf.title_prob - team.title_prob : 0,
        };
      })
      .sort((a, b) => (counterfactual ? a.cfPos - b.cfPos : a.baselinePos - b.baselinePos));
  }, [baseline, counterfactual]);

  return (
    <div className="overflow-x-auto">
      <table className="table-pro text-sm">
        <thead>
          <tr>
            <th>Team</th>
            <th className="text-center">Position</th>
            {counterfactual && <th className="text-center">Change</th>}
            <th className="text-center">Title %</th>
          </tr>
        </thead>
        <tbody>
          {data.map((team) => (
            <tr key={team.team}>
              <td>
                <Link
                  to={`/team/${encodeURIComponent(team.team)}`}
                  className="font-medium text-slate-900 hover:text-primary-600"
                >
                  {team.team}
                </Link>
              </td>
              <td className="text-center">
                {counterfactual ? (
                  <span className="font-semibold text-slate-900">
                    {team.cfPos?.toFixed(1)}
                  </span>
                ) : (
                  <span className="text-slate-600">{team.baselinePos?.toFixed(1)}</span>
                )}
              </td>
              {counterfactual && (
                <td className="text-center">
                  <DiffIndicator value={team.posDiff} inverted />
                </td>
              )}
              <td className="text-center">
                {counterfactual ? (
                  <span
                    className={
                      team.titleDiff > 0.01
                        ? 'text-emerald-600 font-semibold'
                        : team.titleDiff < -0.01
                        ? 'text-red-600 font-semibold'
                        : 'text-slate-600'
                    }
                  >
                    {((team.cfTitle || 0) * 100).toFixed(1)}%
                  </span>
                ) : (
                  <span className="text-slate-600">
                    {(team.baselineTitle * 100).toFixed(1)}%
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DiffIndicator({ value, inverted = false }) {
  if (Math.abs(value) < 0.1) {
    return <span className="text-slate-400">-</span>;
  }

  const isPositive = inverted ? value < 0 : value > 0;
  const color = isPositive ? 'text-emerald-600' : 'text-red-600';
  const sign = value > 0 ? '+' : '';

  return (
    <span className={`font-medium ${color}`}>
      {sign}{value.toFixed(1)}
    </span>
  );
}
