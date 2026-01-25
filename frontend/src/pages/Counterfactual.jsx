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

import { getResults, getFixtures, runIRTCounterfactual, getIRT100MSimulation } from '../api';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';

export default function Counterfactual() {
  const [selectedMatches, setSelectedMatches] = React.useState([]);
  const [counterfactualResults, setCounterfactualResults] = React.useState(null);
  const [showUpcoming, setShowUpcoming] = React.useState(true); // Default to upcoming

  const resultsQuery = useQuery({
    queryKey: ['results'],
    queryFn: () => getResults(),
  });

  const fixturesQuery = useQuery({
    queryKey: ['fixtures', true],
    queryFn: () => getFixtures(true),
  });

  const baselineQuery = useQuery({
    queryKey: ['irt100m'],
    queryFn: getIRT100MSimulation,
  });

  const counterfactualMutation = useMutation({
    mutationFn: (scenarios) => runIRTCounterfactual(scenarios, 10000),
    onSuccess: (data) => {
      setCounterfactualResults(data);
    },
  });

  const pastMatches = resultsQuery.data?.results || [];
  const upcomingMatches = fixturesQuery.data?.fixtures || [];
  const baseline = baselineQuery.data?.teams || [];

  const handleMatchSelect = (match, isUpcoming) => {
    const matchKey = isUpcoming
      ? `${match.home_team}-${match.away_team}`
      : match.match_id;

    setSelectedMatches((prev) => {
      const existing = prev.find((m) => m.key === matchKey);
      if (existing) {
        return prev.filter((m) => m.key !== matchKey);
      }
      return [...prev, {
        key: matchKey,
        home_team: match.home_team,
        away_team: match.away_team,
        result: 'H',
        isUpcoming,
        matchweek: match.matchweek,
      }];
    });
    setCounterfactualResults(null);
  };

  const handleResultChange = (matchKey, result) => {
    setSelectedMatches((prev) =>
      prev.map((m) => (m.key === matchKey ? { ...m, result } : m))
    );
    setCounterfactualResults(null);
  };

  const handleSimulate = () => {
    if (selectedMatches.length === 0) return;
    // Convert to IRT counterfactual format
    const scenarios = selectedMatches.map((m) => ({
      home_team: m.home_team,
      away_team: m.away_team,
      result: m.result,
    }));
    counterfactualMutation.mutate(scenarios);
  };

  const handleReset = () => {
    setSelectedMatches([]);
    setCounterfactualResults(null);
  };

  // Group matches by matchweek
  const matchesByWeek = React.useMemo(() => {
    const grouped = {};
    const matchesToShow = showUpcoming ? upcomingMatches : pastMatches;
    matchesToShow.forEach((match) => {
      if (!grouped[match.matchweek]) {
        grouped[match.matchweek] = [];
      }
      grouped[match.matchweek].push(match);
    });
    return grouped;
  }, [pastMatches, upcomingMatches, showUpcoming]);

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

      {/* Toggle and Instructions */}
      <div className="flex flex-col sm:flex-row gap-4">
        {/* Toggle */}
        <div className="card flex-1">
          <div className="card-body">
            <div className="flex items-center space-x-4">
              <button
                onClick={() => { setShowUpcoming(true); setSelectedMatches([]); setCounterfactualResults(null); }}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  showUpcoming
                    ? 'bg-primary-500 text-white'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                Upcoming Matches
              </button>
              <button
                onClick={() => { setShowUpcoming(false); setSelectedMatches([]); setCounterfactualResults(null); }}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  !showUpcoming
                    ? 'bg-primary-500 text-white'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                Past Results
              </button>
            </div>
            <p className="text-sm text-slate-500 mt-2">
              {showUpcoming
                ? 'Predict upcoming results and see how they change the season outlook'
                : 'Change past results to explore alternate timelines'}
            </p>
          </div>
        </div>

        {/* Instructions */}
        <div className="card bg-amber-50 border-amber-200 flex-1">
          <div className="card-body">
            <div className="flex items-start space-x-3">
              <AlertCircle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-amber-800">
                <p className="font-medium">How to use:</p>
                <ol className="list-decimal list-inside mt-1 space-y-1">
                  <li>Select matches to {showUpcoming ? 'predict' : 'modify'}</li>
                  <li>Choose a result (Home win, Draw, or Away win)</li>
                  <li>Click "Simulate" to see the impact on season outcomes</li>
                </ol>
              </div>
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
            {(showUpcoming ? fixturesQuery.isLoading : resultsQuery.isLoading) ? (
              <LoadingSpinner />
            ) : Object.keys(matchesByWeek).length === 0 ? (
              <p className="text-slate-500 text-center py-4">
                {showUpcoming ? 'No upcoming matches found' : 'No past results found'}
              </p>
            ) : (
              <div className="space-y-4">
                {Object.entries(matchesByWeek)
                  .sort(([a], [b]) => showUpcoming ? Number(a) - Number(b) : Number(b) - Number(a))
                  .map(([week, weekMatches]) => (
                    <MatchWeekSection
                      key={week}
                      week={Number(week)}
                      matches={weekMatches}
                      selectedMatches={selectedMatches}
                      onMatchSelect={handleMatchSelect}
                      onResultChange={handleResultChange}
                      isUpcoming={showUpcoming}
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
              {showUpcoming ? 'Scenario Analysis' : 'The Butterfly Effect'}
            </h3>
            {counterfactualResults.scenarios_applied && (
              <ul className="text-sm text-primary-800 mb-2 list-disc list-inside">
                {counterfactualResults.scenarios_applied.map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ul>
            )}
            <p className="text-sm text-primary-700">
              {showUpcoming
                ? `Based on ${counterfactualResults.n_simulations?.toLocaleString()} simulations of the remaining season.`
                : `Changing ${selectedMatches.length} match${selectedMatches.length !== 1 ? 'es' : ''} ripples through the rest of the season.`}
            </p>
            {counterfactualResults.deltas && counterfactualResults.deltas.length > 0 && (
              <div className="mt-3 pt-3 border-t border-primary-200">
                <h4 className="text-sm font-medium text-primary-900 mb-2">Key Changes:</h4>
                <div className="space-y-1">
                  {counterfactualResults.deltas.map((d) => (
                    <div key={d.team} className="text-sm text-primary-700 flex items-center justify-between">
                      <span>{d.team}</span>
                      <span className={d.p_title_delta > 0 ? 'text-emerald-600' : d.p_title_delta < 0 ? 'text-red-600' : ''}>
                        Title: {d.p_title_delta > 0 ? '+' : ''}{d.p_title_delta.toFixed(2)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
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
  isUpcoming = false,
}) {
  // Expand first two weeks for upcoming, recent weeks for past
  const [isExpanded, setIsExpanded] = React.useState(
    isUpcoming ? week <= (matches[0]?.matchweek || 0) + 1 : week >= (matches[0]?.matchweek || 0) - 2
  );

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
            const matchKey = isUpcoming
              ? `${match.home_team}-${match.away_team}`
              : match.match_id;
            const selected = selectedMatches.find((m) => m.key === matchKey);

            // For past matches, show actual result
            const actualResult = !isUpcoming
              ? match.home_goals > match.away_goals
                ? 'H'
                : match.home_goals < match.away_goals
                ? 'A'
                : 'D'
              : null;

            return (
              <div
                key={matchKey}
                className={`p-3 ${selected ? 'bg-primary-50' : ''}`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      checked={!!selected}
                      onChange={() => onMatchSelect(match, isUpcoming)}
                      className="w-4 h-4 text-primary-600 rounded"
                    />
                    <div>
                      <div className="text-sm font-medium text-slate-900">
                        {match.home_team}
                        {!isUpcoming && ` ${match.home_goals} - ${match.away_goals}`}
                        {isUpcoming && ' vs '}
                        {!isUpcoming && ' '}
                        {match.away_team}
                      </div>
                      {!isUpcoming && (
                        <div className="text-xs text-slate-400">
                          Actual: {actualResult === 'H' ? 'Home win' : actualResult === 'A' ? 'Away win' : 'Draw'}
                        </div>
                      )}
                      {isUpcoming && match.date && (
                        <div className="text-xs text-slate-400">
                          {new Date(match.date).toLocaleDateString('en-GB', {
                            weekday: 'short',
                            day: 'numeric',
                            month: 'short',
                            hour: '2-digit',
                            minute: '2-digit'
                          })}
                        </div>
                      )}
                    </div>
                  </div>

                  {selected && (
                    <div className="flex items-center space-x-2">
                      <span className="text-xs text-slate-500">{isUpcoming ? 'Predict:' : 'Change to:'}</span>
                      <select
                        value={selected.result}
                        onChange={(e) => onResultChange(matchKey, e.target.value)}
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
  // baseline is from 100M simulations (array of {team, predicted_position, position_distribution})
  // counterfactual is from IRT counterfactual (has baseline and counterfactual objects)
  const data = React.useMemo(() => {
    if (!baseline || baseline.length === 0) return [];

    const cfBaseline = counterfactual?.baseline || {};
    const cfResults = counterfactual?.counterfactual || {};

    return baseline
      .map((team) => {
        const teamName = team.team;
        const cf = cfResults[teamName];
        const cfBase = cfBaseline[teamName];

        // Use counterfactual baseline if available, otherwise use 100M baseline
        const baselineTitle = cfBase?.p_title ?? (team.position_distribution?.[0]?.probability * 100) ?? 0;
        const baselinePos = cfBase?.predicted_position ?? team.predicted_position;

        return {
          team: teamName,
          baselinePos: baselinePos,
          baselineTitle: baselineTitle / 100, // Convert to decimal
          cfPos: cf?.predicted_position,
          cfTitle: cf?.p_title ? cf.p_title / 100 : null, // Convert to decimal
          posDiff: cf ? cf.predicted_position - baselinePos : 0,
          titleDiff: cf ? (cf.p_title - baselineTitle) / 100 : 0,
        };
      })
      .sort((a, b) => (counterfactual ? (a.cfPos || a.baselinePos) - (b.cfPos || b.baselinePos) : a.baselinePos - b.baselinePos));
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
                  to={`/teams/${encodeURIComponent(team.team)}`}
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
