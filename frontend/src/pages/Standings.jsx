import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { Trophy, TrendingUp, TrendingDown, Minus, ArrowRight } from 'lucide-react';

import { getCurrentStandings, getPredictedStandings } from '../api';
import PositionBadge from '../components/PositionBadge';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';

export default function Standings() {
  const [view, setView] = React.useState('actual'); // 'actual' or 'predicted'

  const actualQuery = useQuery({
    queryKey: ['standings'],
    queryFn: getCurrentStandings,
  });

  const predictedQuery = useQuery({
    queryKey: ['predictedStandings'],
    queryFn: getPredictedStandings,
  });

  const isLoading = actualQuery.isLoading || predictedQuery.isLoading;
  const error = actualQuery.error || predictedQuery.error;

  // Combine actual and predicted for comparison
  const combinedData = React.useMemo(() => {
    if (!actualQuery.data?.standings || !predictedQuery.data?.standings) return [];

    const predicted = new Map(
      predictedQuery.data.standings.map((p) => [p.team, p])
    );

    return actualQuery.data.standings.map((team) => {
      const pred = predicted.get(team.team);
      return {
        ...team,
        predictedPosition: pred?.expected_position || team.position,
        titleProb: pred?.title_prob || 0,
        top4Prob: pred?.top4_prob || 0,
        relegationProb: pred?.relegation_prob || 0,
        expectedPoints: pred?.expected_points || team.points,
      };
    });
  }, [actualQuery.data, predictedQuery.data]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">League Standings</h1>
          <p className="text-slate-500 mt-1">
            Current table vs predicted final positions
          </p>
        </div>

        {/* View Toggle */}
        <div className="mt-4 md:mt-0 flex items-center bg-slate-100 rounded-lg p-1">
          <button
            onClick={() => setView('actual')}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              view === 'actual'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Current
          </button>
          <button
            onClick={() => setView('predicted')}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              view === 'predicted'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Predicted
          </button>
          <button
            onClick={() => setView('comparison')}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              view === 'comparison'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Comparison
          </button>
        </div>
      </div>

      {isLoading ? (
        <LoadingSpinner />
      ) : error ? (
        <ErrorMessage message="Failed to load standings" />
      ) : view === 'actual' ? (
        <ActualStandingsTable standings={actualQuery.data?.standings || []} />
      ) : view === 'predicted' ? (
        <PredictedStandingsTable standings={predictedQuery.data?.standings || []} />
      ) : (
        <ComparisonTable data={combinedData} />
      )}

      {/* Legend */}
      <div className="flex flex-wrap gap-4 text-sm text-slate-500">
        <div className="flex items-center">
          <div className="w-3 h-3 bg-blue-500 rounded-full mr-2" />
          Champions League (1-4)
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-emerald-500 rounded-full mr-2" />
          Europa League (5-6)
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 bg-red-500 rounded-full mr-2" />
          Relegation (18-20)
        </div>
      </div>
    </div>
  );
}

function ActualStandingsTable({ standings }) {
  return (
    <div className="card overflow-hidden">
      <div className="overflow-x-auto">
        <table className="table-pro">
          <thead>
            <tr>
              <th className="w-16">Pos</th>
              <th>Team</th>
              <th className="text-center">P</th>
              <th className="text-center">W</th>
              <th className="text-center">D</th>
              <th className="text-center">L</th>
              <th className="text-center">GF</th>
              <th className="text-center">GA</th>
              <th className="text-center">GD</th>
              <th className="text-center">Pts</th>
            </tr>
          </thead>
          <tbody>
            {standings.map((team) => (
              <tr
                key={team.team}
                className={getRowClass(team.position)}
              >
                <td>
                  <PositionBadge position={team.position} />
                </td>
                <td>
                  <Link
                    to={`/team/${encodeURIComponent(team.team)}`}
                    className="font-medium text-slate-900 hover:text-primary-600"
                  >
                    {team.team}
                  </Link>
                </td>
                <td className="text-center text-slate-600">{team.played}</td>
                <td className="text-center text-slate-600">{team.won}</td>
                <td className="text-center text-slate-600">{team.drawn}</td>
                <td className="text-center text-slate-600">{team.lost}</td>
                <td className="text-center text-slate-600">{team.goals_for}</td>
                <td className="text-center text-slate-600">{team.goals_against}</td>
                <td className="text-center font-medium">
                  <span
                    className={
                      team.goal_difference > 0
                        ? 'text-emerald-600'
                        : team.goal_difference < 0
                        ? 'text-red-600'
                        : 'text-slate-600'
                    }
                  >
                    {team.goal_difference > 0 ? '+' : ''}
                    {team.goal_difference}
                  </span>
                </td>
                <td className="text-center">
                  <span className="font-bold text-slate-900">{team.points}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function PredictedStandingsTable({ standings }) {
  // Sort by expected position
  const sorted = [...standings].sort((a, b) => a.expected_position - b.expected_position);

  return (
    <div className="card overflow-hidden">
      <div className="overflow-x-auto">
        <table className="table-pro">
          <thead>
            <tr>
              <th className="w-16">Pos</th>
              <th>Team</th>
              <th className="text-center">Exp Pts</th>
              <th className="text-center">Title %</th>
              <th className="text-center">Top 4 %</th>
              <th className="text-center">Top 6 %</th>
              <th className="text-center">Rel %</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((team, idx) => (
              <tr
                key={team.team}
                className={getRowClass(idx + 1)}
              >
                <td>
                  <PositionBadge position={idx + 1} />
                </td>
                <td>
                  <Link
                    to={`/team/${encodeURIComponent(team.team)}`}
                    className="font-medium text-slate-900 hover:text-primary-600"
                  >
                    {team.team}
                  </Link>
                </td>
                <td className="text-center">
                  <span className="font-semibold text-slate-900">
                    {team.expected_points?.toFixed(1)}
                  </span>
                  <span className="text-xs text-slate-400 ml-1">
                    ±{team.points_std?.toFixed(1)}
                  </span>
                </td>
                <td className="text-center">
                  <ProbCell value={team.title_prob} color="amber" />
                </td>
                <td className="text-center">
                  <ProbCell value={team.top4_prob} color="blue" />
                </td>
                <td className="text-center">
                  <ProbCell value={team.top6_prob} color="emerald" />
                </td>
                <td className="text-center">
                  <ProbCell value={team.relegation_prob} color="red" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ComparisonTable({ data }) {
  return (
    <div className="card overflow-hidden">
      <div className="overflow-x-auto">
        <table className="table-pro">
          <thead>
            <tr>
              <th className="w-16">Now</th>
              <th>Team</th>
              <th className="text-center">Pts</th>
              <th className="text-center">Pred Pos</th>
              <th className="text-center">Diff</th>
              <th className="text-center">Exp Pts</th>
            </tr>
          </thead>
          <tbody>
            {data.map((team) => {
              const diff = team.position - Math.round(team.predictedPosition);
              return (
                <tr key={team.team} className={getRowClass(team.position)}>
                  <td>
                    <PositionBadge position={team.position} />
                  </td>
                  <td>
                    <Link
                      to={`/team/${encodeURIComponent(team.team)}`}
                      className="font-medium text-slate-900 hover:text-primary-600"
                    >
                      {team.team}
                    </Link>
                  </td>
                  <td className="text-center font-semibold text-slate-900">
                    {team.points}
                  </td>
                  <td className="text-center">
                    <span className="text-slate-600">
                      {team.predictedPosition?.toFixed(1)}
                    </span>
                  </td>
                  <td className="text-center">
                    <DiffBadge diff={diff} />
                  </td>
                  <td className="text-center text-slate-600">
                    {team.expectedPoints?.toFixed(1)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ProbCell({ value, color }) {
  if (!value || value < 0.001) {
    return <span className="text-slate-300">-</span>;
  }

  const pct = (value * 100).toFixed(1);
  const colorClass = {
    amber: 'text-amber-600',
    blue: 'text-blue-600',
    emerald: 'text-emerald-600',
    red: 'text-red-600',
  }[color];

  return (
    <span className={`font-medium ${colorClass}`}>
      {pct}%
    </span>
  );
}

function DiffBadge({ diff }) {
  if (diff === 0) {
    return (
      <span className="inline-flex items-center text-slate-400">
        <Minus className="w-4 h-4" />
      </span>
    );
  }

  if (diff > 0) {
    // Team is higher (better) than predicted
    return (
      <span className="inline-flex items-center text-emerald-600">
        <TrendingUp className="w-4 h-4 mr-1" />
        {Math.abs(diff)}
      </span>
    );
  }

  // Team is lower (worse) than predicted
  return (
    <span className="inline-flex items-center text-red-600">
      <TrendingDown className="w-4 h-4 mr-1" />
      {Math.abs(diff)}
    </span>
  );
}

function getRowClass(position) {
  if (position <= 4) return 'bg-blue-50/50';
  if (position <= 6) return 'bg-emerald-50/50';
  if (position >= 18) return 'bg-red-50/50';
  return '';
}
