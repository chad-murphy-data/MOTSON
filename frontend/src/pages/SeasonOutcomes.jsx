import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { Trophy, Medal, AlertTriangle, TrendingUp } from 'lucide-react';

import { getSeasonProbabilities, getTitleRace, getRelegationBattle } from '../api';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import ProbabilityBar from '../components/ProbabilityBar';

export default function SeasonOutcomes() {
  const [view, setView] = React.useState('overview'); // 'overview', 'title', 'top4', 'relegation'

  const probsQuery = useQuery({
    queryKey: ['seasonProbabilities'],
    queryFn: getSeasonProbabilities,
  });

  const isLoading = probsQuery.isLoading;
  const predictions = probsQuery.data?.predictions || [];

  // Sort by different criteria based on view
  const sortedData = React.useMemo(() => {
    const sorted = [...predictions];
    switch (view) {
      case 'title':
        return sorted.sort((a, b) => b.title_prob - a.title_prob);
      case 'top4':
        return sorted.sort((a, b) => b.top4_prob - a.top4_prob);
      case 'relegation':
        return sorted.sort((a, b) => b.relegation_prob - a.relegation_prob);
      default:
        return sorted.sort((a, b) => a.expected_position - b.expected_position);
    }
  }, [predictions, view]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Season Outcomes</h1>
          <p className="text-slate-500 mt-1">
            Monte Carlo simulation of {10000} possible seasons
          </p>
        </div>
      </div>

      {/* View Toggle */}
      <div className="flex flex-wrap gap-2">
        <ViewButton
          active={view === 'overview'}
          onClick={() => setView('overview')}
          icon={TrendingUp}
          label="Overview"
        />
        <ViewButton
          active={view === 'title'}
          onClick={() => setView('title')}
          icon={Trophy}
          label="Title Race"
          color="amber"
        />
        <ViewButton
          active={view === 'top4'}
          onClick={() => setView('top4')}
          icon={Medal}
          label="Top 4"
          color="blue"
        />
        <ViewButton
          active={view === 'relegation'}
          onClick={() => setView('relegation')}
          icon={AlertTriangle}
          label="Relegation"
          color="red"
        />
      </div>

      {isLoading ? (
        <LoadingSpinner />
      ) : (
        <>
          {/* Chart */}
          <div className="card">
            <div className="card-header">
              <h2 className="font-semibold text-slate-900">
                {view === 'title'
                  ? 'Title Probability'
                  : view === 'top4'
                  ? 'Top 4 Probability'
                  : view === 'relegation'
                  ? 'Relegation Risk'
                  : 'Expected Final Position'}
              </h2>
            </div>
            <div className="card-body">
              <ResponsiveContainer width="100%" height={400}>
                <BarChart
                  data={sortedData.slice(0, view === 'overview' ? 20 : 10)}
                  layout="vertical"
                  margin={{ left: 100, right: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis
                    type="number"
                    domain={view === 'overview' ? [1, 20] : [0, 100]}
                    tick={{ fontSize: 12, fill: '#64748b' }}
                    tickFormatter={view === 'overview' ? undefined : (v) => `${v}%`}
                  />
                  <YAxis
                    type="category"
                    dataKey="team"
                    tick={{ fontSize: 12, fill: '#64748b' }}
                    width={90}
                  />
                  <Tooltip
                    formatter={(value) =>
                      view === 'overview'
                        ? [value.toFixed(1), 'Expected Position']
                        : [`${(value * 100).toFixed(1)}%`, 'Probability']
                    }
                    contentStyle={{
                      backgroundColor: 'white',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                    }}
                  />
                  <Bar
                    dataKey={
                      view === 'title'
                        ? 'title_prob'
                        : view === 'top4'
                        ? 'top4_prob'
                        : view === 'relegation'
                        ? 'relegation_prob'
                        : 'expected_position'
                    }
                    radius={[0, 4, 4, 0]}
                  >
                    {sortedData.slice(0, view === 'overview' ? 20 : 10).map((entry, index) => (
                      <Cell
                        key={entry.team}
                        fill={getCellColor(view, entry, index)}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Detailed Table */}
          <div className="card">
            <div className="card-header">
              <h2 className="font-semibold text-slate-900">Full Breakdown</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="table-pro">
                <thead>
                  <tr>
                    <th>Team</th>
                    <th className="text-center">Exp Pos</th>
                    <th className="text-center">Title</th>
                    <th className="text-center">Top 4</th>
                    <th className="text-center">Top 6</th>
                    <th className="text-center">Relegation</th>
                    <th className="text-center">Exp Pts</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedData.map((team) => (
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
                        <span className="font-semibold text-slate-900">
                          {team.expected_position?.toFixed(1)}
                        </span>
                        <span className="text-xs text-slate-400 ml-1">
                          ±{team.position_std?.toFixed(1)}
                        </span>
                      </td>
                      <td className="text-center">
                        <ProbabilityCell value={team.title_prob} color="amber" />
                      </td>
                      <td className="text-center">
                        <ProbabilityCell value={team.top4_prob} color="blue" />
                      </td>
                      <td className="text-center">
                        <ProbabilityCell value={team.top6_prob} color="emerald" />
                      </td>
                      <td className="text-center">
                        <ProbabilityCell value={team.relegation_prob} color="red" />
                      </td>
                      <td className="text-center text-slate-600">
                        {team.expected_points?.toFixed(0)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Methodology */}
          <div className="card bg-slate-50">
            <div className="card-body">
              <h3 className="font-semibold text-slate-900 mb-2">How It Works</h3>
              <p className="text-sm text-slate-600">
                We simulate the remaining season 10,000 times. Each simulation
                uses the current team strengths (theta) to generate match
                outcomes probabilistically. The percentages show how often each
                outcome occurred across all simulations. A 25% title probability
                means that team won the league in 2,500 of 10,000 simulated seasons.
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function ViewButton({ active, onClick, icon: Icon, label, color = 'primary' }) {
  const colorClasses = {
    primary: active ? 'bg-primary-500 text-white' : 'bg-white text-slate-600 hover:bg-slate-50',
    amber: active ? 'bg-amber-500 text-white' : 'bg-white text-slate-600 hover:bg-slate-50',
    blue: active ? 'bg-blue-500 text-white' : 'bg-white text-slate-600 hover:bg-slate-50',
    red: active ? 'bg-red-500 text-white' : 'bg-white text-slate-600 hover:bg-slate-50',
  };

  return (
    <button
      onClick={onClick}
      className={`flex items-center space-x-2 px-4 py-2 rounded-lg border border-slate-200 transition-colors ${
        colorClasses[color]
      }`}
    >
      <Icon className="w-4 h-4" />
      <span className="font-medium">{label}</span>
    </button>
  );
}

function ProbabilityCell({ value, color }) {
  if (!value || value < 0.001) {
    return <span className="text-slate-300">-</span>;
  }

  const pct = value * 100;
  const colorClass = {
    amber: pct > 10 ? 'text-amber-600 font-semibold' : 'text-amber-400',
    blue: pct > 50 ? 'text-blue-600 font-semibold' : 'text-blue-400',
    emerald: pct > 50 ? 'text-emerald-600 font-semibold' : 'text-emerald-400',
    red: pct > 10 ? 'text-red-600 font-semibold' : 'text-red-400',
  }[color];

  return <span className={colorClass}>{pct.toFixed(1)}%</span>;
}

function getCellColor(view, entry, index) {
  if (view === 'overview') {
    // Color by expected position
    if (entry.expected_position <= 4) return '#3b82f6'; // blue
    if (entry.expected_position <= 6) return '#10b981'; // emerald
    if (entry.expected_position >= 18) return '#ef4444'; // red
    return '#64748b'; // slate
  }

  if (view === 'title') {
    const intensity = Math.min(1, entry.title_prob * 3);
    return `rgba(245, 158, 11, ${0.3 + intensity * 0.7})`; // amber
  }

  if (view === 'top4') {
    const intensity = Math.min(1, entry.top4_prob);
    return `rgba(59, 130, 246, ${0.3 + intensity * 0.7})`; // blue
  }

  if (view === 'relegation') {
    const intensity = Math.min(1, entry.relegation_prob * 2);
    return `rgba(239, 68, 68, ${0.3 + intensity * 0.7})`; // red
  }

  return '#3b82f6';
}
