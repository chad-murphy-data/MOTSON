import React from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { TrendingUp, Activity, Grid3X3, Trophy } from 'lucide-react';

import {
  getHistoricalPoints,
  getHistoricalStrength,
  getHistoricalPositions,
  getHistoricalTitleRace,
} from '../api';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';

// Team colors for the charts
const TEAM_COLORS = {
  'Arsenal': '#EF0107',
  'Aston Villa': '#95BFE5',
  'Bournemouth': '#DA291C',
  'Brentford': '#E30613',
  'Brighton': '#0057B8',
  'Chelsea': '#034694',
  'Crystal Palace': '#1B458F',
  'Everton': '#003399',
  'Fulham': '#000000',
  'Ipswich': '#0044AA',
  'Leicester': '#003090',
  'Liverpool': '#C8102E',
  'Man City': '#6CABDD',
  'Man United': '#DA291C',
  'Newcastle': '#241F20',
  'Nottingham Forest': '#DD0000',
  "Nott'm Forest": '#DD0000',
  'Southampton': '#D71920',
  'Spurs': '#132257',
  'Tottenham': '#132257',
  'West Ham': '#7A263A',
  'Wolves': '#FDB913',
};

const getTeamColor = (team) => {
  return TEAM_COLORS[team] || '#6366f1';
};

export default function Analytics() {
  const [view, setView] = React.useState('title'); // 'title', 'points', 'strength', 'heatmap'
  const [selectedTeams, setSelectedTeams] = React.useState([]);

  const pointsQuery = useQuery({
    queryKey: ['historicalPoints'],
    queryFn: getHistoricalPoints,
  });

  const strengthQuery = useQuery({
    queryKey: ['historicalStrength'],
    queryFn: getHistoricalStrength,
  });

  const positionsQuery = useQuery({
    queryKey: ['historicalPositions'],
    queryFn: getHistoricalPositions,
  });

  const titleRaceQuery = useQuery({
    queryKey: ['historicalTitleRace'],
    queryFn: getHistoricalTitleRace,
  });

  const isLoading = view === 'title' ? titleRaceQuery.isLoading
    : view === 'points' ? pointsQuery.isLoading
    : view === 'strength' ? strengthQuery.isLoading
    : positionsQuery.isLoading;

  const error = view === 'title' ? titleRaceQuery.error
    : view === 'points' ? pointsQuery.error
    : view === 'strength' ? strengthQuery.error
    : positionsQuery.error;

  // Get all available teams
  const allTeams = React.useMemo(() => {
    const data = view === 'title' ? titleRaceQuery.data?.history
      : view === 'points' ? pointsQuery.data?.history
      : view === 'strength' ? strengthQuery.data?.history
      : null;
    return data ? Object.keys(data).sort() : [];
  }, [view, titleRaceQuery.data, pointsQuery.data, strengthQuery.data]);

  // Initialize with top 3 teams if none selected
  React.useEffect(() => {
    if (allTeams.length > 0 && selectedTeams.length === 0) {
      // Select the title contenders by default
      const defaultTeams = ['Arsenal', 'Liverpool', 'Man City']
        .filter(t => allTeams.includes(t));
      setSelectedTeams(defaultTeams);
    }
  }, [allTeams, selectedTeams]);

  const toggleTeam = (team) => {
    setSelectedTeams(prev =>
      prev.includes(team)
        ? prev.filter(t => t !== team)
        : [...prev, team]
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Analytics</h1>
          <p className="text-slate-500 mt-1">
            Historical trends and position probability distributions
          </p>
        </div>
      </div>

      {/* View Toggle */}
      <div className="flex flex-wrap gap-2">
        <ViewButton
          active={view === 'title'}
          onClick={() => setView('title')}
          icon={Trophy}
          label="Title Race"
          color="gold"
        />
        <ViewButton
          active={view === 'points'}
          onClick={() => setView('points')}
          icon={TrendingUp}
          label="Predicted Points"
        />
        <ViewButton
          active={view === 'strength'}
          onClick={() => setView('strength')}
          icon={Activity}
          label="Team Strength"
        />
        <ViewButton
          active={view === 'heatmap'}
          onClick={() => setView('heatmap')}
          icon={Grid3X3}
          label="Position Heatmap"
          color="red"
        />
      </div>

      {isLoading ? (
        <LoadingSpinner />
      ) : error ? (
        <div className="card">
          <div className="card-body text-center py-8">
            <p className="text-slate-500">
              Run "Update Predictions" on the dashboard to generate historical data
            </p>
          </div>
        </div>
      ) : view === 'heatmap' ? (
        <PositionHeatmap data={positionsQuery.data} />
      ) : (
        <>
          {/* Team Selector */}
          <div className="card">
            <div className="card-header">
              <h2 className="font-semibold text-slate-900">Select Teams</h2>
            </div>
            <div className="card-body">
              <div className="flex flex-wrap gap-2">
                {allTeams.map(team => (
                  <button
                    key={team}
                    onClick={() => toggleTeam(team)}
                    className={`px-3 py-1.5 text-sm rounded-full transition-colors border ${
                      selectedTeams.includes(team)
                        ? 'text-white border-transparent'
                        : 'bg-white text-slate-600 border-slate-200 hover:bg-slate-50'
                    }`}
                    style={selectedTeams.includes(team) ? { backgroundColor: getTeamColor(team) } : {}}
                  >
                    {team}
                  </button>
                ))}
              </div>
              <div className="mt-3 flex gap-2">
                <button
                  onClick={() => setSelectedTeams(allTeams)}
                  className="text-xs text-primary-600 hover:text-primary-700"
                >
                  Select All
                </button>
                <button
                  onClick={() => setSelectedTeams([])}
                  className="text-xs text-slate-500 hover:text-slate-600"
                >
                  Clear All
                </button>
              </div>
            </div>
          </div>

          {/* Chart */}
          {view === 'title' ? (
            <TitleRaceChart
              data={titleRaceQuery.data}
              selectedTeams={selectedTeams}
            />
          ) : view === 'points' ? (
            <PointsChart
              data={pointsQuery.data}
              selectedTeams={selectedTeams}
            />
          ) : (
            <StrengthChart
              data={strengthQuery.data}
              selectedTeams={selectedTeams}
            />
          )}
        </>
      )}

      {/* Methodology */}
      <div className="card bg-slate-50">
        <div className="card-body">
          <h3 className="font-semibold text-slate-900 mb-2">
            {view === 'title' && 'Title Probability Trajectories'}
            {view === 'points' && 'Predicted vs Actual Points'}
            {view === 'strength' && 'Team Strength (Theta)'}
            {view === 'heatmap' && 'Position Probability Heatmap'}
          </h3>
          <p className="text-sm text-slate-600">
            {view === 'title' &&
              'Shows how title-winning probability has evolved for each team throughout the season. Probabilities are calibrated using historical lead survival rates - teams with large leads may see their raw Monte Carlo probabilities adjusted downward to reflect that historically, big leads don\'t hold as often as pure probability suggests.'}
            {view === 'points' &&
              'Shows how each team\'s predicted end-of-season points total has evolved week by week, compared to their actual accumulated points. Large gaps indicate the model expects significant regression or improvement.'}
            {view === 'strength' &&
              'Theta represents team strength on a standardized scale. Higher values mean stronger teams. The model updates theta based on cumulative calibration - only when a team systematically over/under-performs, not for individual match surprises.'}
            {view === 'heatmap' &&
              'Each cell shows the probability of a team finishing in that position. Darker red indicates higher probability. Teams are ordered by expected final position.'}
          </p>
        </div>
      </div>
    </div>
  );
}

function ViewButton({ active, onClick, icon: Icon, label, color = 'primary' }) {
  const colorClasses = {
    primary: active ? 'bg-primary-500 text-white' : 'bg-white text-slate-600 hover:bg-slate-50',
    red: active ? 'bg-red-500 text-white' : 'bg-white text-slate-600 hover:bg-slate-50',
    gold: active ? 'bg-amber-500 text-white' : 'bg-white text-slate-600 hover:bg-slate-50',
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

function PointsChart({ data, selectedTeams }) {
  if (!data?.history || !data?.weeks?.length) {
    return (
      <div className="card">
        <div className="card-body text-center py-8">
          <p className="text-slate-500">No historical data available yet</p>
        </div>
      </div>
    );
  }

  const hasMultipleWeeks = data.weeks.length > 1;

  // Transform data for recharts
  const chartData = data.weeks.map(week => {
    const point = { week };
    selectedTeams.forEach(team => {
      const teamData = data.history[team]?.find(d => d.week === week);
      if (teamData) {
        point[`${team}_expected`] = teamData.expected_points;
        point[`${team}_actual`] = teamData.actual_points;
      }
    });
    return point;
  });

  if (!hasMultipleWeeks) {
    return (
      <div className="card">
        <div className="card-body text-center py-8">
          <p className="text-slate-500 mb-2">
            Only 1 week of data available (Week {data.weeks[0]})
          </p>
          <p className="text-xs text-slate-400">
            Historical trends will appear as more matchweeks are tracked.
            The scheduled update runs automatically after each matchday.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="font-semibold text-slate-900">Predicted Points Over Season</h2>
      </div>
      <div className="card-body">
        <ResponsiveContainer width="100%" height={500}>
          <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="week"
              tick={{ fontSize: 12, fill: '#64748b' }}
              label={{ value: 'Matchweek', position: 'bottom', offset: 10 }}
            />
            <YAxis
              tick={{ fontSize: 12, fill: '#64748b' }}
              label={{ value: 'Points', angle: -90, position: 'insideLeft' }}
              domain={[0, 'auto']}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
              }}
            />
            <Legend wrapperStyle={{ paddingTop: '20px' }} />
            {selectedTeams.map(team => (
              <React.Fragment key={team}>
                <Line
                  type="monotone"
                  dataKey={`${team}_expected`}
                  name={`${team} (Predicted)`}
                  stroke={getTeamColor(team)}
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey={`${team}_actual`}
                  name={`${team} (Actual)`}
                  stroke={getTeamColor(team)}
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  dot={false}
                />
              </React.Fragment>
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function StrengthChart({ data, selectedTeams }) {
  if (!data?.history || !data?.weeks?.length) {
    return (
      <div className="card">
        <div className="card-body text-center py-8">
          <p className="text-slate-500">No historical data available yet</p>
        </div>
      </div>
    );
  }

  const hasMultipleWeeks = data.weeks.length > 1;

  // Transform data for recharts
  const chartData = data.weeks.map(week => {
    const point = { week };
    selectedTeams.forEach(team => {
      const teamData = data.history[team]?.find(d => d.week === week);
      if (teamData) {
        point[team] = teamData.theta_avg;
      }
    });
    return point;
  });

  if (!hasMultipleWeeks) {
    return (
      <div className="card">
        <div className="card-body text-center py-8">
          <p className="text-slate-500 mb-2">
            Only 1 week of data available (Week {data.weeks[0]})
          </p>
          <p className="text-xs text-slate-400">
            Historical strength trajectories will appear as more matchweeks are tracked.
            The scheduled update runs automatically after each matchday.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="font-semibold text-slate-900">Team Strength (Theta) Over Season</h2>
      </div>
      <div className="card-body">
        <ResponsiveContainer width="100%" height={500}>
          <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="week"
              tick={{ fontSize: 12, fill: '#64748b' }}
              label={{ value: 'Matchweek', position: 'bottom', offset: 10 }}
            />
            <YAxis
              tick={{ fontSize: 12, fill: '#64748b' }}
              label={{ value: 'Theta (Strength)', angle: -90, position: 'insideLeft' }}
              domain={['auto', 'auto']}  // Keep auto for theta since it can be negative
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
              }}
              formatter={(value) => [value.toFixed(3), 'Theta']}
            />
            <Legend wrapperStyle={{ paddingTop: '20px' }} />
            {selectedTeams.map(team => (
              <Line
                key={team}
                type="monotone"
                dataKey={team}
                name={team}
                stroke={getTeamColor(team)}
                strokeWidth={2}
                dot={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function TitleRaceChart({ data, selectedTeams }) {
  if (!data?.history || !data?.weeks?.length) {
    return (
      <div className="card">
        <div className="card-body text-center py-8">
          <p className="text-slate-500">No historical data available yet</p>
        </div>
      </div>
    );
  }

  const hasMultipleWeeks = data.weeks.length > 1;

  // Transform data for recharts
  const chartData = data.weeks.map(week => {
    const point = { week };
    selectedTeams.forEach(team => {
      const teamData = data.history[team]?.find(d => d.week === week);
      if (teamData) {
        point[team] = teamData.title_prob * 100; // Convert to percentage
      }
    });
    return point;
  });

  if (!hasMultipleWeeks) {
    return (
      <div className="card">
        <div className="card-body text-center py-8">
          <p className="text-slate-500 mb-2">
            Only 1 week of data available (Week {data.weeks[0]})
          </p>
          <p className="text-xs text-slate-400">
            Title race trajectories will appear as more matchweeks are tracked.
            The scheduled update runs automatically after each matchday.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="font-semibold text-slate-900">Title Probability Over Season</h2>
      </div>
      <div className="card-body">
        <ResponsiveContainer width="100%" height={500}>
          <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              dataKey="week"
              tick={{ fontSize: 12, fill: '#64748b' }}
              label={{ value: 'Matchweek', position: 'bottom', offset: 10 }}
            />
            <YAxis
              tick={{ fontSize: 12, fill: '#64748b' }}
              label={{ value: 'Title Probability (%)', angle: -90, position: 'insideLeft' }}
              domain={[0, 100]}
              tickFormatter={(value) => `${value}%`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
              }}
              formatter={(value) => [`${value.toFixed(1)}%`, 'Title Prob']}
            />
            <Legend wrapperStyle={{ paddingTop: '20px' }} />
            {selectedTeams.map(team => (
              <Line
                key={team}
                type="monotone"
                dataKey={team}
                name={team}
                stroke={getTeamColor(team)}
                strokeWidth={2}
                dot={{ r: 3 }}
                activeDot={{ r: 5 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function PositionHeatmap({ data }) {
  if (!data?.predictions?.length) {
    return (
      <div className="card">
        <div className="card-body text-center py-8">
          <p className="text-slate-500">No position probability data available yet</p>
        </div>
      </div>
    );
  }

  const predictions = data.predictions;
  const positions = Array.from({ length: 20 }, (_, i) => i + 1);

  // Find max probability for color scaling
  const maxProb = Math.max(
    ...predictions.flatMap(p => p.position_probs)
  );

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="font-semibold text-slate-900">
          Position Probability Distribution (Week {data.week})
        </h2>
      </div>
      <div className="card-body overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr>
              <th className="text-left py-2 px-1 font-semibold text-slate-700 sticky left-0 bg-white">
                Team
              </th>
              {positions.map(pos => (
                <th
                  key={pos}
                  className={`py-2 px-1 font-semibold text-center ${
                    pos <= 4 ? 'text-blue-600' : pos >= 18 ? 'text-red-600' : 'text-slate-500'
                  }`}
                >
                  {pos}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {predictions.map((team) => (
              <tr key={team.team} className="border-t border-slate-100">
                <td className="py-1.5 px-1 font-medium text-slate-900 sticky left-0 bg-white whitespace-nowrap">
                  {team.team}
                </td>
                {team.position_probs.map((prob, idx) => (
                  <td
                    key={idx}
                    className="py-1.5 px-1 text-center"
                    style={{
                      backgroundColor: prob > 0.001
                        ? `rgba(239, 68, 68, ${Math.min(prob / maxProb, 1) * 0.9})`
                        : 'transparent',
                      color: prob > maxProb * 0.5 ? 'white' : prob > 0.01 ? '#1e293b' : '#94a3b8',
                    }}
                  >
                    {prob >= 0.01 ? `${(prob * 100).toFixed(0)}` : prob > 0.001 ? '<1' : '-'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>

        {/* Legend */}
        <div className="mt-4 flex items-center justify-center gap-4 text-xs text-slate-500">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)' }} />
            <span>Low probability</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgba(239, 68, 68, 0.5)' }} />
            <span>Medium</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgba(239, 68, 68, 0.9)' }} />
            <span>High probability</span>
          </div>
        </div>
      </div>
    </div>
  );
}
