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
  BarChart,
  Bar,
  Cell,
  ReferenceLine,
} from 'recharts';
import { TrendingUp, Activity, Grid3X3, Trophy, Scale } from 'lucide-react';

import {
  getHistoricalPoints,
  getHistoricalStrength,
  getHistoricalTitleRace,
  getIRT100MPositions,
  getIRTTeams,
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
  const [view, setView] = React.useState('title'); // 'title', 'points', 'strength', 'heatmap', 'deviation'
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
    queryKey: ['100mPositions'],
    queryFn: getIRT100MPositions,  // Use pre-computed 100M results - no memory issues!
  });

  const titleRaceQuery = useQuery({
    queryKey: ['historicalTitleRace'],
    queryFn: getHistoricalTitleRace,
  });

  const teamsQuery = useQuery({
    queryKey: ['irtTeams'],
    queryFn: getIRTTeams,
  });

  const isLoading = view === 'title' ? titleRaceQuery.isLoading
    : view === 'points' ? pointsQuery.isLoading
    : view === 'strength' ? strengthQuery.isLoading
    : view === 'deviation' ? teamsQuery.isLoading
    : positionsQuery.isLoading;

  const error = view === 'title' ? titleRaceQuery.error
    : view === 'points' ? pointsQuery.error
    : view === 'strength' ? strengthQuery.error
    : view === 'deviation' ? teamsQuery.error
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
      const defaultTeams = ['Arsenal', 'Aston Villa', 'Manchester City']
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
          label="Final Points"
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
        <ViewButton
          active={view === 'deviation'}
          onClick={() => setView('deviation')}
          icon={Scale}
          label="Performance vs Expected"
          color="primary"
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
      ) : view === 'deviation' ? (
        <PerformanceDeviationChart data={teamsQuery.data} />
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
            {view === 'points' && 'Predicted Final Points'}
            {view === 'strength' && 'Team Strength (Theta)'}
            {view === 'heatmap' && 'Position Probability Heatmap'}
            {view === 'deviation' && 'Performance vs Expected Points'}
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
            {view === 'deviation' &&
              'Shows how each team is performing relative to what the model expected given their strength (theta). Positive values (green) indicate over-performance - the team has earned more points than expected. Negative values (red) indicate under-performance. Large deviations may suggest luck, injuries, or factors the model hasn\'t captured.'}
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
        // predicted_final_points = predicted end-of-season total
        point[`${team}_predicted`] = teamData.predicted_final_points;
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
        <h2 className="font-semibold text-slate-900">Predicted Final Points Over Season</h2>
        <p className="text-sm text-slate-500">How MOTSON's end-of-season predictions evolve week by week</p>
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
              domain={['auto', 'auto']}
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
                  dataKey={`${team}_predicted`}
                  name={`${team} (Predicted Final)`}
                  stroke={getTeamColor(team)}
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey={`${team}_actual`}
                  name={`${team} (Current)`}
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
  if (!data?.teams?.length) {
    return (
      <div className="card">
        <div className="card-body text-center py-8">
          <p className="text-slate-500">No position probability data available yet</p>
        </div>
      </div>
    );
  }

  const teams = data.teams;
  const positions = Array.from({ length: 20 }, (_, i) => i + 1);

  // Convert position_distribution array to a map for easier lookup
  // and find max probability for color scaling
  const teamsWithProbMap = teams.map(team => {
    const probMap = {};
    team.position_distribution.forEach(({ position, probability }) => {
      probMap[position] = probability;  // probability is already in % (e.g., 91 for 91%)
    });
    return { ...team, probMap };
  });

  const maxProb = Math.max(
    ...teamsWithProbMap.flatMap(t => positions.map(pos => t.probMap[pos] || 0))
  );

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="font-semibold text-slate-900">
          Position Probability Distribution (Week {data.week})
        </h2>
        <p className="text-xs text-slate-500 mt-1">
          Based on {(data.simulations || 100000).toLocaleString()} IRT simulations
        </p>
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
            {teamsWithProbMap.map((team) => (
              <tr key={team.team} className="border-t border-slate-100">
                <td className="py-1.5 px-1 font-medium text-slate-900 sticky left-0 bg-white whitespace-nowrap">
                  {team.team}
                </td>
                {positions.map((pos) => {
                  const prob = team.probMap[pos] || 0;  // prob is in % (e.g., 91)
                  return (
                    <td
                      key={pos}
                      className="py-1.5 px-1 text-center"
                      style={{
                        backgroundColor: prob > 0.1
                          ? `rgba(239, 68, 68, ${Math.min(prob / maxProb, 1) * 0.9})`
                          : 'transparent',
                        color: prob > maxProb * 0.5 ? 'white' : prob > 1 ? '#1e293b' : '#94a3b8',
                      }}
                    >
                      {prob >= 1 ? `${prob.toFixed(0)}` : prob > 0.1 ? '<1' : '-'}
                    </td>
                  );
                })}
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

function PerformanceDeviationChart({ data }) {
  if (!data?.teams?.length) {
    return (
      <div className="card">
        <div className="card-body text-center py-8">
          <p className="text-slate-500">No team data available yet</p>
        </div>
      </div>
    );
  }

  // Sort teams by performance deviation (most over-performing at top)
  const sortedTeams = [...data.teams].sort(
    (a, b) => b.performance_vs_expected - a.performance_vs_expected
  );

  // Prepare data for the horizontal bar chart
  const chartData = sortedTeams.map(team => ({
    team: team.team,
    deviation: team.performance_vs_expected,
    actual: team.actual_points,
    expected: team.expected_points,
  }));

  // Find the max absolute deviation for symmetric axis
  const maxDeviation = Math.max(
    ...chartData.map(d => Math.abs(d.deviation))
  );
  const axisLimit = Math.ceil(maxDeviation + 1);

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="font-semibold text-slate-900">
          Performance vs Expected Points
        </h2>
        <p className="text-xs text-slate-500 mt-1">
          Actual points minus expected points based on team strength (theta)
        </p>
      </div>
      <div className="card-body">
        <ResponsiveContainer width="100%" height={600}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 10, right: 30, left: 100, bottom: 10 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={true} vertical={false} />
            <XAxis
              type="number"
              domain={[-axisLimit, axisLimit]}
              tick={{ fontSize: 12, fill: '#64748b' }}
              axisLine={{ stroke: '#e2e8f0' }}
              tickFormatter={(value) => `${value > 0 ? '+' : ''}${value}`}
            />
            <YAxis
              type="category"
              dataKey="team"
              tick={{ fontSize: 12, fill: '#1e293b' }}
              axisLine={{ stroke: '#e2e8f0' }}
              width={95}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: 'white',
                border: '1px solid #e2e8f0',
                borderRadius: '8px',
              }}
              formatter={(value, name, props) => {
                const { actual, expected } = props.payload;
                return [
                  <div key="tooltip" className="text-sm">
                    <div className="font-semibold">
                      {value > 0 ? '+' : ''}{value.toFixed(1)} pts
                    </div>
                    <div className="text-slate-500 text-xs mt-1">
                      Actual: {actual} pts | Expected: {expected.toFixed(1)} pts
                    </div>
                  </div>,
                  ''
                ];
              }}
              labelFormatter={(label) => label}
            />
            <ReferenceLine x={0} stroke="#94a3b8" strokeWidth={2} />
            <Bar dataKey="deviation" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.deviation >= 0 ? '#10b981' : '#ef4444'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        {/* Legend */}
        <div className="mt-4 flex items-center justify-center gap-6 text-sm text-slate-600">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: '#10b981' }} />
            <span>Over-performing</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded" style={{ backgroundColor: '#ef4444' }} />
            <span>Under-performing</span>
          </div>
        </div>
      </div>
    </div>
  );
}
