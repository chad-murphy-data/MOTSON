import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
  BarChart,
  Bar,
} from 'recharts';
import {
  ArrowLeft,
  TrendingUp,
  TrendingDown,
  Target,
  Activity,
  AlertCircle,
  CheckCircle,
} from 'lucide-react';

import { getIRTTeamDetail, getIRTTeamHistory, getIRT100MSimulation } from '../api';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import PositionBadge from '../components/PositionBadge';

export default function TeamDetail() {
  const { teamName } = useParams();
  const decodedName = decodeURIComponent(teamName);

  const detailQuery = useQuery({
    queryKey: ['irtTeamDetail', decodedName],
    queryFn: () => getIRTTeamDetail(decodedName),
  });

  const historyQuery = useQuery({
    queryKey: ['irtTeamHistory', decodedName],
    queryFn: () => getIRTTeamHistory(decodedName),
  });

  const probsQuery = useQuery({
    queryKey: ['irt100MSimulation'],
    queryFn: getIRT100MSimulation,
  });

  const team = detailQuery.data;
  const history = historyQuery.data?.history || [];

  // Get this team's season predictions from 100M simulation
  const teamPrediction = probsQuery.data?.teams?.find(
    (p) => p.team === decodedName
  );

  if (detailQuery.isLoading) return <LoadingSpinner />;
  if (detailQuery.error) return <ErrorMessage message="Failed to load team data" />;

  return (
    <div className="space-y-6">
      {/* Back Link */}
      <Link
        to="/standings"
        className="inline-flex items-center text-slate-500 hover:text-slate-700"
      >
        <ArrowLeft className="w-4 h-4 mr-2" />
        Back to Standings
      </Link>

      {/* Header */}
      <div className="card">
        <div className="card-body">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between">
            <div>
              <h1 className="text-2xl font-bold text-slate-900">{decodedName}</h1>
              <p className="text-slate-500 mt-1">
                Team strength analysis and predictions
              </p>
            </div>
            <div className="mt-4 md:mt-0 flex items-center space-x-4">
              <div className="px-3 py-1 bg-primary-50 text-primary-700 text-sm rounded-full">
                Gravity Weight: {(team?.gravity_weight * 100)?.toFixed(0)}%
              </div>
              {team?.is_promoted && (
                <div className="px-3 py-1 bg-amber-50 text-amber-700 text-sm rounded-full">
                  Promoted
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Theta"
          value={team?.theta?.toFixed(3)}
          icon={Target}
          description="Team ability"
        />
        <StatCard
          label="Home Difficulty"
          value={team?.b_home?.toFixed(3)}
          icon={Activity}
          description="How hard to beat at home"
        />
        <StatCard
          label="Away Difficulty"
          value={team?.b_away?.toFixed(3)}
          icon={Activity}
          description="How hard to beat away"
        />
        <StatCard
          label="Performance"
          value={`${team?.performance_vs_expected > 0 ? '+' : ''}${team?.performance_vs_expected?.toFixed(1)}`}
          icon={team?.performance_vs_expected > 0 ? TrendingUp : TrendingDown}
          description="vs expected points"
          highlight={Math.abs(team?.performance_vs_expected) > 3}
        />
      </div>

      {/* Calibration Status */}
      <div className="card">
        <div className="card-header">
          <h2 className="font-semibold text-slate-900">Calibration Status</h2>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <div className="text-sm text-slate-500">Actual Points</div>
              <div className="text-3xl font-bold text-slate-900">
                {team?.actual_points_season}
              </div>
              <div className="text-sm text-slate-500">
                from {team?.matches_played} matches
              </div>
            </div>
            <div>
              <div className="text-sm text-slate-500">Expected Points</div>
              <div className="text-3xl font-bold text-slate-900">
                {team?.expected_points_season?.toFixed(1)}
              </div>
              <div className="text-sm text-slate-500">given theta</div>
            </div>
            <div>
              <div className="text-sm text-slate-500">Performance</div>
              <CalibrationIndicator
                actual={team?.actual_points_season}
                expected={team?.expected_points_season}
                diff={team?.performance_vs_expected}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Theta Trajectory */}
        <div className="card">
          <div className="card-header">
            <h2 className="font-semibold text-slate-900">Theta Trajectory</h2>
            <p className="text-sm text-slate-500">
              Team strength over the season
            </p>
          </div>
          <div className="card-body">
            {history.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={history}>
                  <defs>
                    <linearGradient id="thetaGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis
                    dataKey="week"
                    tick={{ fontSize: 12, fill: '#64748b' }}
                    axisLine={{ stroke: '#e2e8f0' }}
                  />
                  <YAxis
                    tick={{ fontSize: 12, fill: '#64748b' }}
                    axisLine={{ stroke: '#e2e8f0' }}
                    domain={['auto', 'auto']}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'white',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="theta"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    fill="url(#thetaGradient)"
                    name="Theta"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-250 flex items-center justify-center text-slate-400">
                No history data yet
              </div>
            )}
          </div>
        </div>

        {/* Position Probability Distribution */}
        <div className="card">
          <div className="card-header">
            <h2 className="font-semibold text-slate-900">Position Distribution</h2>
            <p className="text-sm text-slate-500">
              Probability of finishing in each position
            </p>
          </div>
          <div className="card-body">
            {teamPrediction?.position_probs ? (
              <ResponsiveContainer width="100%" height={250}>
                <BarChart
                  data={teamPrediction.position_probs.map((prob, idx) => ({
                    position: idx + 1,
                    probability: prob * 100,
                  }))}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis
                    dataKey="position"
                    tick={{ fontSize: 10, fill: '#64748b' }}
                    axisLine={{ stroke: '#e2e8f0' }}
                  />
                  <YAxis
                    tick={{ fontSize: 12, fill: '#64748b' }}
                    axisLine={{ stroke: '#e2e8f0' }}
                    unit="%"
                  />
                  <Tooltip
                    formatter={(value) => [`${value.toFixed(1)}%`, 'Probability']}
                    contentStyle={{
                      backgroundColor: 'white',
                      border: '1px solid #e2e8f0',
                      borderRadius: '8px',
                    }}
                  />
                  <Bar
                    dataKey="probability"
                    fill="#3b82f6"
                    radius={[4, 4, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-250 flex items-center justify-center text-slate-400">
                No prediction data yet
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Season Outcomes */}
      {teamPrediction && (
        <div className="card">
          <div className="card-header">
            <h2 className="font-semibold text-slate-900">Season Outcomes</h2>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <OutcomeCard
                label="Title"
                probability={teamPrediction.p_title / 100}
                color="amber"
              />
              <OutcomeCard
                label="Top 4"
                probability={teamPrediction.p_top4 / 100}
                color="blue"
              />
              <OutcomeCard
                label="Top 6"
                probability={teamPrediction.p_top6 / 100}
                color="emerald"
              />
              <OutcomeCard
                label="Relegation"
                probability={teamPrediction.p_relegation / 100}
                color="red"
              />
              <div className="bg-slate-50 rounded-lg p-4">
                <div className="text-sm text-slate-500">Expected Position</div>
                <div className="text-2xl font-bold text-slate-900">
                  {teamPrediction.expected_position?.toFixed(1)}
                </div>
                <div className="text-xs text-slate-400">
                  ~{teamPrediction.expected_points?.toFixed(0)} pts
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Model Explanation */}
      <div className="card bg-slate-50">
        <div className="card-body">
          <h3 className="font-semibold text-slate-900 mb-2">Understanding the IRT Model</h3>
          <div className="text-sm text-slate-600 space-y-2">
            <p>
              <strong>Theta ({team?.theta?.toFixed(3)})</strong>:
              Team ability - how good this team is at winning matches. Higher = stronger team.
              Blends 5-year historical prior ({team?.theta_prior?.toFixed(3)}) with current season
              performance ({team?.theta_season?.toFixed(3)}).
            </p>
            <p>
              <strong>Home Difficulty ({team?.b_home?.toFixed(3)})</strong>:
              How hard this team is to beat at home. Higher values mean opponents struggle more here.
            </p>
            <p>
              <strong>Away Difficulty ({team?.b_away?.toFixed(3)})</strong>:
              How hard this team is to beat away. Higher values mean they're tough on the road.
            </p>
            <p>
              <strong>Gravity Weight ({(team?.gravity_weight * 100)?.toFixed(0)}%)</strong>:
              How much the 5-year prior influences the current estimate. Early in the season this
              is higher; as matches are played, current form has more weight.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value, icon: Icon, description, highlight }) {
  return (
    <div className={`stat-card ${highlight ? 'ring-2 ring-primary-200' : ''}`}>
      <div className="flex items-center justify-between">
        <span className="text-sm text-slate-500">{label}</span>
        <Icon className={`w-5 h-5 ${highlight ? 'text-primary-500' : 'text-slate-400'}`} />
      </div>
      <div className="stat-value mt-1">{value}</div>
      <div className="text-xs text-slate-400 mt-1">{description}</div>
    </div>
  );
}

function CalibrationIndicator({ actual, expected, diff }) {
  const pointsDiff = diff ?? (actual - expected);
  const isOverperforming = pointsDiff > 0;
  const isSignificant = Math.abs(pointsDiff) > 3;

  return (
    <div className="flex items-center space-x-2">
      {isSignificant ? (
        <>
          <AlertCircle
            className={`w-6 h-6 ${isOverperforming ? 'text-emerald-500' : 'text-red-500'}`}
          />
          <div>
            <div
              className={`text-lg font-bold ${
                isOverperforming ? 'text-emerald-600' : 'text-red-600'
              }`}
            >
              {isOverperforming ? '+' : ''}{pointsDiff?.toFixed(1)} pts
            </div>
            <div className="text-xs text-slate-500">
              {isOverperforming ? 'Over' : 'Under'}-performing
            </div>
          </div>
        </>
      ) : (
        <>
          <CheckCircle className="w-6 h-6 text-primary-500" />
          <div>
            <div className="text-lg font-bold text-slate-700">
              {pointsDiff > 0 ? '+' : ''}{pointsDiff?.toFixed(1)} pts
            </div>
            <div className="text-xs text-slate-500">Within expectations</div>
          </div>
        </>
      )}
    </div>
  );
}

function OutcomeCard({ label, probability, color }) {
  const colorClasses = {
    amber: 'bg-amber-50 text-amber-700',
    blue: 'bg-blue-50 text-blue-700',
    emerald: 'bg-emerald-50 text-emerald-700',
    red: 'bg-red-50 text-red-700',
  };

  const pct = (probability * 100).toFixed(1);

  return (
    <div className={`rounded-lg p-4 ${colorClasses[color]}`}>
      <div className="text-sm opacity-75">{label}</div>
      <div className="text-2xl font-bold">{pct}%</div>
    </div>
  );
}
