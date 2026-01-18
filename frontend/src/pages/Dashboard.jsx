import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import {
  Trophy,
  TrendingUp,
  AlertTriangle,
  ChevronRight,
  RefreshCw,
  Activity,
} from 'lucide-react';

import {
  getCurrentStandings,
  getTitleRace,
  getRelegationBattle,
  getSeasonProbabilities,
  getNextWeekPredictions,
  triggerUpdate,
} from '../api';

import ProbabilityBar from '../components/ProbabilityBar';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';

export default function Dashboard() {
  const standingsQuery = useQuery({
    queryKey: ['standings'],
    queryFn: getCurrentStandings,
  });

  const titleQuery = useQuery({
    queryKey: ['titleRace'],
    queryFn: getTitleRace,
  });

  const relegationQuery = useQuery({
    queryKey: ['relegation'],
    queryFn: getRelegationBattle,
  });

  const seasonProbsQuery = useQuery({
    queryKey: ['seasonProbabilities'],
    queryFn: getSeasonProbabilities,
  });

  // Get top 4 probabilities sorted by top4_prob
  const top4Data = React.useMemo(() => {
    const predictions = seasonProbsQuery.data?.predictions || [];
    return [...predictions]
      .sort((a, b) => b.top4_prob - a.top4_prob)
      .slice(0, 6)
      .map(team => ({
        team: team.team,
        probability: team.top4_prob,
      }));
  }, [seasonProbsQuery.data]);

  const predictionsQuery = useQuery({
    queryKey: ['nextPredictions'],
    queryFn: getNextWeekPredictions,
  });

  const [isUpdating, setIsUpdating] = React.useState(false);

  const handleUpdate = async () => {
    setIsUpdating(true);
    try {
      await triggerUpdate();
      // Refetch all data
      standingsQuery.refetch();
      titleQuery.refetch();
      relegationQuery.refetch();
      seasonProbsQuery.refetch();
      predictionsQuery.refetch();
    } catch (error) {
      console.error('Update failed:', error);
    } finally {
      setIsUpdating(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Dashboard</h1>
          <p className="text-slate-500 mt-1">
            EPL predictions powered by Bayesian inference
          </p>
        </div>
        <button
          onClick={handleUpdate}
          disabled={isUpdating}
          className="mt-4 md:mt-0 flex items-center space-x-2 px-4 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600 transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${isUpdating ? 'animate-spin' : ''}`} />
          <span>{isUpdating ? 'Updating...' : 'Update Predictions'}</span>
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Title Race */}
        <div className="card">
          <div className="card-header flex items-center justify-between">
            <h2 className="font-semibold text-slate-900 flex items-center">
              <Trophy className="w-5 h-5 text-amber-500 mr-2" />
              Title Race
            </h2>
            <Link
              to="/outcomes"
              className="text-primary-500 hover:text-primary-600 text-sm flex items-center"
            >
              View all <ChevronRight className="w-4 h-4" />
            </Link>
          </div>
          <div className="card-body">
            {titleQuery.isLoading ? (
              <LoadingSpinner />
            ) : titleQuery.error || !titleQuery.data?.title_race?.length ? (
              <div className="text-center py-4 text-sm text-slate-500">
                Run "Update Predictions" to see title race probabilities
              </div>
            ) : (
              <div className="space-y-4">
                {titleQuery.data?.title_race?.slice(0, 5).map((team, idx) => (
                  <div key={team.team} className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <span className="text-sm font-medium text-slate-500 w-4">
                        {idx + 1}
                      </span>
                      <Link
                        to={`/team/${encodeURIComponent(team.team)}`}
                        className="font-medium text-slate-900 hover:text-primary-600"
                      >
                        {team.team}
                      </Link>
                    </div>
                    <div className="flex items-center space-x-3">
                      <ProbabilityBar
                        value={team.probability}
                        color="amber"
                        width={100}
                      />
                      <span className="text-sm font-semibold text-slate-700 w-12 text-right">
                        {(team.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Top 4 Battle */}
        <div className="card">
          <div className="card-header flex items-center justify-between">
            <h2 className="font-semibold text-slate-900 flex items-center">
              <TrendingUp className="w-5 h-5 text-blue-500 mr-2" />
              Champions League
            </h2>
            <Link
              to="/outcomes"
              className="text-primary-500 hover:text-primary-600 text-sm flex items-center"
            >
              View all <ChevronRight className="w-4 h-4" />
            </Link>
          </div>
          <div className="card-body">
            {seasonProbsQuery.isLoading ? (
              <LoadingSpinner />
            ) : seasonProbsQuery.error || !top4Data.length ? (
              <div className="text-center py-4 text-sm text-slate-500">
                Run "Update Predictions" to see Top 4 probabilities
              </div>
            ) : (
              <div className="space-y-4">
                {top4Data.map((team, idx) => (
                  <div key={team.team} className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <span className="text-sm font-medium text-slate-500 w-4">
                        {idx + 1}
                      </span>
                      <Link
                        to={`/team/${encodeURIComponent(team.team)}`}
                        className="font-medium text-slate-900 hover:text-primary-600"
                      >
                        {team.team}
                      </Link>
                    </div>
                    <div className="flex items-center space-x-3">
                      <ProbabilityBar
                        value={team.probability}
                        color="blue"
                        width={100}
                      />
                      <span className="text-sm font-semibold text-slate-700 w-12 text-right">
                        {(team.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Relegation Battle */}
        <div className="card">
          <div className="card-header flex items-center justify-between">
            <h2 className="font-semibold text-slate-900 flex items-center">
              <AlertTriangle className="w-5 h-5 text-red-500 mr-2" />
              Relegation Battle
            </h2>
            <Link
              to="/outcomes"
              className="text-primary-500 hover:text-primary-600 text-sm flex items-center"
            >
              View all <ChevronRight className="w-4 h-4" />
            </Link>
          </div>
          <div className="card-body">
            {relegationQuery.isLoading ? (
              <LoadingSpinner />
            ) : relegationQuery.error || !relegationQuery.data?.relegation_battle?.length ? (
              <div className="text-center py-4 text-sm text-slate-500">
                Run "Update Predictions" to see relegation probabilities
              </div>
            ) : (
              <div className="space-y-4">
                {relegationQuery.data?.relegation_battle?.slice(0, 5).map((team, idx) => (
                  <div key={team.team} className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <span className="text-sm font-medium text-slate-500 w-4">
                        {idx + 1}
                      </span>
                      <Link
                        to={`/team/${encodeURIComponent(team.team)}`}
                        className="font-medium text-slate-900 hover:text-primary-600"
                      >
                        {team.team}
                      </Link>
                    </div>
                    <div className="flex items-center space-x-3">
                      <ProbabilityBar
                        value={team.probability}
                        color="red"
                        width={100}
                      />
                      <span className="text-sm font-semibold text-slate-700 w-12 text-right">
                        {(team.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Next Week Predictions */}
      <div className="card">
        <div className="card-header flex items-center justify-between">
          <h2 className="font-semibold text-slate-900 flex items-center">
            <Activity className="w-5 h-5 text-primary-500 mr-2" />
            Next Week Predictions
          </h2>
          <Link
            to="/predictions"
            className="text-primary-500 hover:text-primary-600 text-sm flex items-center"
          >
            All predictions <ChevronRight className="w-4 h-4" />
          </Link>
        </div>
        <div className="card-body">
          {predictionsQuery.isLoading ? (
            <LoadingSpinner />
          ) : predictionsQuery.error ? (
            <ErrorMessage message="Failed to load predictions" />
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {predictionsQuery.data?.predictions?.slice(0, 6).map((match) => (
                <MatchPredictionCard key={match.match_id} match={match} />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Model Philosophy */}
      <div className="card bg-gradient-to-br from-primary-50 to-white">
        <div className="card-body">
          <h3 className="font-semibold text-primary-900 mb-2">
            How MOTSON Works
          </h3>
          <p className="text-sm text-primary-700 leading-relaxed">
            "Track distributions, not point estimates. Update on cumulative calibration,
            not individual surprises." Each team has a strength parameter (theta) with
            uncertainty (sigma). We only update beliefs when a team is <em>systematically</em>
            over or under-performing, not for individual fluky results. Teams like Man City
            have "sticky" beliefs - one loss doesn't change much. Promoted teams have wide
            uncertainty - every result is informative.
          </p>
        </div>
      </div>
    </div>
  );
}

function MatchPredictionCard({ match }) {
  const homeWin = match.home_win_prob * 100;
  const draw = match.draw_prob * 100;
  const awayWin = match.away_win_prob * 100;

  const favorite =
    homeWin > awayWin ? 'home' : awayWin > homeWin ? 'away' : 'draw';

  return (
    <div className="bg-slate-50 rounded-lg p-4">
      <div className="flex justify-between items-center mb-3">
        <Link
          to={`/team/${encodeURIComponent(match.home_team)}`}
          className={`font-medium hover:text-primary-600 ${
            favorite === 'home' ? 'text-slate-900' : 'text-slate-600'
          }`}
        >
          {match.home_team}
        </Link>
        <span className="text-xs text-slate-400">vs</span>
        <Link
          to={`/team/${encodeURIComponent(match.away_team)}`}
          className={`font-medium hover:text-primary-600 ${
            favorite === 'away' ? 'text-slate-900' : 'text-slate-600'
          }`}
        >
          {match.away_team}
        </Link>
      </div>

      {/* Probability bars */}
      <div className="flex h-3 rounded-full overflow-hidden bg-slate-200">
        <div
          className="bg-primary-500 transition-all"
          style={{ width: `${homeWin}%` }}
          title={`Home: ${homeWin.toFixed(1)}%`}
        />
        <div
          className="bg-slate-400 transition-all"
          style={{ width: `${draw}%` }}
          title={`Draw: ${draw.toFixed(1)}%`}
        />
        <div
          className="bg-primary-300 transition-all"
          style={{ width: `${awayWin}%` }}
          title={`Away: ${awayWin.toFixed(1)}%`}
        />
      </div>

      {/* Legend */}
      <div className="flex justify-between mt-2 text-xs">
        <span className="text-primary-600 font-medium">{homeWin.toFixed(0)}%</span>
        <span className="text-slate-500">{draw.toFixed(0)}%</span>
        <span className="text-primary-400 font-medium">{awayWin.toFixed(0)}%</span>
      </div>

      {/* Confidence */}
      <div className="mt-2 flex items-center justify-center">
        <span className="text-xs text-slate-400">
          Confidence: {(match.confidence * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  );
}
