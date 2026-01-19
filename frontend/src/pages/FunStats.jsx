import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import {
  Skull,
  Trophy,
  TrendingDown,
  TrendingUp,
  Target,
  AlertTriangle,
  Sparkles,
  BarChart3,
  Swords,
} from 'lucide-react';

import {
  getIRTFunStats,
  getIRTSimulationHistory,
  getIRT100MSimulation,
  getIRTRivalries,
} from '../api';

import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';

export default function FunStats() {
  const funStatsQuery = useQuery({
    queryKey: ['funStats'],
    queryFn: () => getIRTFunStats(100000),
  });

  const sim100MQuery = useQuery({
    queryKey: ['sim100m'],
    queryFn: getIRT100MSimulation,
  });

  // Get Wolves history for the tracker
  const wolvesHistoryQuery = useQuery({
    queryKey: ['wolvesHistory'],
    queryFn: () => getIRTSimulationHistory('Wolves'),
  });

  // Get rivalry data
  const rivalriesQuery = useQuery({
    queryKey: ['rivalries'],
    queryFn: () => getIRTRivalries(100000),
  });

  const isLoading = funStatsQuery.isLoading || sim100MQuery.isLoading;
  const error = funStatsQuery.error || sim100MQuery.error;

  if (isLoading) return <LoadingSpinner />;
  if (error) return <ErrorMessage message={error.message} />;

  const funStats = funStatsQuery.data?.stats || [];
  const sim100M = sim100MQuery.data;
  const wolvesHistory = wolvesHistoryQuery.data?.history || [];
  const rivalries = rivalriesQuery.data?.rivalries || [];

  // Get specific stats
  const wolvesWorse = funStats.find(
    s => s.team === 'Wolves' && s.category === 'worse_than_derby'
  );
  const wolvesSurvival = funStats.find(
    s => s.team === 'Wolves' && s.category === 'survival_watch'
  );
  const historicallyBad = funStats.filter(s => s.category === 'historically_bad');
  const centurionWatch = funStats.filter(s => s.category === 'centurion_watch');
  const excellentSeasons = funStats.filter(s => s.category === 'excellent_season');
  const surpriseTitles = funStats.filter(s => s.category === 'surprise_title');
  const survivalWatch = funStats.filter(s => s.category === 'survival_watch');

  // Get 100M data for ultra-precise probabilities
  const wolves100M = sim100M?.teams?.find(t => t.team === 'Wolves');

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900 flex items-center">
          <Sparkles className="w-6 h-6 text-amber-500 mr-2" />
          Fun Stats
        </h1>
        <p className="text-slate-500 mt-1">
          Extreme scenario probabilities - the Opta Troll Edition
        </p>
        {sim100M && (
          <p className="text-xs text-slate-400 mt-1">
            Based on 10,000 supercomputers running 10,000 simulations each (Week {sim100M.week})
          </p>
        )}
      </div>

      {/* Wolves Worst-Ever Tracker - THE MAIN EVENT */}
      <div className="card bg-gradient-to-br from-amber-50 to-orange-50 border-amber-200">
        <div className="card-header">
          <h2 className="font-bold text-lg text-amber-900 flex items-center">
            <Skull className="w-6 h-6 text-amber-600 mr-2" />
            Wolves Worst-Ever Tracker
          </h2>
          <p className="text-sm text-amber-700 mt-1">
            Derby County (2007-08) hold the record with 11 points. Can Wolves do worse?
          </p>
        </div>
        <div className="card-body space-y-6">
          {/* Big number display */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Worse than Derby */}
            <div className="bg-white rounded-xl p-6 shadow-sm border border-amber-100">
              <div className="text-center">
                <div className="text-4xl font-bold text-amber-600">
                  {wolves100M ? wolves100M.p_relegation.toFixed(5) : wolvesWorse?.probability?.toFixed(4) || '0'}%
                </div>
                <div className="text-sm text-slate-600 mt-2">
                  chance of being <strong>worst ever</strong>
                </div>
                <div className="text-xs text-slate-400 mt-1">
                  (&lt;11 points - worse than Derby)
                </div>
                {wolves100M && (
                  <div className="text-xs text-amber-600 mt-2 font-medium">
                    That's {(wolves100M.p_relegation * 100000000 / 100).toLocaleString()} out of 100M parallel universes
                  </div>
                )}
              </div>
            </div>

            {/* Survival odds */}
            <div className="bg-white rounded-xl p-6 shadow-sm border border-green-100">
              <div className="text-center">
                <div className="text-4xl font-bold text-green-600">
                  {wolves100M
                    ? (100 - wolves100M.p_relegation).toFixed(3)
                    : wolvesSurvival?.probability?.toFixed(3) || '0'}%
                </div>
                <div className="text-sm text-slate-600 mt-2">
                  chance of <strong>survival</strong>
                </div>
                <div className="text-xs text-slate-400 mt-1">
                  (staying up somehow)
                </div>
                {wolves100M && (
                  <div className="text-xs text-green-600 mt-2 font-medium">
                    {((100 - wolves100M.p_relegation) * 1000000 / 100).toLocaleString()} survival universes
                  </div>
                )}
              </div>
            </div>

            {/* Current points */}
            <div className="bg-white rounded-xl p-6 shadow-sm border border-slate-100">
              <div className="text-center">
                <div className="text-4xl font-bold text-slate-700">
                  {wolves100M?.current_points || 8}
                </div>
                <div className="text-sm text-slate-600 mt-2">
                  current points
                </div>
                <div className="text-xs text-slate-400 mt-1">
                  ({wolves100M?.expected_points?.toFixed(1)} predicted final)
                </div>
                <div className="text-xs text-red-500 mt-2 font-medium">
                  {wolves100M?.expected_points < 20 ? 'On track for historically bad' : 'Fighting for survival'}
                </div>
              </div>
            </div>
          </div>

          {/* Week-by-week tracker */}
          {wolvesHistory.length > 0 && (
            <div>
              <h3 className="font-semibold text-amber-900 mb-3">
                Week-by-Week Relegation Probability
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-amber-200">
                      <th className="text-left py-2 px-3 text-amber-700">Week</th>
                      <th className="text-right py-2 px-3 text-amber-700">Theta</th>
                      <th className="text-right py-2 px-3 text-amber-700">Pred. Position</th>
                      <th className="text-right py-2 px-3 text-amber-700">p(Relegation)</th>
                      <th className="text-right py-2 px-3 text-amber-700">Trend</th>
                    </tr>
                  </thead>
                  <tbody>
                    {wolvesHistory.slice(-10).map((week, idx, arr) => {
                      const prevWeek = arr[idx - 1];
                      const relDiff = prevWeek ? week.p_relegation - prevWeek.p_relegation : 0;
                      return (
                        <tr key={week.week} className="border-b border-amber-100 hover:bg-amber-50">
                          <td className="py-2 px-3 font-medium">{week.week}</td>
                          <td className="py-2 px-3 text-right">{week.theta?.toFixed(3)}</td>
                          <td className="py-2 px-3 text-right">{week.predicted_position?.toFixed(1)}</td>
                          <td className="py-2 px-3 text-right font-semibold text-red-600">
                            {week.p_relegation?.toFixed(1)}%
                          </td>
                          <td className="py-2 px-3 text-right">
                            {relDiff > 0 ? (
                              <span className="text-red-500">+{relDiff.toFixed(1)}%</span>
                            ) : relDiff < 0 ? (
                              <span className="text-green-500">{relDiff.toFixed(1)}%</span>
                            ) : (
                              <span className="text-slate-400">-</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Rivalry Comparisons */}
      {rivalries.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h2 className="font-bold text-lg text-slate-900 flex items-center">
              <Swords className="w-6 h-6 text-purple-500 mr-2" />
              Rivalry Watch: Who Finishes Higher?
            </h2>
            <p className="text-sm text-slate-500 mt-1">
              Head-to-head final position probabilities from 10,000 supercomputers
            </p>
          </div>
          <div className="card-body">
            <div className="space-y-4">
              {rivalries.map((rivalry, idx) => {
                const t1Higher = rivalry.team1_higher;
                const t2Higher = rivalry.team2_higher;
                const maxProb = Math.max(t1Higher, t2Higher);
                const favorite = t1Higher > t2Higher ? rivalry.team1 : rivalry.team2;

                return (
                  <div key={idx} className="bg-slate-50 rounded-lg p-4">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-xs text-slate-500 font-medium uppercase tracking-wide">
                        {rivalry.rivalry_name}
                      </span>
                      <span className="text-xs text-purple-600 font-medium">
                        {favorite} favored
                      </span>
                    </div>
                    <div className="flex items-center justify-between mb-3">
                      <Link
                        to={`/team/${encodeURIComponent(rivalry.team1)}`}
                        className={`font-semibold hover:text-primary-600 ${
                          t1Higher > t2Higher ? 'text-slate-900' : 'text-slate-500'
                        }`}
                      >
                        {rivalry.team1}
                        <span className="text-xs text-slate-400 ml-1">
                          ({rivalry.team1_current_points} pts)
                        </span>
                      </Link>
                      <span className="text-slate-400 text-sm">vs</span>
                      <Link
                        to={`/team/${encodeURIComponent(rivalry.team2)}`}
                        className={`font-semibold hover:text-primary-600 text-right ${
                          t2Higher > t1Higher ? 'text-slate-900' : 'text-slate-500'
                        }`}
                      >
                        {rivalry.team2}
                        <span className="text-xs text-slate-400 ml-1">
                          ({rivalry.team2_current_points} pts)
                        </span>
                      </Link>
                    </div>
                    {/* Probability bar */}
                    <div className="flex h-4 rounded-full overflow-hidden bg-slate-200">
                      <div
                        className="bg-purple-500 transition-all"
                        style={{ width: `${t1Higher}%` }}
                        title={`${rivalry.team1}: ${t1Higher.toFixed(1)}%`}
                      />
                      <div
                        className="bg-slate-300 transition-all"
                        style={{ width: `${rivalry.same_position}%` }}
                        title={`Same position: ${rivalry.same_position?.toFixed(1)}%`}
                      />
                      <div
                        className="bg-purple-300 transition-all"
                        style={{ width: `${t2Higher}%` }}
                        title={`${rivalry.team2}: ${t2Higher.toFixed(1)}%`}
                      />
                    </div>
                    <div className="flex justify-between mt-2 text-sm">
                      <span className="text-purple-600 font-semibold">{t1Higher.toFixed(1)}%</span>
                      <span className="text-slate-400 text-xs">
                        Avg diff: {rivalry.avg_points_difference > 0 ? '+' : ''}{rivalry.avg_points_difference} pts
                      </span>
                      <span className="text-purple-400 font-semibold">{t2Higher.toFixed(1)}%</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Other Fun Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Centurion Watch */}
        {centurionWatch.length > 0 && (
          <div className="card">
            <div className="card-header">
              <h2 className="font-semibold text-slate-900 flex items-center">
                <Trophy className="w-5 h-5 text-amber-500 mr-2" />
                Centurion Watch (100+ pts)
              </h2>
            </div>
            <div className="card-body">
              <div className="text-sm text-slate-500 mb-4">
                Only Man City (2017-18) have reached 100 points
              </div>
              {centurionWatch.length > 0 ? (
                <div className="space-y-3">
                  {centurionWatch.map(stat => (
                    <div key={stat.team} className="flex justify-between items-center">
                      <Link
                        to={`/team/${encodeURIComponent(stat.team)}`}
                        className="font-medium text-slate-900 hover:text-primary-600"
                      >
                        {stat.team}
                      </Link>
                      <div className="text-right">
                        <span className="font-semibold text-amber-600">
                          {stat.probability.toFixed(4)}%
                        </span>
                        <span className="text-xs text-slate-400 ml-2">
                          ({stat.simulations.toLocaleString()} sims)
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-slate-500 text-sm">No team has a realistic shot at 100 points</p>
              )}
            </div>
          </div>
        )}

        {/* 90+ Points Club */}
        <div className="card">
          <div className="card-header">
            <h2 className="font-semibold text-slate-900 flex items-center">
              <TrendingUp className="w-5 h-5 text-green-500 mr-2" />
              90+ Points Club
            </h2>
          </div>
          <div className="card-body">
            <div className="text-sm text-slate-500 mb-4">
              Excellent season territory
            </div>
            {excellentSeasons.length > 0 ? (
              <div className="space-y-3">
                {excellentSeasons.sort((a, b) => b.probability - a.probability).map(stat => (
                  <div key={stat.team} className="flex justify-between items-center">
                    <Link
                      to={`/team/${encodeURIComponent(stat.team)}`}
                      className="font-medium text-slate-900 hover:text-primary-600"
                    >
                      {stat.team}
                    </Link>
                    <span className="font-semibold text-green-600">
                      {stat.probability.toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-slate-500 text-sm">No team above 1% chance for 90+ points</p>
            )}
          </div>
        </div>

        {/* Historically Bad Seasons */}
        <div className="card">
          <div className="card-header">
            <h2 className="font-semibold text-slate-900 flex items-center">
              <TrendingDown className="w-5 h-5 text-red-500 mr-2" />
              Historically Bad (&lt;20 pts)
            </h2>
          </div>
          <div className="card-body">
            <div className="text-sm text-slate-500 mb-4">
              Only 5 teams have finished below 20 points
            </div>
            {historicallyBad.length > 0 ? (
              <div className="space-y-3">
                {historicallyBad.sort((a, b) => b.probability - a.probability).map(stat => (
                  <div key={stat.team} className="flex justify-between items-center">
                    <Link
                      to={`/team/${encodeURIComponent(stat.team)}`}
                      className="font-medium text-slate-900 hover:text-primary-600"
                    >
                      {stat.team}
                    </Link>
                    <span className="font-semibold text-red-600">
                      {stat.probability.toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-slate-500 text-sm">No team above 1% chance for &lt;20 points</p>
            )}
          </div>
        </div>

        {/* Survival Watch */}
        <div className="card">
          <div className="card-header">
            <h2 className="font-semibold text-slate-900 flex items-center">
              <AlertTriangle className="w-5 h-5 text-orange-500 mr-2" />
              Survival Watch
            </h2>
          </div>
          <div className="card-body">
            <div className="text-sm text-slate-500 mb-4">
              Teams more likely to go down than stay up
            </div>
            {survivalWatch.length > 0 ? (
              <div className="space-y-3">
                {survivalWatch.sort((a, b) => a.probability - b.probability).map(stat => (
                  <div key={stat.team} className="flex justify-between items-center">
                    <Link
                      to={`/team/${encodeURIComponent(stat.team)}`}
                      className="font-medium text-slate-900 hover:text-primary-600"
                    >
                      {stat.team}
                    </Link>
                    <div className="text-right">
                      <span className="font-semibold text-orange-600">
                        {stat.probability.toFixed(2)}%
                      </span>
                      <span className="text-xs text-slate-400 ml-1">survival</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-slate-500 text-sm">No team is more likely relegated than not</p>
            )}
          </div>
        </div>
      </div>

      {/* 100M Simulation Banner */}
      {sim100M && (
        <div className="card bg-gradient-to-r from-slate-800 to-slate-900 text-white">
          <div className="card-body">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-bold text-lg flex items-center">
                  <BarChart3 className="w-5 h-5 mr-2 text-amber-400" />
                  10,000 Supercomputers x 10,000 Simulations
                </h3>
                <p className="text-slate-300 text-sm mt-1">
                  The Opta Troll Edition - generated {new Date(sim100M.generated_at).toLocaleDateString()}
                </p>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-amber-400">
                  100M
                </div>
                <div className="text-xs text-slate-400">parallel universes explored</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
