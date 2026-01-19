import React from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Trophy,
  Users,
  TrendingUp,
  Calendar,
  Sparkles,
  Settings,
  RefreshCw,
  BarChart3,
  Skull,
} from 'lucide-react';

import Dashboard from './pages/Dashboard';
import Standings from './pages/Standings';
import TeamDetail from './pages/TeamDetail';
import Predictions from './pages/Predictions';
import SeasonOutcomes from './pages/SeasonOutcomes';
import Counterfactual from './pages/Counterfactual';
import Analytics from './pages/Analytics';
import FunStats from './pages/FunStats';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-50">
        {/* Header */}
        <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              {/* Logo */}
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-primary-500 rounded-xl flex items-center justify-center">
                  <Trophy className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-lg font-bold text-slate-900">MOTSON</h1>
                  <p className="text-xs text-slate-500">EPL Predictions</p>
                </div>
              </div>

              {/* Navigation */}
              <nav className="hidden md:flex items-center space-x-1">
                <NavItem to="/" icon={LayoutDashboard} label="Dashboard" />
                <NavItem to="/standings" icon={Trophy} label="Standings" />
                <NavItem to="/predictions" icon={Calendar} label="Predictions" />
                <NavItem to="/outcomes" icon={TrendingUp} label="Outcomes" />
                <NavItem to="/fun-stats" icon={Skull} label="Fun Stats" />
                <NavItem to="/counterfactual" icon={Sparkles} label="What If?" />
              </nav>

              {/* Season Badge */}
              <div className="hidden md:flex items-center space-x-4">
                <span className="px-3 py-1 bg-primary-50 text-primary-700 text-sm font-medium rounded-full">
                  2025-26 Season
                </span>
              </div>
            </div>
          </div>

          {/* Mobile Navigation */}
          <nav className="md:hidden border-t border-slate-100 px-2 py-2 flex justify-around">
            <NavItemMobile to="/" icon={LayoutDashboard} label="Home" />
            <NavItemMobile to="/standings" icon={Trophy} label="Table" />
            <NavItemMobile to="/predictions" icon={Calendar} label="Matches" />
            <NavItemMobile to="/outcomes" icon={TrendingUp} label="Season" />
            <NavItemMobile to="/analytics" icon={BarChart3} label="Analytics" />
          </nav>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/standings" element={<Standings />} />
            <Route path="/team/:teamName" element={<TeamDetail />} />
            <Route path="/predictions" element={<Predictions />} />
            <Route path="/outcomes" element={<SeasonOutcomes />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/fun-stats" element={<FunStats />} />
            <Route path="/counterfactual" element={<Counterfactual />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="bg-white border-t border-slate-200 mt-auto">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
              <div className="text-sm text-slate-500">
                <span className="font-medium">MOTSON</span> - Model Of Team Strength Outcome Network
              </div>
              <div className="text-xs text-slate-400">
                Bayesian inference with cumulative calibration. Data from football-data.org
              </div>
            </div>
          </div>
        </footer>
      </div>
    </BrowserRouter>
  );
}

function NavItem({ to, icon: Icon, label }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
          isActive
            ? 'bg-primary-50 text-primary-700'
            : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
        }`
      }
    >
      <Icon className="w-4 h-4" />
      <span>{label}</span>
    </NavLink>
  );
}

function NavItemMobile({ to, icon: Icon, label }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `flex flex-col items-center px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
          isActive
            ? 'text-primary-600'
            : 'text-slate-500 hover:text-slate-700'
        }`
      }
    >
      <Icon className="w-5 h-5 mb-1" />
      <span>{label}</span>
    </NavLink>
  );
}

export default App;
