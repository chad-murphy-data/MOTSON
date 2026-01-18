import React from 'react';

export default function ProbabilityBar({ value, color = 'primary', width = 80 }) {
  const percentage = Math.min(100, Math.max(0, value * 100));

  const colorClasses = {
    primary: 'bg-primary-500',
    amber: 'bg-amber-500',
    blue: 'bg-blue-500',
    emerald: 'bg-emerald-500',
    red: 'bg-red-500',
  };

  return (
    <div
      className="h-2 bg-slate-100 rounded-full overflow-hidden"
      style={{ width: `${width}px` }}
    >
      <div
        className={`h-full rounded-full transition-all duration-500 ease-out ${colorClasses[color]}`}
        style={{ width: `${percentage}%` }}
      />
    </div>
  );
}
