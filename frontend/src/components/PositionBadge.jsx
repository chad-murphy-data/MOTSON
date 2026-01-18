import React from 'react';

export default function PositionBadge({ position }) {
  const getClass = () => {
    if (position === 1) return 'gold';
    if (position === 2) return 'silver';
    if (position === 3) return 'bronze';
    if (position <= 4) return 'champions-league';
    if (position <= 6) return 'europa';
    if (position >= 18) return 'relegation';
    return 'default';
  };

  return (
    <span className={`position-badge ${getClass()}`}>
      {position}
    </span>
  );
}
