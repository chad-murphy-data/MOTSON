import React from 'react';

export default function LoadingSpinner({ size = 'md' }) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
  };

  return (
    <div className="flex items-center justify-center py-8">
      <div
        className={`${sizeClasses[size]} border-2 border-primary-200 border-t-primary-500 rounded-full animate-spin`}
      />
    </div>
  );
}
