import React from 'react';
import { AlertCircle } from 'lucide-react';

export default function ErrorMessage({ message, retry }) {
  return (
    <div className="flex flex-col items-center justify-center py-8 text-center">
      <AlertCircle className="w-10 h-10 text-red-400 mb-3" />
      <p className="text-slate-600 mb-3">{message || 'Something went wrong'}</p>
      {retry && (
        <button
          onClick={retry}
          className="px-4 py-2 text-sm bg-primary-500 text-white rounded-lg hover:bg-primary-600"
        >
          Try Again
        </button>
      )}
    </div>
  );
}
