import React from 'react';
import './ArtworkPreviewFallback.css';

export default function ArtworkPreviewFallback({ compact = false }) {
  return (
    <div className={`artwork-preview-fallback${compact ? ' compact' : ''}`} role="img" aria-label="Artwork preview unavailable">
      <svg viewBox="0 0 120 90" aria-hidden="true">
        <path className="fallback-frame" d="M18 12h84a8 8 0 0 1 8 8v50a8 8 0 0 1-8 8H18a8 8 0 0 1-8-8V20a8 8 0 0 1 8-8Z" />
        <path className="fallback-mountain" d="m19 65 22-22 14 14 13-13 33 27H18Z" />
        <circle className="fallback-eye" cx="46" cy="31" r="3" />
        <circle className="fallback-eye" cx="74" cy="31" r="3" />
        <path className="fallback-mouth" d="M49 41c7-6 15-6 22 0" />
        <path className="fallback-break" d="m63 10-8 15 10 10-11 16 9 12-7 17" />
      </svg>
      <span>Preview unavailable</span>
    </div>
  );
}
