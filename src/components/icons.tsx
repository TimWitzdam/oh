'use client';

import type { ModelInfo } from '@/lib/types';

/** Hand-drawn tier glyphs; no icon dependency, no network request. */

export function TierIcon({ tier, className = '' }: { tier: ModelInfo['tier']; className?: string }) {
  const common = {
    width: 20,
    height: 20,
    viewBox: '0 0 20 20',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: 1.6,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
    className,
    'aria-hidden': true,
  };

  if (tier === 'lite') {
    // bolt: the quick tier
    return (
      <svg {...common}>
        <path d="M11 2 4.5 11H9l-1 7 6.5-9H10l1-7Z" />
      </svg>
    );
  }

  if (tier === 'balanced') {
    // balance scale: the recommended middle
    return (
      <svg {...common}>
        <path d="M10 3v14M5 17h10M3 8h14M10 5 3 8m7-3 7 3" />
        <path d="M3 8 1.2 12.4h3.6L3 8Zm14 0-1.8 4.4h3.6L17 8Z" />
      </svg>
    );
  }

  // magnifier over text lines: the careful tier
  return (
    <svg {...common}>
      <path d="M3 4h9M3 7.5h6M3 11h5" />
      <circle cx="12.5" cy="11.5" r="4.5" />
      <path d="m15.8 14.8 2.7 2.7" />
    </svg>
  );
}

export function CheckIcon({ className = '' }: { className?: string }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden
    >
      <path d="m3 8.5 3.2 3.2L13 5" />
    </svg>
  );
}

export function DownloadIcon({ className = '' }: { className?: string }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden
    >
      <path d="M8 2v8m0 0 3-3m-3 3L5 7M2.5 12.5h11" />
    </svg>
  );
}