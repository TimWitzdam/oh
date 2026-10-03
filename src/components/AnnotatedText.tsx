'use client';

import { Fragment } from 'react';

import type { SegmentResult } from '@/lib/types';

const MACHINE = [180, 68, 26] as const;

/** Machine colour mixed into the page background, 4% at zero to 55% at one. */
function tint(ai: number): string {
  const strength = 0.04 + Math.max(0, Math.min(1, ai)) * 0.51;
  return `rgba(${MACHINE[0]}, ${MACHINE[1]}, ${MACHINE[2]}, ${strength.toFixed(3)})`;
}

/**
 * Highlights each scored sentence inside the original text. The gaps between
 * sentences are emitted verbatim, so paragraph breaks and spacing survive and
 * the reader still sees the text as they pasted it.
 */
export function AnnotatedText({
  text,
  segments,
  threshold,
}: {
  text: string;
  segments: SegmentResult[];
  threshold: number;
}) {
  if (segments.length === 0) {
    return (
      <p className="whitespace-pre-wrap text-[1.0625rem] leading-[1.9] text-ink">{text}</p>
    );
  }

  // Pre-computed so the render pass stays free of mutation.
  const parts = segments.map((segment, index) => ({
    gap: text.slice(index === 0 ? 0 : segments[index - 1].end, segment.start),
    segment,
  }));
  const tail = text.slice(segments[segments.length - 1].end);

  return (
    <p className="whitespace-pre-wrap text-[1.0625rem] leading-[1.9] text-ink">
      {parts.map(({ gap, segment }, index) => {
        const flagged = segment.ai >= threshold;
        const percent = Math.round(segment.ai * 100);

        return (
          <Fragment key={`${segment.start}-${index}`}>
            {gap}
            <span className="group relative inline">
              <span
                className={`sentence ${flagged ? 'sentence-flagged' : ''}`}
                style={{ backgroundColor: tint(segment.ai) }}
                aria-label={`${percent}% machine-written`}
              >
                {text.slice(segment.start, segment.end)}
              </span>
              <span
                aria-hidden
                className="pointer-events-none absolute -top-6 left-0 z-10 hidden rounded border border-rule-strong bg-paper-raised px-2 py-1 font-mono text-xs whitespace-nowrap text-ink group-hover:block"
              >
                {percent}% machine
              </span>
            </span>
          </Fragment>
        );
      })}
      {tail}
    </p>
  );
}