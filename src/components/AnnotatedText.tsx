'use client';

import { Fragment, useEffect, useMemo, useState } from 'react';

import type { SegmentResult } from '@/lib/types';
import { Button, Panel } from './primitives';

const MACHINE = [180, 68, 26] as const;

/** Machine colour mixed into the page background, 4% at zero to 55% at one. */
function tint(ai: number): string {
  const strength = 0.04 + Math.max(0, Math.min(1, ai)) * 0.51;
  return `rgba(${MACHINE[0]}, ${MACHINE[1]}, ${MACHINE[2]}, ${strength.toFixed(3)})`;
}

/** The same ramp the sentences are painted with, so the key cannot drift. */
function ramp(): string {
  const stops = [0, 0.25, 0.5, 0.75, 1].map((step) => `${tint(step)} ${step * 100}%`);
  return `linear-gradient(90deg, ${stops.join(', ')})`;
}

export function sentenceId(start: number): string {
  return `sentence-${start}`;
}

/**
 * Highlights each scored sentence inside the original text. The gaps between
 * sentences are emitted verbatim, so paragraph breaks and spacing survive and
 * the reader still sees the text as they pasted it.
 *
 * `active` is the one sentence currently selected, and it is owned by the parent
 * so the passage list and this view can point at each other. Hovering is a local
 * concern: it only decides which score bubble is showing, and it never touches
 * the parent's state.
 */
export function AnnotatedText({
  text,
  segments,
  threshold,
  active = null,
  onActivate,
}: {
  text: string;
  segments: SegmentResult[];
  threshold: number;
  active?: number | null;
  onActivate?: (start: number | null) => void;
}) {
  const [hovered, setHovered] = useState<number | null>(null);
  const [shading, setShading] = useState(true);

  // Pre-computed so the render pass stays free of mutation, and kept in a memo
  // so hovering does not re-slice the document on every pointer move.
  const { parts, tail } = useMemo(() => {
    if (segments.length === 0) return { parts: [], tail: '' };
    const built = segments.map((segment, index) => ({
      gap: text.slice(index === 0 ? 0 : segments[index - 1].end, segment.start),
      segment,
    }));
    return { parts: built, tail: text.slice(segments[segments.length - 1].end) };
  }, [text, segments]);

  // Selecting a passage in the list should bring the sentence with it, whether
  // it was picked with a mouse or from the keyboard.
  useEffect(() => {
    if (active === null) return;
    const node = document.getElementById(sentenceId(active));
    if (!node) return;
    const still = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
    node.scrollIntoView({ block: 'center', behavior: still ? 'auto' : 'smooth' });
  }, [active]);

  const body = 'whitespace-pre-wrap text-[1.0625rem] leading-[1.9] text-ink';

  if (segments.length === 0) {
    return (
      <Panel className="mt-4 p-5">
        <p className={body}>{text}</p>
      </Panel>
    );
  }

  // A selection beats a hover, so a reader who has picked a sentence keeps its
  // score on screen instead of losing it to the pointer.
  const scored = active ?? hovered;

  return (
    <Panel className="mt-4 p-5">
      <div className="mb-4 flex flex-wrap items-center gap-x-5 gap-y-2 border-b border-rule pb-3">
        <h3 className="text-lg font-semibold text-ink">The text</h3>

        {shading ? (
          <>
            <span className="flex items-center gap-2 text-xs text-ink-soft">
              <span>more human</span>
              <span
                aria-hidden
                className="h-2.5 w-24 rounded-full border border-rule"
                style={{ backgroundImage: ramp() }}
              />
              <span>more machine</span>
            </span>
            <span className="flex items-center gap-2 text-xs text-ink-soft">
              <span aria-hidden className="w-5 border-b-2 border-machine" />
              over {Math.round(threshold * 100)}%
            </span>
          </>
        ) : null}

        <span className="flex-1" />

        <Button variant="quiet" onClick={() => setShading((on) => !on)}>
          {shading ? 'Hide shading' : 'Show shading'}
        </Button>
      </div>

      <p className={body}>
        {parts.map(({ gap, segment }, index) => {
          const start = segment.start;
          const flagged = segment.ai >= threshold;
          const percent = Math.round(segment.ai * 100);
          const selected = active === start;
          const showScore = scored === start;

          return (
            <Fragment key={`${start}-${index}`}>
              {gap}
              <span
                id={sentenceId(start)}
                className={`sentence relative ${shading && flagged ? 'sentence-flagged' : ''} ${
                  selected ? 'sentence-active' : ''
                } ${onActivate ? 'cursor-pointer' : ''}`}
                style={shading ? { backgroundColor: tint(segment.ai) } : undefined}
                aria-label={`${percent}% machine-written`}
                onMouseEnter={() => setHovered(start)}
                onMouseLeave={() => setHovered((current) => (current === start ? null : current))}
                onClick={() => {
                  // A drag that happens to end over a sentence also ends in a
                  // click. Selecting text to copy it is not choosing it.
                  if (window.getSelection()?.isCollapsed === false) return;
                  onActivate?.(selected ? null : start);
                }}
              >
                {text.slice(segment.start, segment.end)}
                {showScore ? (
                  <span
                    aria-hidden
                    className="pointer-events-none absolute -top-7 left-0 z-20 rounded border border-rule-strong bg-paper-raised px-2 py-1 font-mono text-xs whitespace-nowrap text-ink shadow-sm"
                  >
                    {percent}% machine
                  </span>
                ) : null}
              </span>
            </Fragment>
          );
        })}
        {tail}
      </p>

      {onActivate ? (
        <p className="mt-4 text-sm text-ink-soft">
          Click a sentence to hold its score
          {segments.some((segment) => segment.ai >= threshold)
            ? ', or pick one from the passages below'
            : ''}
          .
        </p>
      ) : null}
    </Panel>
  );
}
