'use client';

import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';

import { MIN_WORDS, countWords } from '@/lib/text';
import type { AnalysisResult, ModelInfo } from '@/lib/types';
import { AnnotatedText } from './AnnotatedText';
import { Composer } from './Composer';
import { Button, Hint, Meter, Panel } from './primitives';
import { useAnalysis } from './useAnalysis';
import { useDraft } from './useDraft';

export function Workspace({
  model,
  models,
  threshold,
  maxWords,
  onSelectModel,
}: {
  model: ModelInfo;
  models: ModelInfo[];
  threshold: number;
  maxWords: number;
  onSelectModel: (modelId: string) => void;
}) {
  const draft = useDraft();
  const analysis = useAnalysis();
  const words = useMemo(() => countWords(draft.text), [draft.text]);

  return (
    <div>
      <Composer
        model={model}
        models={models}
        maxWords={maxWords}
        draft={draft}
        busy={analysis.busy}
        onRun={analysis.run}
        onCancel={analysis.cancel}
        onSelectModel={onSelectModel}
      />

      {analysis.phase === 'error' && analysis.error ? (
        <p className="mt-4 rounded-md border border-danger/40 bg-danger-soft px-4 py-3 text-base text-danger">
          {analysis.error}
        </p>
      ) : null}

      {analysis.busy ? (
        <Panel className="mt-6 p-5">
          <p className="text-base text-ink">
            {analysis.phase === 'loading'
              ? 'Loading the model into memory…'
              : `Scoring window ${Math.min(analysis.progress.done + 1, analysis.progress.total)} of ${analysis.progress.total}`}
          </p>
          <div className="mt-3">
            <Meter
              value={
                analysis.progress.total > 0
                  ? analysis.progress.done / analysis.progress.total
                  : 0.02
              }
            />
          </div>
          {analysis.scores.length > 0 ? (
            <p className="mt-3 text-sm text-ink-soft">
              {analysis.scores
                .slice(-6)
                .map((score) => `${Math.round(score.ai * 100)}%`)
                .join('  ')}
            </p>
          ) : null}
        </Panel>
      ) : null}

      {analysis.result ? (
        <ResultPanel
          result={analysis.result}
          text={analysis.analyzedText}
          stale={analysis.analyzedText !== draft.text}
          onReanalyse={() => analysis.run(draft.text)}
          canReanalyse={!analysis.busy && words >= MIN_WORDS && words <= maxWords}
          threshold={threshold}
        />
      ) : null}
    </div>
  );
}

const VERDICT_LINE: Record<string, string> = {
  human: 'Nothing here reads as machine-written',
  mixed: 'Partly human, partly machine',
  ai: 'Most of this reads as machine-written',
};

/** Passages shown before the list asks to be opened up. */
const PASSAGE_LIMIT = 12;

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function ResultPanel({
  result,
  text,
  stale,
  onReanalyse,
  canReanalyse,
  threshold,
}: {
  result: AnalysisResult;
  text: string;
  stale: boolean;
  onReanalyse: () => void;
  canReanalyse: boolean;
  threshold: number;
}) {
  // One sentence selected in either view; the annotated text and the passage
  // list are two windows onto the same score, so they share the pointer.
  const [active, setActive] = useState<number | null>(null);
  const [order, setOrder] = useState<'ranked' | 'document'>('ranked');
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState<'idle' | 'done' | 'failed'>('idle');
  const copyTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const shareHintId = useId();

  // Counted here rather than read off overall.flagged: the threshold can be
  // moved after a run, and the server counted with the one it was sent.
  const flagged = useMemo(
    () => result.segments.filter((segment) => segment.ai >= threshold),
    [result.segments, threshold],
  );

  // Document order is what flagged is already in, so only the other view sorts.
  const listed = useMemo(
    () => (order === 'ranked' ? [...flagged].sort((a, b) => b.ai - a.ai) : flagged),
    [flagged, order],
  );

  const visible = expanded ? listed : listed.slice(0, PASSAGE_LIMIT);
  const flaggedWords = useMemo(
    () => flagged.reduce((sum, segment) => sum + countWords(segment.text), 0),
    [flagged],
  );

  const copyFlagged = useCallback(async () => {
    if (copyTimer.current) clearTimeout(copyTimer.current);
    try {
      await navigator.clipboard.writeText(flagged.map((segment) => segment.text).join('\n\n'));
      setCopied('done');
    } catch {
      setCopied('failed');
    }
    copyTimer.current = setTimeout(() => setCopied('idle'), 2400);
  }, [flagged]);

  useEffect(
    () => () => {
      if (copyTimer.current) clearTimeout(copyTimer.current);
    },
    [],
  );

  return (
    <section className="mt-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="flex items-baseline gap-2.5">
            <span className="text-4xl font-semibold tracking-tight text-ink">
              {percent(result.overall.aiShare)}
            </span>
            <span className="text-lg text-ink-soft">machine-written</span>
            <Hint id={shareHintId}>
              How much of the document the detector reads as machine-written, weighted by sentence
              length: a long sentence moves the number more than a short one.
            </Hint>
          </p>
          <p className="mt-1 text-lg text-ink">{VERDICT_LINE[result.overall.verdict]}</p>
        </div>
        <p className="text-sm text-ink-soft">
          {flagged.length} of {result.segments.length} sentences over {percent(threshold)} ·{' '}
          {result.overall.words} words · {result.overall.windows} windows ·{' '}
          {(result.overall.elapsedMs / 1000).toFixed(1)}s
        </p>
      </div>

      <div className="mt-4">
        <Meter value={result.overall.aiShare} />
      </div>

      {stale ? (
        <Panel className="mt-6 border-warn/50 bg-warn-soft p-5">
          <p className="text-base text-ink">
            This result is from the text as it was when you analysed it. The text above has changed
            since.
          </p>
          <div className="mt-3">
            <Button variant="plain" onClick={onReanalyse} disabled={!canReanalyse}>
              Analyse the new text
            </Button>
          </div>
        </Panel>
      ) : null}

      <AnnotatedText
        text={text}
        segments={result.segments}
        threshold={threshold}
        active={active}
        onActivate={setActive}
      />

      {flagged.length > 0 ? (
        <Panel className="mt-4 p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h3 className="text-lg font-semibold text-ink">Passages over {percent(threshold)}</h3>
            <div className="flex flex-wrap items-center gap-2">
              <OrderSwitcher order={order} onSelect={setOrder} />
              <Button variant="quiet" onClick={() => void copyFlagged()}>
                {copied === 'done' ? 'Copied' : copied === 'failed' ? 'Copy failed' : 'Copy'}
              </Button>
            </div>
          </div>

          <p className="mt-1 text-sm text-ink-soft">
            {expanded || listed.length <= PASSAGE_LIMIT
              ? `${listed.length} ${listed.length === 1 ? 'passage' : 'passages'}`
              : `Showing ${PASSAGE_LIMIT} of ${listed.length}`}{' '}
            · {flaggedWords.toLocaleString()} of {result.overall.words.toLocaleString()} words
          </p>

          <ol className="mt-3 space-y-1">
            {visible.map((segment) => {
              const selected = active === segment.start;
              return (
                <li key={segment.start}>
                  <button
                    type="button"
                    aria-pressed={selected}
                    onClick={() => setActive(selected ? null : segment.start)}
                    className={`focusable flex w-full cursor-pointer gap-3 rounded-md border px-3 py-2 text-left transition-colors ${
                      selected
                        ? 'border-ink bg-paper'
                        : 'border-transparent hover:border-rule hover:bg-paper'
                    }`}
                  >
                    <span className="w-14 shrink-0 pt-0.5">
                      <span className="block text-right text-sm tabular-nums text-machine">
                        {percent(segment.ai)}
                      </span>
                      <span aria-hidden className="mt-1.5 block h-0.5 rounded-full bg-rule">
                        <span
                          className="block h-full rounded-full bg-machine"
                          style={{ width: `${Math.max(3, segment.ai * 100)}%` }}
                        />
                      </span>
                    </span>
                    <span className="text-base leading-relaxed text-ink">{segment.text}</span>
                  </button>
                </li>
              );
            })}
          </ol>

          {listed.length > PASSAGE_LIMIT ? (
            <div className="mt-3">
              <Button variant="plain" onClick={() => setExpanded((open) => !open)}>
                {expanded ? `Show only the strongest ${PASSAGE_LIMIT}` : `Show all ${listed.length}`}
              </Button>
            </div>
          ) : null}
        </Panel>
      ) : null}
    </section>
  );
}

/** Ranked by score or left in the order the sentences appear. */
function OrderSwitcher({
  order,
  onSelect,
}: {
  order: 'ranked' | 'document';
  onSelect: (order: 'ranked' | 'document') => void;
}) {
  return (
    <div
      role="radiogroup"
      aria-label="Order passages"
      className="flex items-center gap-1 rounded-lg border border-rule bg-paper p-1"
    >
      {(
        [
          ['ranked', 'Strongest first'],
          ['document', 'In the text'],
        ] as const
      ).map(([value, label]) => {
        const selected = order === value;
        return (
          <button
            key={value}
            type="button"
            role="radio"
            aria-checked={selected}
            onClick={() => onSelect(value)}
            className={`focusable cursor-pointer rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              selected ? 'bg-ink text-paper' : 'text-ink-soft hover:bg-paper-raised hover:text-ink'
            }`}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}
