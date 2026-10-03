'use client';

import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';

import { extractPdfText } from '@/lib/client';
import type { AnalysisResult, ModelInfo, SegmentResult } from '@/lib/types';
import { AnnotatedText } from './AnnotatedText';
import { Button, FileButton, Hint, Meter, Panel } from './primitives';
import { useAnalysis } from './useAnalysis';

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
  const [text, setText] = useState('');
  const [importing, setImporting] = useState(false);
  const [importNote, setImportNote] = useState<string | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const analysis = useAnalysis();
  const words = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text]);

  const overLimit = words > maxWords;
  const canRun = words >= 40 && !analysis.busy && !overLimit;

  // The PDF replaces the text rather than joining it: two documents scored as
  // one would produce a number that belongs to neither.
  const handlePdf = useCallback(async (file: File) => {
    setImporting(true);
    setImportNote(null);
    setImportError(null);
    try {
      const extracted = await extractPdfText(file);
      setText(extracted.text);
      const count = extracted.text.trim() ? extracted.text.trim().split(/\s+/).length : 0;
      setImportNote(
        `${file.name} — ${extracted.pages} ${extracted.pages === 1 ? 'page' : 'pages'}, ${count} words${
          extracted.truncated ? ', cut at the character limit' : ''
        }.`,
      );
    } catch (caught) {
      setImportError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setImporting(false);
    }
  }, []);

  return (
    <div>
      <textarea
          value={text}
          onChange={(event) => setText(event.target.value)}
          spellCheck={false}
          aria-label="Text to analyse"
          placeholder="Paste the text you want to check, or add a PDF. Nothing leaves this machine: the model runs locally on CPU."
          className="focusable min-h-[42vh] w-full resize-y rounded-lg border border-rule bg-paper-raised p-5 text-base leading-relaxed text-ink placeholder:text-ink-faint"
        />

        <div className="mt-3 flex flex-wrap items-center gap-3">
          {analysis.busy ? (
            <Button variant="plain" onClick={analysis.cancel}>
              Cancel analysis
            </Button>
          ) : (
            <Button variant="primary" onClick={() => analysis.run(text)} disabled={!canRun}>
              Analyse text
            </Button>
          )}
          <Button variant="quiet" onClick={() => setText('')} disabled={!text}>
            Clear
          </Button>
          <FileButton
            accept="application/pdf,.pdf"
            onFile={(file) => void handlePdf(file)}
            disabled={importing || analysis.busy}
          >
            {importing ? 'Reading PDF…' : 'Add PDF'}
          </FileButton>
          <span className={`text-sm ${overLimit ? 'text-danger' : 'text-ink-soft'}`}>
            {words} {words === 1 ? 'word' : 'words'}
            {overLimit ? ` — over the ${maxWords} word limit` : ''}
          </span>

          <span className="flex-1" />

          <ModelSwitcher models={models} activeId={model.id} onSelect={onSelectModel} />
        </div>

        {overLimit ? null : words > 0 && words < 40 ? (
          <p className="mt-3 text-sm text-ink-soft">
            Add a little more text — scores below a paragraph are unreliable.
          </p>
        ) : null}

        {importNote ? <p className="mt-3 text-sm text-ink-soft">{importNote}</p> : null}

        {importError ? (
          <p className="mt-3 rounded-md border border-danger/40 bg-danger-soft px-4 py-3 text-base text-danger">
            {importError}
          </p>
        ) : null}

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
            stale={analysis.analyzedText !== text}
            onReanalyse={() => analysis.run(text)}
            canReanalyse={!analysis.busy && words >= 40 && !overLimit}
            threshold={threshold}
          />
        ) : null}
    </div>
  );
}

/** Compact tier switcher: the three detectors, only the installed ones live. */
function ModelSwitcher({
  models,
  activeId,
  onSelect,
}: {
  models: ModelInfo[];
  activeId: string;
  onSelect: (modelId: string) => void;
}) {
  return (
    <div
      role="radiogroup"
      aria-label="Detector"
      className="flex items-center gap-1 rounded-lg border border-rule bg-paper-raised p-1"
    >
      {models.map((candidate) => {
        const selected = candidate.id === activeId;
        const usable = candidate.installed;
        return (
          <button
            key={candidate.id}
            type="button"
            role="radio"
            aria-checked={selected}
            disabled={!usable}
            onClick={() => onSelect(candidate.id)}
            className={`focusable cursor-pointer rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              selected
                ? 'bg-ink text-paper'
                : usable
                  ? 'text-ink-soft hover:bg-paper hover:text-ink'
                  : 'cursor-not-allowed text-ink-faint opacity-50'
            }`}
          >
            {candidate.name}
          </button>
        );
      })}
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

/** Ten buckets of ten percent, which is as fine as a sentence count gets. */
const SPREAD_BINS = 10;

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function countWords(text: string): number {
  const trimmed = text.trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
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

      <div className="mt-4">
        <ScoreSpread segments={result.segments} threshold={threshold} />
      </div>

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
      ) : (
        <Panel className="mt-4 p-5">
          <p className="text-base text-ink-soft">
            Nothing reached the {percent(threshold)} highlight threshold. The detector reads all{' '}
            {result.segments.length} sentences as human; lower the threshold in Settings to see its
            strongest passages anyway.
          </p>
        </Panel>
      )}

      <p className="mt-4 text-sm text-ink-soft">
        Scored with {result.model.repo} ({result.model.license}). A score reflects how much the
        text looks machine-written, not proof of who typed it.
      </p>
    </section>
  );
}

/**
 * The shape of the scores rather than their sum: one bar per ten percent of the
 * scale, tall enough to see a document that is uniformly one thing, and the
 * highlight threshold ruled across it.
 */
function ScoreSpread({
  segments,
  threshold,
}: {
  segments: SegmentResult[];
  threshold: number;
}) {
  const counts = useMemo(() => {
    const bins = new Array<number>(SPREAD_BINS).fill(0);
    for (const segment of segments) {
      const bin = Math.min(SPREAD_BINS - 1, Math.max(0, Math.floor(segment.ai * SPREAD_BINS)));
      bins[bin] += 1;
    }
    return bins;
  }, [segments]);

  const scores = useMemo(() => segments.map((segment) => segment.ai).sort((a, b) => a - b), [segments]);
  const median = scores.length > 0 ? scores[Math.floor(scores.length / 2)] : 0;
  const peak = Math.max(1, ...counts);
  // A bin at or past the line belongs to the machine side of it.
  const cut = threshold * SPREAD_BINS;

  return (
    <Panel className="p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h3 className="text-lg font-semibold text-ink">Where the sentences fall</h3>
        <p className="text-sm text-ink-soft">
          median {percent(median)}
          {scores.length > 0 ? ` · highest ${percent(scores[scores.length - 1])}` : ''}
        </p>
      </div>

      <div className="relative mt-4 flex h-24 items-end gap-1">
        {counts.map((count, index) => (
          <div key={index} className="flex h-full flex-1 flex-col justify-end">
            <span className="shrink-0 text-center text-[0.625rem] leading-3 tabular-nums text-ink-faint">
              {count > 0 ? count : ''}
            </span>
            <span
              title={`${index * 10}–${index * 10 + 10}%: ${count} ${
                count === 1 ? 'sentence' : 'sentences'
              }`}
              className={`w-full rounded-t-sm ${index >= cut ? 'bg-machine' : 'bg-human'}`}
              // Stop short of the full height so the tallest bar still has its
              // count above it, inside the panel.
              style={{ height: `${(count / peak) * 86}%`, minHeight: count > 0 ? 3 : 0 }}
            />
          </div>
        ))}
        <span
          aria-hidden
          className="absolute -top-1.5 bottom-0 border-l border-dashed border-warn"
          style={{ left: `${threshold * 100}%` }}
        />
      </div>

      <div className="mt-2 flex justify-between text-xs text-ink-faint">
        <span>0% machine</span>
        <span>100%</span>
      </div>

      <p className="mt-3 text-sm text-ink-soft">
        Every bar is ten percent of the scale wide and counts the sentences that landed in it. The
        dashed line is the {percent(threshold)} highlight threshold, and everything to the right of
        it is underlined in the text below.
      </p>
    </Panel>
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
