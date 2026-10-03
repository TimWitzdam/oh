'use client';

import { useMemo, useState } from 'react';

import type { AnalysisResult, ModelInfo } from '@/lib/types';
import { AnnotatedText } from './AnnotatedText';
import { Button, Meter, Panel } from './primitives';
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
  const analysis = useAnalysis();
  const words = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text]);

  const overLimit = words > maxWords;
  const canRun = words >= 40 && !analysis.busy && !overLimit;

  return (
    <div>
      <textarea
          value={text}
          onChange={(event) => setText(event.target.value)}
          spellCheck={false}
          aria-label="Text to analyse"
          placeholder="Paste the text you want to check. Nothing leaves this machine: the model runs locally on CPU."
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
              <p className="mt-3 font-mono text-sm text-ink-soft">
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

function ResultPanel({
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
  const flagged = useMemo(
    () =>
      result.segments
        .map((segment, index) => ({ segment, index }))
        .filter(({ segment }) => segment.ai >= threshold)
        .sort((a, b) => b.segment.ai - a.segment.ai),
    [result.segments, threshold],
  );

  return (
    <section className="mt-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="flex items-baseline gap-2.5">
            <span className="text-4xl font-semibold tracking-tight text-ink">
              {Math.round(result.overall.aiShare * 100)}%
            </span>
            <span className="text-lg text-ink-soft">machine-written</span>
          </p>
          <p className="mt-1 text-lg text-ink">{VERDICT_LINE[result.overall.verdict]}</p>
        </div>
        <p className="text-sm text-ink-soft">
          {result.overall.flagged} of {result.segments.length} sentences over {Math.round(threshold * 100)}% ·{' '}
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

      <Panel className="mt-6 p-5">
        <AnnotatedText text={text} segments={result.segments} threshold={threshold} />
      </Panel>

      {flagged.length > 0 ? (
        <Panel className="mt-4 p-5">
          <h3 className="text-lg font-semibold text-ink">Most machine-like passages</h3>
          <ol className="mt-3 space-y-3">
            {flagged.slice(0, 12).map(({ segment, index }) => (
              <li key={`${segment.start}-${index}`} className="flex gap-3">
                <span className="w-12 shrink-0 pt-0.5 text-right font-mono text-sm text-machine">
                  {Math.round(segment.ai * 100)}%
                </span>
                <span className="text-base leading-relaxed text-ink">{segment.text}</span>
              </li>
            ))}
          </ol>
        </Panel>
      ) : (
        <Panel className="mt-4 p-5">
          <p className="text-base text-ink-soft">
            Nothing reached the {Math.round(threshold * 100)}% highlight threshold.
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