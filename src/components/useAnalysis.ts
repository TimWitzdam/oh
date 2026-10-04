'use client';

import { useCallback, useSyncExternalStore } from 'react';

import { streamAnalysis } from '@/lib/client';
import type { AnalysisEvent, AnalysisResult } from '@/lib/types';

export type AnalysisPhase = 'idle' | 'loading' | 'scoring' | 'done' | 'error';

export interface WindowScore {
  index: number;
  ai: number;
}

export interface AnalysisState {
  phase: AnalysisPhase;
  result: AnalysisResult | null;
  error: string | null;
  progress: { done: number; total: number };
  scores: WindowScore[];
  /** The text the current result was produced from; the textarea keeps moving. */
  analyzedText: string;
}

const IDLE: AnalysisState = {
  phase: 'idle',
  result: null,
  error: null,
  progress: { done: 0, total: 0 },
  scores: [],
  analyzedText: '',
};

/**
 * The run lives here rather than in the workspace component.
 *
 * Opening the detector list unmounts the workspace, but the server does not
 * notice: it keeps scoring windows for a request whose browser is still there,
 * just not looking. State inside the component meant that trip threw away the
 * progress and the finished result, and coming back and pressing Analyse again
 * answered "another analysis is already running" for the reader's own run. So
 * the store outlives the component - the same shape as the draft store in
 * useDraft.ts - and unmounting is never treated as a cancel.
 */
let state: AnalysisState = IDLE;
let inflight: AbortController | null = null;
const listeners = new Set<() => void>();

function patch(update: Partial<AnalysisState>): void {
  state = { ...state, ...update };
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Stable across renders, and identical to what the server rendered. */
function read(): AnalysisState {
  return state;
}

function serverSnapshot(): AnalysisState {
  return IDLE;
}

async function run(text: string): Promise<void> {
  inflight?.abort();
  const controller = new AbortController();
  inflight = controller;

  // A second run replaces the first, and the first one is finished with by the
  // time its stream unwinds. Anything it still wants to say afterwards is about
  // a result nobody is waiting for, so it goes out rather than into the new
  // run's screen.
  const superseded = () => inflight !== controller;

  patch({
    phase: 'loading',
    result: null,
    error: null,
    scores: [],
    analyzedText: text,
    progress: { done: 0, total: 0 },
  });

  let sawTerminal = false;
  const onEvent = (event: AnalysisEvent) => {
    if (superseded()) return;
    switch (event.type) {
      case 'start':
        patch({ progress: { done: 0, total: event.windows } });
        break;
      case 'progress':
        patch({ phase: event.phase, progress: { done: event.done, total: event.total } });
        break;
      case 'window':
        patch({ scores: [...state.scores, { index: event.index, ai: event.ai }] });
        break;
      case 'done':
        sawTerminal = true;
        patch({ result: event.result, phase: 'done' });
        break;
      case 'error':
        // Counts as an answer: the stream ending after this is the normal
        // shape of a failed run, and the fallback below used to overwrite
        // this message with a generic one on the way out.
        sawTerminal = true;
        patch({ error: event.message, phase: 'error' });
        break;
    }
  };

  try {
    await streamAnalysis(text, onEvent, controller.signal);
    if (superseded()) return;
    if (!sawTerminal) {
      patch({ error: 'The analysis stream ended before a result came back.', phase: 'error' });
    }
  } catch (caught) {
    if (superseded()) return;
    if (caught instanceof DOMException && caught.name === 'AbortError') {
      patch({ phase: 'idle', scores: [], progress: { done: 0, total: 0 } });
      return;
    }
    patch({ error: caught instanceof Error ? caught.message : String(caught), phase: 'error' });
  } finally {
    if (inflight === controller) inflight = null;
  }
}

function cancel(): void {
  inflight?.abort();
}

export function useAnalysis() {
  const current = useSyncExternalStore(subscribe, read, serverSnapshot);
  const onRun = useCallback((text: string) => {
    void run(text);
  }, []);
  const onCancel = useCallback(() => {
    cancel();
  }, []);

  const busy = current.phase === 'loading' || current.phase === 'scoring';

  return { ...current, busy, run: onRun, cancel: onCancel };
}
