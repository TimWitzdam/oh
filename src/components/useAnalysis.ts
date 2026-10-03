'use client';

import { useCallback, useRef, useState } from 'react';

import { streamAnalysis } from '@/lib/client';
import type { AnalysisEvent, AnalysisResult } from '@/lib/types';

export type AnalysisPhase = 'idle' | 'loading' | 'scoring' | 'done' | 'error';

export interface WindowScore {
  index: number;
  ai: number;
}

export function useAnalysis() {
  const [phase, setPhase] = useState<AnalysisPhase>('idle');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [scores, setScores] = useState<WindowScore[]>([]);
  /** The text the current result was produced from; the textarea keeps moving. */
  const [analyzedText, setAnalyzedText] = useState('');
  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(async (text: string) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setPhase('loading');
    setResult(null);
    setError(null);
    setScores([]);
    setAnalyzedText(text);
    setProgress({ done: 0, total: 0 });

    let sawTerminal = false;
    const onEvent = (event: AnalysisEvent) => {
      switch (event.type) {
        case 'start':
          setProgress({ done: 0, total: event.windows });
          break;
        case 'progress':
          setPhase(event.phase);
          setProgress({ done: event.done, total: event.total });
          break;
        case 'window':
          setScores((previous) => [...previous, { index: event.index, ai: event.ai }]);
          break;
        case 'done':
          sawTerminal = true;
          setResult(event.result);
          setPhase('done');
          break;
        case 'error':
          // Counts as an answer: the stream ending after this is the normal
          // shape of a failed run, and the fallback below used to overwrite
          // this message with a generic one on the way out.
          sawTerminal = true;
          setError(event.message);
          setPhase('error');
          break;
      }
    };

    try {
      await streamAnalysis(text, onEvent, controller.signal);
      if (!sawTerminal) {
        setError('The analysis stream ended before a result came back.');
        setPhase('error');
      }
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === 'AbortError') {
        setPhase('idle');
        return;
      }
      setError(caught instanceof Error ? caught.message : String(caught));
      setPhase('error');
    } finally {
      abortRef.current = null;
    }
  }, []);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const busy = phase === 'loading' || phase === 'scoring';

  return { phase, busy, result, error, progress, scores, analyzedText, run, cancel };
}