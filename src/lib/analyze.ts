import { effectiveThreshold, type ModelSpec, type Settings } from './catalog';
import type { AnalysisEvent, AnalysisResult, SegmentResult } from './types';
import { createDetector } from './detect';
import { throwIfAborted } from './detect/onnx';
import type { Detector } from './detect/types';
import { stripFormatting } from './normalize';
import { buildWindows, splitSentences } from './segment';

let inflight = 0;

// Route handlers are bundled separately, so the cache hangs off globalThis to
// make sure every handler looks at the same set of resident weights.
const store = globalThis as typeof globalThis & { __ohDetectors?: Map<string, Detector> };

/** One detector per model id, reused across requests so weights load once. */
const cache: Map<string, Detector> = (store.__ohDetectors ??= new Map());

export function detectorFor(spec: ModelSpec): Detector {
  const existing = cache.get(spec.id);
  if (existing) return existing;
  const detector = createDetector(spec);
  cache.set(spec.id, detector);
  return detector;
}

export function activeInFlight(): number {
  return inflight;
}

export async function* analyze(
  text: string,
  spec: ModelSpec,
  settings: Settings,
  signal: AbortSignal,
): AsyncGenerator<AnalysisEvent> {
  const startedAt = Date.now();
  const sentences = splitSentences(text);
  const windowChars = settings.windowChars || spec.windowChars;
  const windows = buildWindows(sentences, windowChars, spec.maxTokens * 3);

  if (sentences.length === 0 || windows.length === 0) {
    throw new Error('nothing to score: no sentences found');
  }

  if (process.env.OH_DEBUG_WINDOWS) {
    for (const window of windows) {
      console.log(`[window ${window.index}] ${JSON.stringify(window.text.slice(0, 120))}`);
    }
  }

  const model = {
    id: spec.id,
    name: spec.name,
    tier: spec.tier,
    repo: spec.repo,
    license: spec.license,
  };
  yield { type: 'start', model, windows: windows.length, sentences: sentences.length };

  const detector = detectorFor(spec);
  inflight += 1;
  try {
    yield { type: 'progress', done: 0, total: windows.length, phase: 'loading' };
    await detector.ready();
    yield { type: 'progress', done: 0, total: windows.length, phase: 'scoring' };

    const scores = new Map<number, number>();
    // Models see prose: markdown markers are stripped from the text handed to
    // them, while the document keeps its original characters for highlighting.
    const scoringText = windows.map((window) => stripFormatting(window.text));

    for await (const update of detector.score(scoringText, signal, settings.batchSize)) {
      throwIfAborted(signal);
      scores.set(update.index, update.ai);
      const window = windows[update.index];
      yield {
        type: 'window',
        index: update.index,
        ai: update.ai,
        start: window.start,
        end: window.end,
      };
      yield {
        type: 'progress',
        done: scores.size,
        total: windows.length,
        phase: 'scoring',
      };
    }

    const threshold = effectiveThreshold(settings, spec);
    const segments = aggregate(sentences, windows, scores, settings.smoothing, threshold);
    const result = summarize(segments, text, model, windows.length, startedAt, threshold);
    yield { type: 'done', result };
  } finally {
    inflight -= 1;
  }
}

function aggregate(
  sentences: ReturnType<typeof splitSentences>,
  windows: ReturnType<typeof buildWindows>,
  scores: Map<number, number>,
  smoothing: number,
  threshold: number,
): SegmentResult[] {
  const weights = new Array<number>(sentences.length).fill(0);
  const totals = new Array<number>(sentences.length).fill(0);

  for (const window of windows) {
    const score = scores.get(window.index);
    if (score === undefined) continue;
    for (const index of window.sentences) {
      const overlap = Math.min(sentences[index].end, window.end) - Math.max(sentences[index].start, window.start);
      if (overlap <= 0) continue;
      totals[index] += score * overlap;
      weights[index] += overlap;
    }
  }

  // Windows produced from an over-long sentence carry no sentence indexes; the
  // best available signal is the score of the window covering that span.
  for (const window of windows) {
    if (scores.get(window.index) === undefined) continue;
    for (let i = 0; i < sentences.length; i += 1) {
      if (weights[i] > 0) continue;
      const overlap = Math.min(sentences[i].end, window.end) - Math.max(sentences[i].start, window.start);
      if (overlap > 0) {
        totals[i] = scores.get(window.index)! * overlap;
        weights[i] = overlap;
      }
    }
  }

  // A sentence no window covered has no score at all. Parking it on the
  // threshold reads as "exactly at the line", which the UI then lists as
  // flagged on a technicality.
  const raw = weights.map((weight, index) => (weight > 0 ? totals[index] / weight : threshold * 0.99));

  return sentences.map((sentence, index) => {
    const from = Math.max(0, index - smoothing);
    const to = Math.min(raw.length - 1, index + smoothing);
    let sum = 0;
    for (let i = from; i <= to; i += 1) sum += raw[i];
    return {
      start: sentence.start,
      end: sentence.end,
      text: sentence.text,
      ai: sum / (to - from + 1),
    };
  });
}

function summarize(
  segments: SegmentResult[],
  text: string,
  model: AnalysisResult['model'],
  windowCount: number,
  startedAt: number,
  threshold: number,
): AnalysisResult {
  let aiChars = 0;
  let totalChars = 0;
  let flagged = 0;

  for (const segment of segments) {
    const length = segment.end - segment.start;
    totalChars += length;
    aiChars += segment.ai * length;
    if (segment.ai >= threshold) flagged += 1;
  }

  const aiShare = totalChars > 0 ? aiChars / totalChars : 0;

  return {
    model,
    overall: {
      aiShare,
      verdict: aiShare >= 0.6 ? 'ai' : aiShare >= 0.3 ? 'mixed' : 'human',
      words: countWords(text),
      flagged,
      windows: windowCount,
      elapsedMs: Date.now() - startedAt,
    },
    segments,
  };
}

function countWords(text: string): number {
  const trimmed = text.trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
}

