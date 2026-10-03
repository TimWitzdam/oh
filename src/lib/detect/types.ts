import type { ModelSpec } from '../catalog';

export interface ScoreUpdate {
  index: number;
  ai: number;
}

export interface Detector {
  readonly spec: ModelSpec;
  /** Resolves once weights are in memory. Safe to call repeatedly. */
  ready(): Promise<void>;
  /** Yields one update per window, in window order. */
  score(windows: string[], signal: AbortSignal, batchSize: number): AsyncGenerator<ScoreUpdate>;
}