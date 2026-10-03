import type { Settings } from './catalog';
import type { DownloadJob } from './downloads';

/** Shapes shared by the server routes and the browser bundle. */

export interface ModelInfo {
  id: string;
  name: string;
  tier: 'lite' | 'balanced' | 'deep';
  repo: string;
  license: string;
  kind: string;
  windowChars: number;
  bytes: number;
  bytesLabel: string;
  ramMb: number;
  detail: string;
  installed: boolean;
  installedAt: string | null;
  /** Bytes left over from an interrupted download, if any. */
  partialBytes: number;
}

export interface AppState {
  settings: Settings;
  downloads: DownloadJob[];
  models: ModelInfo[];
  /** Models whose weights the server currently holds in memory. */
  loadedModelIds: string[];
}

export interface SegmentResult {
  start: number;
  end: number;
  text: string;
  ai: number;
}

export interface AnalysisResult {
  model: { id: string; name: string; tier: string; repo: string; license: string };
  overall: {
    aiShare: number;
    verdict: 'human' | 'mixed' | 'ai';
    words: number;
    flagged: number;
    windows: number;
    elapsedMs: number;
  };
  segments: SegmentResult[];
}

export type AnalysisEvent =
  | { type: 'start'; model: AnalysisResult['model']; windows: number; sentences: number }
  | { type: 'progress'; done: number; total: number; phase: 'loading' | 'scoring' }
  | { type: 'window'; index: number; ai: number; start: number; end: number }
  | { type: 'error'; message: string }
  | { type: 'done'; result: AnalysisResult };