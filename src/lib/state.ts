import { MODELS, formatBytes } from './catalog';
import { downloads } from './downloads';
import { inspectInstall, inspectPartial } from './models';
import { readSettings } from './store';
import type { AppState, ModelInfo } from './types';

/** Everything the UI needs to render, used by the page and by GET /api/state. */
export async function loadState(): Promise<AppState> {
  const settings = await readSettings();
  const installed = await Promise.all(MODELS.map((spec) => inspectInstall(spec)));
  const partial = await Promise.all(MODELS.map((spec) => inspectPartial(spec)));

  return {
    settings,
    downloads: downloads.list().filter((job) => job.status !== 'done'),
    models: MODELS.map((spec, index): ModelInfo => {
      const info = installed[index];
      return {
        id: spec.id,
        name: spec.name,
        tier: spec.tier,
        repo: spec.repo,
        license: spec.license,
        kind: spec.kind,
        windowChars: spec.windowChars,
        wordsPerSecond: spec.wordsPerSecond,
        bytes: spec.bytes,
        bytesLabel: formatBytes(spec.bytes),
        ramMb: spec.ramMb,
        threshold: spec.threshold,
        detail: spec.detail,
        installed: info !== null,
        installedAt: info?.installedAt ?? null,
        partialBytes: partial[index]?.bytes ?? 0,
      };
    }),
  };
}