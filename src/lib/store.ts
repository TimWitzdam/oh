import { promises as fs } from 'node:fs';

import { DEFAULT_SETTINGS, Settings, findModel } from './catalog';
import { DATA_DIR, SETTINGS_FILE } from './paths';

export async function readSettings(): Promise<Settings> {
  let stored: Partial<Settings> = {};
  try {
    stored = JSON.parse(await fs.readFile(SETTINGS_FILE, 'utf8')) as Partial<Settings>;
  } catch {
    stored = {};
  }
  return sanitize(stored);
}

export async function writeSettings(patch: Partial<Settings>): Promise<Settings> {
  const next = sanitize({ ...(await readSettings()), ...patch });
  await fs.mkdir(DATA_DIR, { recursive: true });
  const tmp = `${SETTINGS_FILE}.${process.pid}.tmp`;
  await fs.writeFile(tmp, `${JSON.stringify(next, null, 2)}\n`, 'utf8');
  await fs.rename(tmp, SETTINGS_FILE);
  return next;
}

/**
 * Settings written before thresholds moved into the catalog stored the one
 * shared default, 0.5. Keeping that value would pin every existing install to
 * the number this change exists to replace, so it is read as "no override" and
 * the active model's measured default takes over. Any other stored value was
 * chosen by a person and is left alone.
 */
const LEGACY_THRESHOLD = 0.5;

function sanitize(input: Partial<Settings>): Settings {
  const model = findModel(input.activeModelId ?? null);
  return {
    activeModelId: model ? model.id : null,
    windowChars: clamp(input.windowChars, 0, 1200, DEFAULT_SETTINGS.windowChars),
    // null keeps the active model's measured default.
    threshold: sanitizeThreshold(input.threshold),
    smoothing: Math.round(clamp(input.smoothing, 0, 3, DEFAULT_SETTINGS.smoothing)),
    maxWords: Math.round(clamp(input.maxWords, 100, 60_000, DEFAULT_SETTINGS.maxWords)),
    batchSize: Math.round(clamp(input.batchSize, 1, 32, DEFAULT_SETTINGS.batchSize)),
  };
}

function sanitizeThreshold(value: unknown): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  if (value === LEGACY_THRESHOLD) return null;
  return clamp(value, 0.1, 0.995, 0.5);
}

function clamp(value: unknown, min: number, max: number, fallback: number): number {
  const parsed = typeof value === 'number' && Number.isFinite(value) ? value : fallback;
  return Math.min(max, Math.max(min, parsed));
}