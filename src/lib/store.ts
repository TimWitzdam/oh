import { randomUUID } from 'node:crypto';
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

/**
 * Writes are queued rather than run side by side.
 *
 * A slider sends one of these per step of its drag, so several land at once.
 * Two problems came out of that: the read-modify-write interleaved and lost
 * whichever patch was based on the older read, and both writes shared one
 * temporary file - named after the process, which is the same for all of them -
 * so the first rename took the file the second was still writing and the second
 * came back as a 500.
 */
let queue: Promise<unknown> = Promise.resolve();

export function writeSettings(patch: Partial<Settings>): Promise<Settings> {
  const written = queue.then(
    () => write(patch),
    () => write(patch),
  );
  // The queue has to survive a failed write, or one error would reject every
  // patch behind it as well.
  queue = written.catch(() => undefined);
  return written;
}

async function write(patch: Partial<Settings>): Promise<Settings> {
  const settings = sanitize({ ...(await readSettings()), ...patch });
  await fs.mkdir(DATA_DIR, { recursive: true });
  const tmp = `${SETTINGS_FILE}.${randomUUID()}.tmp`;
  try {
    await fs.writeFile(tmp, `${JSON.stringify(settings, null, 2)}\n`, 'utf8');
    await fs.rename(tmp, SETTINGS_FILE);
  } catch (error) {
    // Leave nothing of a half-finished write behind in the data volume.
    await fs.rm(tmp, { force: true });
    throw error;
  }
  return settings;
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