import { randomUUID } from 'node:crypto';
import { promises as fs } from 'node:fs';

import { DEFAULT_SETTINGS, Settings, findModel } from './catalog';
import { DATA_DIR, SETTINGS_FILE } from './paths';

/**
 * Bumped when the meaning of a stored field changes, so a migration runs once
 * on read instead of being re-guessed on every read and write. Written into
 * settings.json alongside the settings themselves.
 */
const SETTINGS_VERSION = 2;

/** The one threshold version 1 wrote for every model, override or not. */
const LEGACY_THRESHOLD = 0.5;

export async function readSettings(): Promise<Settings> {
  let stored: Record<string, unknown> = {};
  try {
    stored = JSON.parse(await fs.readFile(SETTINGS_FILE, 'utf8')) as Record<string, unknown>;
  } catch {
    stored = {};
  }
  return sanitize(migrate(stored));
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
    const stored = { version: SETTINGS_VERSION, ...settings };
    await fs.writeFile(tmp, `${JSON.stringify(stored, null, 2)}\n`, 'utf8');
    await fs.rename(tmp, SETTINGS_FILE);
  } catch (error) {
    // Leave nothing of a half-finished write behind in the data volume.
    await fs.rm(tmp, { force: true });
    throw error;
  }
  return settings;
}

/**
 * Version 1 had one shared highlight threshold of 0.5 for every model and no way
 * to tell "somebody chose 0.5" from "nobody chose anything", because the same
 * 0.5 was written either way. Version 2 gives each model its own measured
 * default, so a stored 0.5 is read as "no override" exactly once, which is the
 * only safe reading of an ambiguous value.
 *
 * This runs on read, keyed off the stored version, rather than on every write.
 * Keying it off the value instead meant 0.5 could never be stored again: the
 * write was thrown away as legacy, the setting came back as null, and the
 * slider snapped to the model default with no error and no way back. On Lite
 * (0.99) and Deep (0.9) that quietly made 50% unreachable.
 */
function migrate(stored: Record<string, unknown>): Partial<Settings> {
  const patch: Record<string, unknown> = { ...stored };
  const version = typeof patch.version === 'number' ? patch.version : 1;
  delete patch.version;
  if (version < 2 && patch.threshold === LEGACY_THRESHOLD) {
    patch.threshold = null;
  }
  return patch as Partial<Settings>;
}

function sanitize(input: Partial<Settings>): Settings {
  const model = findModel(input.activeModelId ?? null);
  return {
    activeModelId: model ? model.id : null,
    windowChars: clamp(input.windowChars, 0, 1200, DEFAULT_SETTINGS.windowChars),
    // null keeps the active model's measured default; any number, 0.5
    // included, is a real override the person asked for.
    threshold: sanitizeThreshold(input.threshold),
    smoothing: Math.round(clamp(input.smoothing, 0, 3, DEFAULT_SETTINGS.smoothing)),
    batchSize: Math.round(clamp(input.batchSize, 1, 32, DEFAULT_SETTINGS.batchSize)),
  };
}

function sanitizeThreshold(value: unknown): number | null {
  if (value === null || value === undefined) return null;
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return clamp(value, 0.1, 0.995, 0.5);
}

function clamp(value: unknown, min: number, max: number, fallback: number): number {
  const parsed = typeof value === 'number' && Number.isFinite(value) ? value : fallback;
  return Math.min(max, Math.max(min, parsed));
}