import path from 'node:path';

function resolveDataDir(): string {
  const configured = process.env.OH_DATA_DIR?.trim();
  if (configured) return path.resolve(configured);
  return path.resolve(process.cwd(), 'data');
}

export const DATA_DIR = resolveDataDir();
export const MODELS_DIR = path.join(DATA_DIR, 'models');
export const SETTINGS_FILE = path.join(DATA_DIR, 'settings.json');

/** Python inference service, loopback only, not published by the container. */
export const INFERENCE_URL = process.env.OH_INFERENCE_URL?.trim() || 'http://127.0.0.1:8001';