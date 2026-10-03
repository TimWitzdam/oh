import { promises as fs } from 'node:fs';
import path from 'node:path';

import type { CatalogFile, ModelSpec } from './catalog';
import { MODELS_DIR } from './paths';

export interface ModelManifest {
  id: string;
  repo: string;
  kind: ModelSpec['kind'];
  dtype?: string;
  aiIndex: number;
  temperature?: number;
  quantize?: boolean;
  readout?: ModelSpec['readout'];
  windowChars: number;
  maxTokens: number;
  files: string[];
  bytes: number;
  installedAt: string;
}

const MANIFEST_NAME = 'oh.json';

export function modelDir(id: string): string {
  return path.join(MODELS_DIR, id);
}

export function manifestPath(id: string): string {
  return path.join(modelDir(id), MANIFEST_NAME);
}

export function buildManifest(spec: ModelSpec): ModelManifest {
  return {
    id: spec.id,
    repo: spec.repo,
    kind: spec.kind,
    dtype: spec.dtype,
    aiIndex: spec.aiIndex,
    temperature: spec.temperature,
    quantize: spec.quantize,
    readout: spec.readout,
    windowChars: spec.windowChars,
    maxTokens: spec.maxTokens,
    files: spec.files.map((file) => file.path),
    bytes: spec.bytes,
    installedAt: new Date().toISOString(),
  };
}

export async function readManifest(id: string): Promise<ModelManifest | null> {
  try {
    return JSON.parse(await fs.readFile(manifestPath(id), 'utf8')) as ModelManifest;
  } catch {
    return null;
  }
}

export async function writeManifest(id: string, manifest: ModelManifest): Promise<void> {
  await fs.mkdir(modelDir(id), { recursive: true });
  await fs.writeFile(manifestPath(id), `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');
}

export interface InstalledInfo {
  installedAt: string;
}

export interface PartialInfo {
  bytes: number;
}

/**
 * A model counts as installed when its manifest and every listed file are
 * present at the published size. Anything left over from an interrupted
 * download is reported separately so the picker can offer a resume.
 */
export async function inspectInstall(spec: ModelSpec): Promise<InstalledInfo | null> {
  const manifest = await readManifest(spec.id);
  if (!manifest) return null;
  const sizes = await Promise.all(
    manifest.files.map((file) => sizeOf(path.join(modelDir(spec.id), file))),
  );
  const expected = new Map(spec.files.map((file: CatalogFile) => [file.path, file.bytes]));
  const complete = sizes.every((size, index) => {
    const want = expected.get(manifest.files[index]);
    if (want === undefined) return size > 0;
    return size >= 0 && Math.abs(size - want) <= tolerance(want);
  });
  return complete ? { installedAt: manifest.installedAt } : null;
}

/** Bytes already fetched for files that never finished, across restarts. */
export async function inspectPartial(spec: ModelSpec): Promise<PartialInfo | null> {
  let bytes = 0;
  for (const file of spec.files) {
    const part = await sizeOf(path.join(modelDir(spec.id), `${file.path}.part`));
    bytes += part > 0 ? part : 0;
  }
  return bytes > 0 ? { bytes } : null;
}

/** How far a file may deviate from its published size before we call it corrupt. */
function tolerance(expected: number): number {
  return Math.max(4096, expected * 0.01);
}

async function sizeOf(target: string): Promise<number> {
  try {
    return (await fs.stat(target)).size;
  } catch {
    return -1;
  }
}

export async function removeModel(id: string): Promise<void> {
  await fs.rm(modelDir(id), { recursive: true, force: true });
}