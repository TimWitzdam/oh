import { promises as fs } from 'node:fs';
import path from 'node:path';

import { ModelSpec, findModel } from './catalog';
import { buildManifest, modelDir, writeManifest } from './models';

export type DownloadStatus =
  | 'queued'
  | 'downloading'
  | 'verifying'
  | 'done'
  | 'canceled'
  | 'error';

export interface DownloadJob {
  id: string;
  modelId: string;
  status: DownloadStatus;
  bytesReceived: number;
  bytesTotal: number;
  bytesPerSecond: number;
  etaSeconds: number | null;
  currentFile: string | null;
  error: string | null;
  createdAt: string;
  finishedAt: string | null;
}

interface InternalJob extends DownloadJob {
  controller: AbortController;
  listeners: Set<(job: DownloadJob) => void>;
}

const HUB = process.env.OH_HF_ENDPOINT?.trim() || 'https://huggingface.co';

class DownloadManager {
  private jobs = new Map<string, InternalJob>();

  list(): DownloadJob[] {
    return [...this.jobs.values()]
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
      .map(publicJob);
  }

  get(jobId: string): DownloadJob | null {
    const job = this.jobs.get(jobId);
    return job ? publicJob(job) : null;
  }

  activeFor(modelId: string): DownloadJob | null {
    for (const job of this.jobs.values()) {
      if (job.modelId === modelId && (job.status === 'downloading' || job.status === 'queued')) {
        return publicJob(job);
      }
    }
    return null;
  }

  subscribe(jobId: string, listener: (job: DownloadJob) => void): (() => void) | null {
    const job = this.jobs.get(jobId);
    if (!job) return null;
    job.listeners.add(listener);
    return () => job.listeners.delete(listener);
  }

  start(spec: ModelSpec): DownloadJob {
    const existing = this.activeFor(spec.id);
    if (existing) return existing;

    const job: InternalJob = {
      id: `${spec.id}-${Date.now().toString(36)}`,
      modelId: spec.id,
      status: 'queued',
      bytesReceived: 0,
      bytesTotal: spec.bytes,
      bytesPerSecond: 0,
      etaSeconds: null,
      currentFile: null,
      error: null,
      createdAt: new Date().toISOString(),
      finishedAt: null,
      controller: new AbortController(),
      listeners: new Set(),
    };
    this.jobs.set(job.id, job);
    this.notify(job);

    void this.run(job, spec).catch((error: unknown) => {
      this.finish(job, 'error', error instanceof Error ? error.message : String(error));
    });

    return publicJob(job);
  }

  cancel(jobId: string): DownloadJob | null {
    const job = this.jobs.get(jobId);
    if (!job) return null;
    if (job.status === 'done' || job.status === 'canceled') return publicJob(job);
    job.controller.abort();
    this.finish(job, 'canceled', null);
    return publicJob(job);
  }

  /** Cancels any job for the model and forgets partial progress. */
  async purge(spec: ModelSpec): Promise<void> {
    for (const job of [...this.jobs.values()]) {
      if (job.modelId === spec.id) {
        job.controller.abort();
        this.jobs.delete(job.id);
      }
    }
    await fs.rm(modelDir(spec.id), { recursive: true, force: true });
  }

  private async run(job: InternalJob, spec: ModelSpec): Promise<void> {
    const dir = modelDir(spec.id);
    await fs.mkdir(dir, { recursive: true });
    job.status = 'downloading';
    this.notify(job);

    const startedAt = Date.now();
    job.bytesReceived = 0;
    for (const file of spec.files) {
      const target = path.join(dir, file.path);
      const present = await fileSize(target);
      if (present >= 0 && Math.abs(present - file.bytes) <= tolerance(file.bytes)) {
        // Already on disk from an earlier attempt: count it and move on so a
        // restart after a failure only fetches what is missing.
        job.bytesReceived += present;
        continue;
      }

      const partPath = `${target}.part`;
      await fs.mkdir(path.dirname(partPath), { recursive: true });

      let offset = await fileSize(partPath);
      job.currentFile = file.path;
      if (offset > 0) {
        // Bytes fetched by an earlier attempt still count towards this one.
        job.bytesReceived += offset;
      }
      this.notify(job);

      const url = `${HUB}/${spec.repo}/resolve/main/${file.path}`;
      const headers: Record<string, string> = { 'user-agent': 'oh' };
      if (offset > 0) headers.range = `bytes=${offset}-`;

      const response = await fetch(url, { headers, signal: job.controller.signal });
      if (!response.ok && response.status !== 206) {
        throw new Error(`${file.path}: HTTP ${response.status} from ${url}`);
      }
      if (response.status !== 206) offset = 0;

      const declared = Number(response.headers.get('content-length') ?? 0);
      job.bytesTotal = spec.bytes;
      this.notify(job);

      await writeStream(
        response,
        partPath,
        offset,
        (chunkBytes) => {
          job.bytesReceived += chunkBytes;
          const elapsed = (Date.now() - startedAt) / 1000;
          job.bytesPerSecond = elapsed > 0 ? job.bytesReceived / elapsed : 0;
          const remaining = job.bytesTotal - job.bytesReceived;
          job.etaSeconds = job.bytesPerSecond > 0 ? Math.round(remaining / job.bytesPerSecond) : null;
          this.notify(job);
        },
        declared,
      );

      await fs.rename(partPath, target);
    }

    job.status = 'verifying';
    job.currentFile = null;
    this.notify(job);
    await verify(spec);

    await writeManifest(spec.id, buildManifest(spec));
    this.finish(job, 'done', null);
  }

  private finish(job: InternalJob, status: DownloadStatus, error: string | null): void {
    job.status = status;
    job.error = error;
    job.currentFile = null;
    job.finishedAt = new Date().toISOString();
    if (status !== 'done') job.etaSeconds = null;
    this.notify(job);
  }

  private notify(job: InternalJob): void {
    const snapshot = publicJob(job);
    for (const listener of job.listeners) listener(snapshot);
  }
}

async function verify(spec: ModelSpec): Promise<void> {
  for (const file of spec.files) {
    const full = path.join(modelDir(spec.id), file.path);
    const size = await fileSize(full);
    if (size < 0) throw new Error(`${file.path} missing after download`);
    const drift = Math.abs(size - file.bytes);
    if (drift > tolerance(file.bytes)) {
      throw new Error(`${file.path} is ${size} bytes, expected ${file.bytes}`);
    }
    if (file.path.endsWith('.json')) {
      JSON.parse(await fs.readFile(full, 'utf8'));
    }
  }
}

/** How far a file may deviate from its published size before we call it corrupt. */
function tolerance(expected: number): number {
  return Math.max(4096, expected * 0.01);
}

async function fileSize(target: string): Promise<number> {
  try {
    return (await fs.stat(target)).size;
  } catch {
    return -1;
  }
}

async function writeStream(
  response: Response,
  target: string,
  offset: number,
  onChunk: (bytes: number) => void,
  declared: number,
): Promise<void> {
  const handle = await fs.open(target, offset > 0 ? 'r+' : 'w');
  try {
    if (offset > 0) await handle.truncate(offset);
    let position = offset;
    const reader = response.body?.getReader();
    if (!reader) throw new Error('response had no body');
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      await handle.write(value, 0, value.byteLength, position);
      position += value.byteLength;
      onChunk(value.byteLength);
    }
    await handle.truncate(position);
    if (declared > 0 && position - offset !== declared) {
      throw new Error(`connection closed early: got ${position - offset} of ${declared} bytes`);
    }
  } finally {
    await handle.close();
  }
}

function publicJob(job: InternalJob): DownloadJob {
  return {
    id: job.id,
    modelId: job.modelId,
    status: job.status,
    bytesReceived: job.bytesReceived,
    bytesTotal: job.bytesTotal,
    bytesPerSecond: job.bytesPerSecond,
    etaSeconds: job.etaSeconds,
    currentFile: job.currentFile,
    error: job.error,
    createdAt: job.createdAt,
    finishedAt: job.finishedAt,
  };
}

export const downloads = new DownloadManager();

export function cancelJobForModel(modelId: string): void {
  const job = downloads.list().find((entry) => entry.modelId === modelId);
  if (job) downloads.cancel(job.id);
}

export function assertKnownModel(modelId: string): ModelSpec {
  const spec = findModel(modelId);
  if (!spec) throw new Error(`unknown model: ${modelId}`);
  return spec;
}