'use client';

import type { AnalysisEvent, AppState } from './types';
import type { DownloadJob } from './downloads';

export async function fetchState(): Promise<AppState> {
  const response = await fetch('/api/state', { cache: 'no-store' });
  if (!response.ok) throw new Error(`state request failed: ${response.status}`);
  return (await response.json()) as AppState;
}

export async function saveSettings(patch: Partial<AppState['settings']>): Promise<AppState['settings']> {
  const response = await fetch('/api/settings', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(patch),
  });
  if (!response.ok) throw new Error(`settings update failed: ${response.status}`);
  const data = (await response.json()) as { settings: AppState['settings'] };
  return data.settings;
}

export async function startDownload(modelId: string): Promise<DownloadJob> {
  const response = await fetch(`/api/models/${encodeURIComponent(modelId)}/download`, {
    method: 'POST',
  });
  const data = (await response.json()) as { job?: DownloadJob; error?: string };
  if (!response.ok || !data.job) throw new Error(data.error ?? `download failed: ${response.status}`);
  return data.job;
}

export async function fetchDownloads(): Promise<DownloadJob[]> {
  const response = await fetch('/api/models/downloads', { cache: 'no-store' });
  if (!response.ok) throw new Error(`job list failed: ${response.status}`);
  const data = (await response.json()) as { jobs: DownloadJob[] };
  return data.jobs;
}

export async function cancelDownload(jobId: string): Promise<void> {
  await fetch(`/api/models/downloads/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
}

export async function removeModel(modelId: string): Promise<void> {
  await fetch(`/api/models/${encodeURIComponent(modelId)}`, { method: 'DELETE' });
}

export async function unloadModel(modelId: string): Promise<boolean> {
  const response = await fetch(`/api/models/${encodeURIComponent(modelId)}/unload`, {
    method: 'POST',
  });
  const data = (await response.json().catch(() => ({}))) as { unloaded?: boolean; error?: string };
  if (!response.ok) throw new Error(data.error ?? `unload failed: ${response.status}`);
  return data.unloaded ?? false;
}

export interface PdfText {
  text: string;
  pages: number;
  truncated: boolean;
}

/** Reads the text layer out of a PDF on the server, so nothing lands in the browser bundle. */
export async function extractPdfText(file: File): Promise<PdfText> {
  const response = await fetch('/api/pdf', {
    method: 'POST',
    headers: { 'content-type': 'application/pdf' },
    body: file,
  });
  const data = (await response.json().catch(() => ({}))) as Partial<PdfText> & { error?: string };
  if (!response.ok || typeof data.text !== 'string') {
    throw new Error(data.error ?? `pdf upload failed: ${response.status}`);
  }
  return { text: data.text, pages: data.pages ?? 0, truncated: data.truncated ?? false };
}

/** Streams download progress as server-sent events. */
export async function streamDownload(
  jobId: string,
  onJob: (job: DownloadJob) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(`/api/models/downloads/${encodeURIComponent(jobId)}`, {
    cache: 'no-store',
    signal,
  });
  if (!response.ok) throw new Error(`progress stream failed: ${response.status}`);
  await readEventStream(response, (data) => onJob(data as DownloadJob));
}

export async function streamAnalysis(
  text: string,
  onEvent: (event: AnalysisEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch('/api/analyze', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ text }),
    signal,
  });
  if (!response.ok) {
    const data = (await response.json().catch(() => ({}))) as { error?: string };
    throw new Error(data.error ?? `analysis failed: ${response.status}`);
  }
  await readNdjson(response, (data) => onEvent(data as AnalysisEvent));
}

async function readEventStream(response: Response, onData: (value: unknown) => void): Promise<void> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error('stream had no body');
  const decoder = new TextDecoder();
  let buffer = '';

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let index = buffer.indexOf('\n\n');
    while (index !== -1) {
      const chunk = buffer.slice(0, index);
      buffer = buffer.slice(index + 2);
      for (const line of chunk.split('\n')) {
        if (!line.startsWith('data:')) continue;
        onData(JSON.parse(line.slice(5).trim()));
      }
      index = buffer.indexOf('\n\n');
    }
  }
}

async function readNdjson(response: Response, onData: (value: unknown) => void): Promise<void> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error('stream had no body');
  const decoder = new TextDecoder();
  let buffer = '';

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let index = buffer.indexOf('\n');
    while (index !== -1) {
      const line = buffer.slice(0, index).trim();
      buffer = buffer.slice(index + 1);
      if (line) onData(JSON.parse(line));
      index = buffer.indexOf('\n');
    }
  }
  const rest = buffer.trim();
  if (rest) onData(JSON.parse(rest));
}