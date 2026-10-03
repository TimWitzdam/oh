'use client';

import { MAX_TEXT_CHARS, MIN_WORDS, countWords, tidyText } from './text';
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
  const data = (await response.json().catch(() => ({}))) as {
    settings?: AppState['settings'];
    error?: string;
  };
  if (!response.ok || !data.settings) throw new Error(data.error ?? `settings update failed: ${response.status}`);
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

/** Plain text has nothing to parse, so it never has to leave the browser. */
const MAX_TEXT_FILE_BYTES = 8 * 1024 * 1024;

/**
 * Reads a .txt or .md file as text, tidied and capped the same way the server
 * would have done it. Two files that are not text are caught here rather than
 * scored: one that decodes into NUL bytes is a .docx wearing the wrong name,
 * and one full of replacement characters was saved in a single-byte encoding
 * this browser cannot read.
 */
export async function readTextFile(file: File): Promise<{ text: string; truncated: boolean }> {
  if (file.size > MAX_TEXT_FILE_BYTES) {
    throw new Error(
      `That file is larger than the ${MAX_TEXT_FILE_BYTES / 1024 / 1024} MB limit for text.`,
    );
  }

  const { text: cleaned } = tidyText(await file.text());
  if (cleaned.includes('\u0000') || countWords(cleaned) === 0) {
    throw new Error(
      'That file did not read as text. Export it as .txt or .md, or drop the PDF itself.',
    );
  }
  const mangled = (cleaned.match(/\ufffd/g)?.length ?? 0) / cleaned.length;
  if (mangled > 0.01) {
    throw new Error(
      'That file is not in a readable text encoding. Re-save it as UTF-8 plain text, or drop the PDF.',
    );
  }

  const truncated = cleaned.length > MAX_TEXT_CHARS;
  const text = truncated ? cleaned.slice(0, MAX_TEXT_CHARS) : cleaned;
  if (countWords(text) < MIN_WORDS) {
    throw new Error(
      `That file held fewer than ${MIN_WORDS} words, and a score needs a paragraph to say anything.`,
    );
  }
  return { text, truncated };
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