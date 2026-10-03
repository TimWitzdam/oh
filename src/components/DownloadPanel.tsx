'use client';

import type { DownloadJob } from '@/lib/downloads';
import type { ModelInfo } from '@/lib/types';
import { CheckIcon, DownloadIcon } from './icons';
import { Button, Meter } from './primitives';
import { isRunning } from './useDownloads';

function formatRate(bytesPerSecond: number): string {
  if (!bytesPerSecond || bytesPerSecond < 1) return '';
  const mb = bytesPerSecond / 1_000_000;
  return mb >= 1 ? `${mb.toFixed(1)} MB/s` : `${(bytesPerSecond / 1000).toFixed(0)} kB/s`;
}

function formatEta(seconds: number | null): string {
  if (seconds === null || seconds < 0) return '';
  if (seconds < 60) return `${seconds}s left`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min left`;
  return `${(seconds / 3600).toFixed(1)} h left`;
}

function describe(job: DownloadJob): string {
  switch (job.status) {
    case 'queued':
      return 'Waiting to start…';
    case 'verifying':
      return 'Checking files…';
    case 'done':
      return 'Installed';
    case 'canceled':
      return 'Paused — resuming keeps what was already fetched';
    case 'error':
      return job.error ?? 'Download failed';
    default: {
      const parts = [`${(job.bytesReceived / 1e6).toFixed(1)} of ${(job.bytesTotal / 1e6).toFixed(1)} MB`];
      const rate = formatRate(job.bytesPerSecond);
      const eta = formatEta(job.etaSeconds);
      if (rate) parts.push(rate);
      if (eta) parts.push(eta);
      if (job.currentFile) parts.push(job.currentFile.split('/').pop() ?? job.currentFile);
      return parts.join(' · ');
    }
  }
}

export function DownloadPanel({
  model,
  job,
  onStart,
  onCancel,
  onRemove,
  selected,
  onActivate,
  busy,
}: {
  model: ModelInfo;
  job?: DownloadJob;
  onStart: () => void;
  onCancel: () => void;
  onRemove: () => void;
  selected: boolean;
  onActivate: () => void;
  busy: boolean;
}) {
  const running = isRunning(job);
  const progress = job ? Math.min(1, job.bytesTotal > 0 ? job.bytesReceived / job.bytesTotal : 0) : 0;

  if (model.installed) {
    return (
      <div className="flex flex-wrap items-center gap-3">
        {selected ? (
          <span className="inline-flex items-center gap-2 text-base text-ink">
            <CheckIcon className="text-human" />
            Selected
          </span>
        ) : (
          <Button variant="primary" onClick={onActivate} disabled={busy}>
            <CheckIcon />
            Use {model.name}
          </Button>
        )}
        <Button variant="quiet" onClick={onRemove} disabled={busy}>
          Remove files
        </Button>
      </div>
    );
  }

  return (
    <div>
      {job && job.status !== 'queued' ? <Meter value={running ? progress : job.status === 'error' ? 0 : 1} /> : null}

      <p className="mt-3 text-sm text-ink-soft">{job ? describe(job) : `${model.bytesLabel} download`}</p>

      <div className="mt-3 flex flex-wrap items-center gap-3">
        {running ? (
          <Button variant="plain" onClick={onCancel}>
            Pause
          </Button>
        ) : (
          <Button variant="primary" onClick={onStart} disabled={busy}>
            <DownloadIcon />
            {job?.status === 'error'
              ? 'Retry download'
              : job?.status === 'canceled'
                ? 'Resume download'
                : 'Download'}
          </Button>
        )}
        {!running && job && job.status !== 'done' ? (
          <Button variant="quiet" onClick={onRemove} disabled={busy}>
            Clear partial files
          </Button>
        ) : null}
      </div>
    </div>
  );
}