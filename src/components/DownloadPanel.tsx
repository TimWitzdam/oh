'use client';

import type { DownloadJob } from '@/lib/downloads';
import type { ModelInfo } from '@/lib/types';
import { CheckIcon, DownloadIcon } from './icons';
import { Button, Meter } from './primitives';
import { isRunning } from './useDownloads';

const RATE_WIDTH = 8;
const ETA_WIDTH = 10;

function formatRate(bytesPerSecond: number): string {
  if (!bytesPerSecond || bytesPerSecond < 1) return '—';
  const mb = bytesPerSecond / 1_000_000;
  return mb >= 100 ? `${Math.round(mb)} MB/s` : mb >= 1 ? `${mb.toFixed(1)} MB/s` : `${(bytesPerSecond / 1000).toFixed(0)} kB/s`;
}

function formatEta(seconds: number | null): string {
  if (seconds === null || seconds < 0) return '—';
  if (seconds < 60) return `${seconds}s left`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min left`;
  return `${(seconds / 3600).toFixed(1)} h left`;
}

function describe(job: DownloadJob): string | null {
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
    default:
      return null;
  }
}

function Cell({ width, children }: { width: number; children: string }) {
  return (
    <span className="inline-block shrink-0 font-mono tabular-nums" style={{ width: `${width}ch` }}>
      {children}
    </span>
  );
}

function Dot() {
  return (
    <span aria-hidden className="shrink-0 text-rule-strong">
      ·
    </span>
  );
}

/**
 * Progress arrives several times a second, so each number gets its own
 * fixed-width monospace cell: the byte count can never change the width of
 * anything around it, and the rate and ETA hold their slot even before the
 * first sample lands. Three lines with reserved heights mean no reflow, so the
 * progress bar and the buttons below stay put for the whole download.
 */
function Progress({ job }: { job: DownloadJob }) {
  const total = (job.bytesTotal / 1e6).toFixed(1);
  const received = (job.bytesReceived / 1e6).toFixed(1).padStart(total.length);
  const file = job.currentFile?.split('/').pop() ?? job.currentFile ?? '';

  return (
    <div className="mt-3">
      <p className="flex items-baseline gap-x-1.5 whitespace-nowrap text-sm text-ink-soft">
        <Cell width={total.length}>{received}</Cell>
        <span>of</span>
        <Cell width={total.length}>{total}</Cell>
        <span>MB</span>
      </p>
      <p className="mt-1 flex items-baseline gap-x-1.5 whitespace-nowrap text-sm text-ink-soft">
        <Cell width={RATE_WIDTH}>{formatRate(job.bytesPerSecond)}</Cell>
        <Dot />
        <Cell width={ETA_WIDTH}>{formatEta(job.etaSeconds)}</Cell>
      </p>
      <p className="mt-1 h-4 truncate text-xs text-ink-faint">{file}</p>
    </div>
  );
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
  const status = job ? describe(job) : null;

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

      {job && status !== null ? (
        <p className="mt-3 text-sm text-ink-soft">{status}</p>
      ) : job && running ? (
        <Progress job={job} />
      ) : (
        <p className="mt-3 text-sm text-ink-soft">{`${model.bytesLabel} download`}</p>
      )}

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