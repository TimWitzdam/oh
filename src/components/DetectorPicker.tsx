'use client';

import type { DownloadJob } from '@/lib/downloads';
import type { ModelInfo } from '@/lib/types';
import { DownloadPanel } from './DownloadPanel';
import { CheckIcon } from './icons';

const TIER_NOTE: Record<ModelInfo['tier'], string> = {
  lite: 'Smallest and fastest, weakest on unseen writing',
  balanced: 'A different kind of checker, not a sharper one',
  deep: 'Most accurate, slowest',
};

const SPEED: Record<ModelInfo['tier'], string> = {
  lite: 'Instant',
  balanced: 'Fast',
  deep: 'Slow',
};

/** Bar length relative to the slowest tier, from measured cost per passage. */
const SPEED_FRACTION: Record<ModelInfo['tier'], number> = {
  lite: 0.06,
  balanced: 0.8,
  deep: 1,
};

export function DetectorPicker({
  models,
  jobs,
  activeModelId,
  busy,
  onStart,
  onCancel,
  onRemove,
  onActivate,
}: {
  models: ModelInfo[];
  jobs: Record<string, DownloadJob>;
  activeModelId: string | null;
  busy: boolean;
  onStart: (modelId: string) => void;
  onCancel: (jobId: string) => void;
  onRemove: (modelId: string) => void;
  onActivate: (modelId: string) => void;
}) {
  const maxBytes = Math.max(...models.map((model) => model.bytes));
  const maxRam = Math.max(...models.map((model) => model.ramMb));

  return (
    <div
      role="radiogroup"
      aria-label="Detector"
      className="grid gap-4 md:grid-cols-3"
    >
      {models.map((model) => (
        <ModelCard
          key={model.id}
          model={model}
          job={latestJob(jobs, model.id) ?? partialJob(model)}
          selected={activeModelId === model.id}
          busy={busy}
          scale={{ maxBytes, maxRam }}
          onSelect={() => onActivate(model.id)}
          onStart={() => onStart(model.id)}
          onCancel={(jobId) => onCancel(jobId)}
          onRemove={() => onRemove(model.id)}
        />
      ))}
    </div>
  );
}

function ModelCard({
  model,
  job,
  selected,
  busy,
  scale,
  onSelect,
  onStart,
  onCancel,
  onRemove,
}: {
  model: ModelInfo;
  job?: DownloadJob;
  selected: boolean;
  busy: boolean;
  scale: { maxBytes: number; maxRam: number };
  onSelect: () => void;
  onStart: () => void;
  onCancel: (jobId: string) => void;
  onRemove: () => void;
}) {
  const usable = model.installed;
  const cardState = selected ? 'selected' : usable ? 'available' : 'missing';

  return (
    <section
      role="radio"
      aria-checked={selected}
      aria-disabled={!usable || busy}
      tabIndex={usable && !busy ? 0 : -1}
      onClick={usable && !busy ? onSelect : undefined}
      onKeyDown={(event) => {
        if (!usable || busy) return;
        if (event.key === ' ' || event.key === 'Enter') {
          event.preventDefault();
          onSelect();
        }
      }}
      className={`tile tile-hover @container flex cursor-pointer flex-col rounded-lg p-5 ${
        selected ? 'border-ink bg-paper' : ''
      }`}
      data-state={cardState}
    >
      <div className="flex items-center gap-3">
        <span
          aria-hidden
          className={`grid h-5 w-5 shrink-0 place-items-center rounded-full border ${
            selected ? 'border-ink bg-ink text-paper' : 'border-rule-strong bg-paper-raised'
          }`}
        >
          {selected ? <CheckIcon className="h-3 w-3" /> : null}
        </span>
        <h3 className="text-2xl font-semibold tracking-tight text-ink">{model.name}</h3>
      </div>

      <p className="mt-2 text-base text-ink-soft">{TIER_NOTE[model.tier]}</p>

      <p className="mt-4 text-base text-ink-soft">{model.detail}</p>

      <dl className="mt-5 space-y-2.5">
        <Meter
          label="Download"
          value={model.bytesLabel}
          fraction={model.bytes / scale.maxBytes}
        />
        <Meter
          label="Memory"
          value={`${model.ramMb >= 1000 ? `${(model.ramMb / 1000).toFixed(1)} GB` : `${model.ramMb} MB`}`}
          fraction={model.ramMb / scale.maxRam}
        />
        <Meter label="Speed" value={SPEED[model.tier]} fraction={SPEED_FRACTION[model.tier]} />
      </dl>

      <div
        className="mt-auto pt-5"
        onClick={(event) => event.stopPropagation()}
        onKeyDown={(event) => event.stopPropagation()}
      >
        <DownloadPanel
          model={model}
          job={job}
          selected={selected}
          busy={busy}
          onStart={onStart}
          onCancel={() => job && onCancel(job.id)}
          onRemove={onRemove}
          onActivate={onSelect}
        />
      </div>
    </section>
  );
}

/**
 * Label, bar, reading. The bar is the only part that gives way: it fills
 * whatever the label and the reading leave over, and once the card is too
 * narrow to hold the three of them in one row the reading moves up beside the
 * label and the bar takes a full line of its own, rather than being squeezed
 * flat to nothing. The switch is a container query on the card and not a
 * breakpoint on the window, because a card's width comes from the grid around
 * it; 19rem is where the middle of the row stops being a bar worth reading.
 */
function Meter({
  label,
  value,
  fraction,
}: {
  label: string;
  value: string;
  fraction: number;
}) {
  const width = Math.max(6, Math.min(100, fraction * 100));
  return (
    <div className="grid grid-cols-[auto_1fr] items-baseline gap-x-3 gap-y-1 @min-[19rem]:grid-cols-[5rem_1fr_5rem] @min-[19rem]:items-center @min-[19rem]:gap-y-0">
      <dt className="text-sm text-ink-soft">{label}</dt>
      <dd
        aria-hidden
        className="col-span-2 row-start-2 h-1.5 overflow-hidden rounded-full bg-rule @min-[19rem]:col-span-1 @min-[19rem]:col-start-2 @min-[19rem]:row-start-1"
      >
        <span
          className={`block h-full rounded-full ${label === 'Speed' ? 'bg-human' : 'bg-machine'}`}
          style={{ width: `${width}%` }}
        />
      </dd>
      <dd className="col-start-2 row-start-1 text-right text-sm whitespace-nowrap text-ink @min-[19rem]:col-start-3">
        {value}
      </dd>
    </div>
  );
}

function latestJob(
  jobs: Record<string, DownloadJob>,
  modelId: string,
): DownloadJob | undefined {
  return Object.values(jobs)
    .filter((job) => job.modelId === modelId)
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt))[0];
}

/**
 * Bytes left on disk by a download that was interrupted (a restart, a crash)
 * show up here so the card can offer a resume instead of a fresh download.
 */
function partialJob(model: ModelInfo): DownloadJob | undefined {
  if (model.partialBytes <= 0) return undefined;
  return {
    id: `partial-${model.id}`,
    modelId: model.id,
    status: 'canceled',
    bytesReceived: model.partialBytes,
    bytesTotal: model.bytes,
    bytesPerSecond: 0,
    etaSeconds: null,
    currentFile: null,
    error: null,
    createdAt: new Date(0).toISOString(),
    finishedAt: null,
  };
}