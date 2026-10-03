'use client';

import type { ButtonHTMLAttributes, ReactNode } from 'react';

type Variant = 'primary' | 'plain' | 'quiet' | 'danger';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  children: ReactNode;
}

const VARIANTS: Record<Variant, string> = {
  primary: 'bg-ink text-paper border-ink hover:bg-ink-soft',
  plain: 'bg-paper-raised text-ink border-rule-strong hover:border-ink',
  quiet: 'bg-transparent text-ink-soft border-transparent hover:text-ink hover:border-rule',
  danger: 'bg-transparent text-danger border-danger/40 hover:border-danger',
};

export function Button({ variant = 'plain', className = '', children, ...rest }: ButtonProps) {
  return (
    <button
      type="button"
      className={`focusable inline-flex cursor-pointer items-center gap-2 rounded-md border px-3.5 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-45 ${VARIANTS[variant]} ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}

export function Panel({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <div className={`tile rounded-lg ${className}`}>{children}</div>;
}

export function Meter({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div
      aria-hidden
      className="h-2 w-full overflow-hidden rounded-full border border-rule bg-paper"
    >
      <div
        className="h-full bg-machine transition-[width] duration-200"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: ReactNode;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="block text-sm font-medium text-ink">{label}</span>
      {hint ? <span className="mt-0.5 block text-sm text-ink-soft">{hint}</span> : null}
      <div className="mt-2">{children}</div>
    </label>
  );
}

export function NumberInput({
  value,
  onCommit,
  min,
  max,
  step = 1,
  suffix,
  disabled,
}: {
  value: number;
  onCommit: (next: number) => void;
  min: number;
  max: number;
  step?: number;
  suffix?: string;
  disabled?: boolean;
}) {
  return (
    <span className="inline-flex items-center gap-2">
      <input
        type="number"
        className="focusable w-28 rounded-md border border-rule-strong bg-paper-raised px-3 py-2 font-mono text-sm text-ink"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(event) => {
          const next = Number(event.target.value);
          if (Number.isFinite(next)) onCommit(next);
        }}
      />
      {suffix ? <span className="text-sm text-ink-soft">{suffix}</span> : null}
    </span>
  );
}

export function Slider({
  value,
  onCommit,
  min,
  max,
  step,
  format,
  disabled,
}: {
  value: number;
  onCommit: (next: number) => void;
  min: number;
  max: number;
  step: number;
  format: (value: number) => string;
  disabled?: boolean;
}) {
  return (
    <span className="flex items-center gap-3">
      <input
        type="range"
        className="focusable h-2 w-full max-w-xs cursor-pointer appearance-none rounded-full border border-rule bg-paper accent-(--color-ink)"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(event) => onCommit(Number(event.target.value))}
      />
      <span className="w-28 shrink-0 text-right font-mono text-sm text-ink">{format(value)}</span>
    </span>
  );
}