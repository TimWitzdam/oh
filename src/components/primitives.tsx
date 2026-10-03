'use client';

import { createContext, useContext, useId } from 'react';
import type { ButtonHTMLAttributes, ReactNode } from 'react';

import { InfoIcon } from './icons';

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

/**
 * A file picker wearing button clothes. The input lives inside the label so the
 * pair stays one tab stop and keeps the styling of a quiet button.
 */
export function FileButton({
  accept,
  onFile,
  disabled,
  children,
}: {
  accept: string;
  onFile: (file: File) => void;
  disabled?: boolean;
  children: ReactNode;
}) {
  return (
    <label
      className={`inline-flex cursor-pointer items-center gap-2 rounded-md border border-transparent px-3.5 py-2 text-sm font-medium text-ink-soft transition-colors hover:border-rule hover:text-ink has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-(--color-ink) has-[:disabled]:cursor-not-allowed has-[:disabled]:opacity-45 ${disabled ? 'pointer-events-none' : ''}`}
    >
      {children}
      <input
        type="file"
        accept={accept}
        disabled={disabled}
        className="sr-only"
        onChange={(event) => {
          const file = event.target.files?.[0];
          // Reset first, so picking the same file twice still fires a change.
          event.target.value = '';
          if (file) onFile(file);
        }}
      />
    </label>
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

/**
 * An explanation that waits to be asked for. The trigger is a real button, so
 * the note opens on hover, on tap and on keyboard focus rather than only on a
 * pointer; the bubble is a tooltip, so a control names it with
 * aria-describedby and screen readers get the words without the popup.
 */
export function Hint({ id, children }: { id: string; children: ReactNode }) {
  return (
    <span className="group relative ml-1 inline-flex align-middle">
      <button
        type="button"
        aria-describedby={id}
        className="focusable cursor-help rounded-full text-ink-faint transition-colors hover:text-ink"
      >
        <InfoIcon />
        <span className="sr-only">What this means</span>
      </button>
      <span
        id={id}
        role="tooltip"
        className="pointer-events-none absolute left-0 top-full z-50 mt-1.5 w-60 rounded-md border border-rule bg-paper-raised px-3 py-2 text-sm font-normal leading-snug text-ink-soft opacity-0 shadow-sm transition-opacity duration-150 group-hover:opacity-100 group-focus-within:opacity-100"
      >
        {children}
      </span>
    </span>
  );
}

/**
 * Ids the field hands down, so the control can take its name from the label
 * alone. A wrapping label would drag the whole explanation into the name.
 */
const FieldControl = createContext<{ labelId: string; hintId?: string; inputId: string } | null>(
  null,
);

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: ReactNode;
  children: ReactNode;
}) {
  const base = useId();
  const labelId = `${base}-label`;
  const hintId = `${base}-hint`;
  const inputId = `${base}-input`;
  return (
    <div>
      {/* The label sits on the caption rather than wrapping the row, so clicking
          it focuses the input without also firing the buttons underneath. */}
      <div className="flex items-center gap-1.5">
        <label htmlFor={inputId} id={labelId} className="cursor-pointer text-sm font-medium text-ink">
          {label}
        </label>
        {hint ? <Hint id={hintId}>{hint}</Hint> : null}
      </div>
      <FieldControl.Provider value={{ labelId, hintId: hint ? hintId : undefined, inputId }}>
        <div className="mt-2">{children}</div>
      </FieldControl.Provider>
    </div>
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
  const field = useContext(FieldControl);
  return (
    <span className="inline-flex items-center gap-2">
      <input
        type="number"
        id={field?.inputId}
        aria-labelledby={field?.labelId}
        aria-describedby={field?.hintId}
        className="focusable w-28 rounded-md border border-rule-strong bg-paper-raised px-3 py-2 text-sm tabular-nums text-ink"
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
  const field = useContext(FieldControl);
  return (
    <span className="flex items-center gap-3">
      <input
        type="range"
        id={field?.inputId}
        aria-labelledby={field?.labelId}
        aria-describedby={field?.hintId}
        className="focusable h-2 w-full max-w-xs cursor-pointer appearance-none rounded-full border border-rule bg-paper accent-(--color-ink)"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(event) => onCommit(Number(event.target.value))}
      />
      <span className="w-28 shrink-0 text-right text-sm tabular-nums text-ink">{format(value)}</span>
    </span>
  );
}