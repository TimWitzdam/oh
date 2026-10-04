'use client';

import { useCallback, useEffect, useId, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import type {
  ClipboardEvent as ReactClipboardEvent,
  DragEvent as ReactDragEvent,
  KeyboardEvent as ReactKeyboardEvent,
} from 'react';

import { extractPdfText, readTextFile } from '@/lib/client';
import {
  MAX_TEXT_CHARS,
  MIN_WORDS,
  countWords,
  estimateSeconds,
  formatDuration,
  tidyText,
} from '@/lib/text';
import type { ModelInfo } from '@/lib/types';
import { Button, FileButton } from './primitives';
import type { Draft } from './useDraft';

/** What a drop or the file picker will take: a PDF, or text in some spelling. */
const TEXT_FILE = /\.(txt|md|markdown|text)$/i;

/** Nothing to listen to: the value below is a property of the browser, not a change. */
const noSubscription = () => () => {};

/** Command versus Control, read after hydration rather than guessed at on the server. */
function useIsMac(): boolean {
  return useSyncExternalStore(
    noSubscription,
    () => /mac|iphone|ipad|ipod/i.test(navigator.userAgent),
    () => false,
  );
}

export function Composer({
  model,
  models,
  draft,
  busy,
  onRun,
  onCancel,
  onSelectModel,
}: {
  model: ModelInfo;
  models: ModelInfo[];
  draft: Draft;
  busy: boolean;
  onRun: (text: string) => void;
  onCancel: () => void;
  onSelectModel: (modelId: string) => void;
}) {
  const { text, setText, atLoad } = draft;
  const [importing, setImporting] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [confirmingClear, setConfirmingClear] = useState(false);
  const area = useRef<HTMLTextAreaElement>(null);
  /** Drag events fire per element, so the overlay needs a depth, not a boolean. */
  const dragDepth = useRef(0);
  const countsId = useId();

  const isMac = useIsMac();
  const words = useMemo(() => countWords(text), [text]);
  const empty = words === 0;
  const short = words < MIN_WORDS;
  const canRun = !short && !busy;
  /** Untouched since the page opened, so it is the text from the last visit. */
  const restored = atLoad !== null && text === atLoad;

  /** Anything the user does to the text retires the messages about the old text. */
  const edit = useCallback(
    (value: string) => {
      setNote(null);
      setProblem(null);
      setConfirmingClear(false);
      setText(value);
    },
    [setText],
  );

  /**
   * Puts new text in the box and leaves the caret at the end of it. The caret
   * waits for the render that carries the value, hence the frame.
   */
  const apply = useCallback(
    (body: string, message: string | null) => {
      edit(body);
      if (message) setNote(message);
      requestAnimationFrame(() => {
        const field = area.current;
        if (!field) return;
        field.focus();
        field.setSelectionRange(body.length, body.length);
      });
    },
    [edit],
  );

  const handleFile = useCallback(
    async (file: File) => {
      setNote(null);
      setProblem(null);

      const pdf = file.type === 'application/pdf' || /\.pdf$/i.test(file.name);
      if (!pdf && !TEXT_FILE.test(file.name) && !file.type.startsWith('text/')) {
        setProblem(
          `${file.name} is not a format this can read. Drop a PDF, or a plain .txt or .md file.`,
        );
        return;
      }

      setImporting(true);
      try {
        if (pdf) {
          const extracted = await extractPdfText(file);
          const count = countWords(extracted.text);
          apply(
            extracted.text,
            `${file.name} — ${extracted.pages} ${extracted.pages === 1 ? 'page' : 'pages'}, ${count.toLocaleString()} words${
              extracted.truncated ? `, cut at the ${MAX_TEXT_CHARS.toLocaleString()} character limit` : ''
            }.`,
          );
        } else {
          const imported = await readTextFile(file);
          const count = countWords(imported.text);
          apply(
            imported.text,
            `${file.name} — ${count.toLocaleString()} words${
              imported.truncated ? `, cut at the ${MAX_TEXT_CHARS.toLocaleString()} character limit` : ''
            }.`,
          );
        }
      } catch (caught) {
        setProblem(caught instanceof Error ? caught.message : String(caught));
      } finally {
        setImporting(false);
      }
    },
    [apply],
  );

  const onDragEnter = (event: ReactDragEvent<HTMLDivElement>) => {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    dragDepth.current += 1;
    setDragging(true);
  };

  const onDragOver = (event: ReactDragEvent<HTMLDivElement>) => {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
  };

  const onDragLeave = (event: ReactDragEvent<HTMLDivElement>) => {
    if (!carriesFiles(event)) return;
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setDragging(false);
  };

  const onDrop = (event: ReactDragEvent<HTMLDivElement>) => {
    if (!carriesFiles(event)) return;
    event.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) void handleFile(file);
  };

  // A file let go anywhere else on the page navigates the tab to it, which
  // would cost the reader the only copy of their text.
  useEffect(() => {
    const swallow = (event: DragEvent) => {
      if (carriesFiles(event)) event.preventDefault();
    };
    window.addEventListener('dragover', swallow);
    window.addEventListener('drop', swallow);
    return () => {
      window.removeEventListener('dragover', swallow);
      window.removeEventListener('drop', swallow);
    };
  }, []);

  const onKeyDown = (event: ReactKeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter' || !(event.metaKey || event.ctrlKey)) return;
    event.preventDefault();
    if (canRun) onRun(text);
  };

  const onPaste = (event: ReactClipboardEvent<HTMLTextAreaElement>) => {
    const pasted = event.clipboardData.getData('text/plain');
    if (!pasted) return;
    const { text: cleaned, fixes } = tidyText(pasted);
    // Nothing to repair: let the browser paste it and keep its own undo stack.
    if (cleaned === pasted) return;

    event.preventDefault();
    const field = event.currentTarget;
    const caret = field.selectionStart + cleaned.length;
    edit(text.slice(0, field.selectionStart) + cleaned + text.slice(field.selectionEnd));
    if (fixes > 0) {
      setNote(
        `Tidied ${fixes} stray ${fixes === 1 ? 'character' : 'characters'} and line endings out of the paste.`,
      );
    }
    // The value is replaced on the next render, so the caret waits for it.
    requestAnimationFrame(() => field.setSelectionRange(caret, caret));
  };

  return (
    <div>
      <div
        onDragEnter={onDragEnter}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
      >
        <div className="relative overflow-hidden rounded-lg border border-rule bg-paper-raised focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-(--color-ink)">
          <textarea
            ref={area}
            value={text}
            onChange={(event) => edit(event.target.value)}
            onKeyDown={onKeyDown}
            onPaste={onPaste}
            spellCheck={false}
            aria-label="Text to analyse"
            aria-describedby={countsId}
            placeholder="Paste the text you want to check."
            className="block min-h-[42vh] w-full resize-y border-0 bg-transparent p-5 text-base leading-relaxed text-ink placeholder:text-ink-faint focus:outline-none"
          />

{dragging ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 bg-paper-raised/95 text-center">
                <p className="text-lg font-medium text-ink">Drop it to replace the text</p>
                <p className="text-sm text-ink-soft">PDF, .txt or .md</p>
              </div>
            ) : null}

          <div
            id={countsId}
            className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-rule px-5 py-2.5 text-sm text-ink-soft"
          >
            <span className="tabular-nums">
              {words.toLocaleString()} {words === 1 ? 'word' : 'words'}
            </span>
            {empty ? null : (
              <span className="tabular-nums text-ink-faint">
                {text.length.toLocaleString()} characters
              </span>
            )}
            <span className="flex-1" />
            {restored ? <span className="text-ink-faint">restored from your last visit</span> : null}
            {short && !empty ? (
              <span>
                {MIN_WORDS - words} more {MIN_WORDS - words === 1 ? 'word' : 'words'} to reach the
                minimum
              </span>
            ) : null}
            {!short ? (
              <span>
                {formatDuration(estimateSeconds(words, model.wordsPerSecond))} on {model.name}
              </span>
            ) : null}
          </div>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-3">
        {busy ? (
          <Button variant="plain" onClick={onCancel}>
            Cancel analysis
          </Button>
        ) : (
          <Button variant="primary" onClick={() => canRun && onRun(text)} disabled={!canRun}>
            Analyse text
            <span aria-hidden className="text-xs opacity-70">
              {isMac ? '⌘' : 'Ctrl+'}⏎
            </span>
          </Button>
        )}

        {confirmingClear ? (
          <span className="flex items-center gap-2">
            <Button
              variant="danger"
              onClick={() => {
                setText('');
                setNote(null);
                setProblem(null);
                setConfirmingClear(false);
                area.current?.focus();
              }}
            >
              Clear {words.toLocaleString()} words
            </Button>
            <Button variant="quiet" onClick={() => setConfirmingClear(false)}>
              Keep
            </Button>
          </span>
        ) : (
          <Button variant="quiet" onClick={() => setConfirmingClear(true)} disabled={empty}>
            Clear
          </Button>
        )}

        <FileButton
          accept="application/pdf,.pdf,.txt,.md,.markdown,.text,text/plain"
          onFile={(file) => void handleFile(file)}
          disabled={importing}
        >
          {importing ? 'Reading…' : 'Add PDF or text'}
        </FileButton>

        <span className="flex-1" />

        <ModelSwitcher models={models} activeId={model.id} onSelect={onSelectModel} />
      </div>

      {empty ? null : short ? (
        <p className="mt-3 text-sm text-ink-soft">
          A paragraph is the floor here. Under {MIN_WORDS} words there is not enough text to
          compare, so the button stays out of reach.
        </p>
      ) : null}

      {note ? <p className="mt-3 text-sm text-ink-soft">{note}</p> : null}

      {problem ? (
        <p className="mt-3 rounded-md border border-danger/40 bg-danger-soft px-4 py-3 text-base text-danger">
          {problem}
        </p>
      ) : null}
    </div>
  );
}

/** True for a drag that carries files, not for a text selection or a tab. */
function carriesFiles(event: { dataTransfer?: DataTransfer | null }): boolean {
  return Array.from(event.dataTransfer?.types ?? []).includes('Files');
}

/** Compact tier switcher: the three detectors, only the installed ones live. */
function ModelSwitcher({
  models,
  activeId,
  onSelect,
}: {
  models: ModelInfo[];
  activeId: string;
  onSelect: (modelId: string) => void;
}) {
  return (
    <div
      role="radiogroup"
      aria-label="Detector"
      className="flex items-center gap-1 rounded-lg border border-rule bg-paper-raised p-1"
    >
      {models.map((candidate) => {
        const selected = candidate.id === activeId;
        const usable = candidate.installed;
        return (
          <button
            key={candidate.id}
            type="button"
            role="radio"
            aria-checked={selected}
            disabled={!usable}
            onClick={() => onSelect(candidate.id)}
            className={`focusable cursor-pointer rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              selected
                ? 'bg-ink text-paper'
                : usable
                  ? 'text-ink-soft hover:bg-paper hover:text-ink'
                  : 'cursor-not-allowed text-ink-faint opacity-50'
            }`}
          >
            {candidate.name}
          </button>
        );
      })}
    </div>
  );
}