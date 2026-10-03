'use client';

import { useCallback, useSyncExternalStore } from 'react';

/**
 * The box keeps what you typed, in this browser only.
 *
 * A pasted essay is an afternoon of work and the page has no other copy of it: a
 * reload, a crash or a trip to the detector list would otherwise throw it away.
 * localStorage is an external store rather than React state, so it lives here
 * and reaches the composer through useSyncExternalStore - which is also what
 * keeps the server's empty render and the browser's restored draft from being
 * two different versions of the same page.
 *
 * Writes are debounced so typing does not hit storage on every keystroke, and a
 * browser that refuses (private mode, full quota) is reported rather than
 * quietly losing the text.
 */

const KEY = 'oh:draft.v1';
const WRITE_DELAY = 400;

let snapshot = '';
let atLoad: string | null = null;
let stored: boolean | null = null;
let loaded = false;
let listening = false;
let timer: ReturnType<typeof setTimeout> | null = null;
const listeners = new Set<() => void>();

function emit(): void {
  for (const listener of listeners) listener();
}

/** Once, the first time anything asks the store what the draft is. */
function load(): void {
  if (loaded) return;
  loaded = true;
  try {
    snapshot = window.localStorage.getItem(KEY) ?? '';
  } catch {
    snapshot = '';
  }
  atLoad = snapshot || null;
}

function subscribe(listener: () => void): () => void {
  load();
  // The store outlives every component on the page, so the tab listener is
  // added once and left there rather than counted with each subscriber.
  if (!listening) {
    listening = true;
    window.addEventListener('storage', onStorage);
  }
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function read(): string {
  load();
  return snapshot;
}

/** Another tab wrote a draft: take it, so two tabs do not disagree. */
function onStorage(event: StorageEvent): void {
  if (event.key !== null && event.key !== KEY) return;
  snapshot = event.newValue ?? '';
  emit();
}

function write(value: string): void {
  snapshot = value;
  emit();
  if (timer) clearTimeout(timer);
  timer = setTimeout(() => {
    timer = null;
    try {
      if (value) window.localStorage.setItem(KEY, value);
      else window.localStorage.removeItem(KEY);
      stored = true;
    } catch {
      stored = false;
    }
    emit();
  }, WRITE_DELAY);
}

export interface Draft {
  text: string;
  setText: (value: string) => void;
  /** The text as it was found on arrival, or null when the box started empty. */
  atLoad: string | null;
  /** null until storage has been asked; false when it said no. */
  stored: boolean | null;
}

export function useDraft(): Draft {
  const text = useSyncExternalStore(subscribe, read, () => '');
  const loaded = useSyncExternalStore(
    subscribe,
    () => atLoad,
    () => null,
  );
  const canStore = useSyncExternalStore(
    subscribe,
    () => stored,
    () => null,
  );
  const setText = useCallback((value: string) => write(value), []);
  return { text, setText, atLoad: loaded, stored: canStore };
}