'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';

import type { Settings } from '@/lib/catalog';
import { fetchState, removeModel, saveSettings } from '@/lib/client';
import type { AppState, ModelInfo } from '@/lib/types';
import { DetectorPicker } from './DetectorPicker';
import { SettingsPanel } from './SettingsPanel';
import { Button } from './primitives';
import { useDownloads } from './useDownloads';
import { Workspace } from './Workspace';

type View = 'analyze' | 'detectors';

export function App({ initialState }: { initialState: AppState }) {
  const [state, setState] = useState<AppState>(initialState);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>('analyze');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setState(await fetchState());
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }, []);

  const patch = useCallback(async (update: Partial<Settings>) => {
    const settings = await saveSettings(update);
    setState((previous) => ({ ...previous, settings }));
  }, []);

  // Writes and reads are kept strictly in sequence: two refreshes in flight at
  // once let a stale snapshot win and undo the write that came after it.
  const activateIfIdle = useCallback(
    async (modelId: string, current: string | null) => {
      if (current) {
        await refresh();
        return;
      }
      await saveSettings({ activeModelId: modelId });
      await refresh();
    },
    [refresh],
  );

  const { jobs, start, cancel, rehydrate } = useDownloads((job) => {
    if (job.status === 'done') {
      void activateIfIdle(job.modelId, state.settings.activeModelId);
    } else {
      void refresh();
    }
  });

  // Picks up downloads that were already running when the page was loaded.
  useEffect(() => {
    rehydrate(initialState.downloads);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const models = state.models;
  const activeModel: ModelInfo | null = useMemo(
    () => models.find((model) => model.id === state.settings.activeModelId && model.installed) ?? null,
    [models, state.settings.activeModelId],
  );
  const anyInstalled = models.some((model) => model.installed);
  const showDetectors = view === 'detectors' || !anyInstalled;

  const handleStart = useCallback(
    async (modelId: string) => {
      setBusy(true);
      try {
        await start(modelId);
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : String(caught));
      } finally {
        setBusy(false);
      }
    },
    [start],
  );

  const handleRemove = useCallback(
    async (modelId: string) => {
      setBusy(true);
      try {
        await removeModel(modelId);
        await saveSettings({ activeModelId: null });
        await refresh();
      } finally {
        setBusy(false);
      }
    },
    [refresh],
  );

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-6xl flex-col px-5 pb-16">
      <header className="sticky top-0 z-30 -mx-5 border-b border-rule bg-paper/95 px-5 py-4 backdrop-blur">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="group relative text-2xl font-semibold tracking-tight text-ink">
            <span
              tabIndex={0}
              aria-describedby="name-note"
              className="cursor-help rounded-sm underline decoration-dotted decoration-rule-strong underline-offset-4"
            >
              oh
            </span>
            <span
              id="name-note"
              role="tooltip"
              className="pointer-events-none absolute left-0 top-full z-40 mt-2 w-56 rounded-md border border-rule bg-paper-raised px-3 py-2 text-sm font-normal leading-snug text-ink-soft opacity-0 shadow-sm transition-opacity duration-150 group-hover:opacity-100 group-focus-within:opacity-100"
            >
              The name is the sound you make once you find out your paper was flagged.
            </span>
          </h1>
          <span className="text-base text-ink-soft">
            {activeModel
              ? `${activeModel.name} detector · ${formatRam(activeModel.ramMb)} of weights`
              : 'no detector yet'}
          </span>
          <span className="flex-1" />
          {anyInstalled ? (
            <Button variant="quiet" onClick={() => setView(showDetectors ? 'analyze' : 'detectors')}>
              {showDetectors ? 'Back to analysis' : 'Detectors'}
            </Button>
          ) : null}
          <Button variant="plain" onClick={() => setSettingsOpen(true)}>
            Settings
          </Button>
        </div>
      </header>

      {error ? (
        <p className="mt-4 rounded-md border border-danger/40 bg-danger-soft px-4 py-3 text-base text-danger">
          {error}
        </p>
      ) : null}

      <main className="flex-1 pt-8">
        {showDetectors ? (
          <section>
            <h2 className="text-3xl font-semibold tracking-tight text-ink">Choose a detector</h2>
            <div className="mt-6">
              <DetectorPicker
                models={models}
                jobs={jobs}
                activeModelId={state.settings.activeModelId}
                busy={busy}
                onStart={handleStart}
                onCancel={(jobId) => void cancel(jobId)}
                onRemove={handleRemove}
                onActivate={async (modelId) => {
                  await patch({ activeModelId: modelId });
                  setView('analyze');
                }}
              />
            </div>
          </section>
        ) : activeModel ? (
          <Workspace
            model={activeModel}
            models={models}
            threshold={state.settings.threshold}
            maxWords={state.settings.maxWords}
            onSelectModel={(modelId) => void patch({ activeModelId: modelId })}
          />
        ) : (
          <p className="text-base text-ink-soft">Pick a detector to start analysing text.</p>
        )}
      </main>

      {settingsOpen ? (
        <SettingsPanel
          settings={state.settings}
          models={models}
          activeModel={activeModel}
          onPatch={(update) => void patch(update)}
          onSelectModel={(modelId) => void patch({ activeModelId: modelId })}
          onRemoveModel={handleRemove}
          onOpenDetectors={() => {
            setSettingsOpen(false);
            setView('detectors');
          }}
          onClose={() => setSettingsOpen(false)}
        />
      ) : null}
    </div>
  );
}

function formatRam(megabytes: number): string {
  return megabytes >= 1000 ? `${(megabytes / 1000).toFixed(1)} GB` : `${megabytes} MB`;
}