'use client';

import { useId } from 'react';

import type { ModelInfo } from '@/lib/types';
import type { Settings } from '@/lib/catalog';
import { Button, Field, Hint, NumberInput, Slider } from './primitives';

export function SettingsPanel({
  settings,
  models,
  activeModel,
  onPatch,
  onSelectModel,
  onOpenDetectors,
  onClose,
}: {
  settings: Settings;
  models: ModelInfo[];
  activeModel: ModelInfo | null;
  onPatch: (patch: Partial<Settings>) => void;
  onSelectModel: (modelId: string) => void;
  onOpenDetectors: () => void;
  onClose: () => void;
}) {
  const detectorNoteId = useId();
  // Detector scores are not probabilities: each model needs its own cut, so the
  // default belongs to the model and an override is opt-in.
  const modelDefault = activeModel?.threshold ?? 0.5;
  const threshold = settings.threshold ?? modelDefault;
  const overridden = settings.threshold !== null && settings.threshold !== undefined;

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <button
        type="button"
        aria-label="Close settings"
        className="absolute inset-0 bg-ink/15"
        onClick={onClose}
      />
      <div className="relative flex h-full w-full max-w-md flex-col gap-6 overflow-y-auto border-l border-rule bg-paper-raised p-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-semibold tracking-tight text-ink">Settings</h2>
          <Button variant="quiet" onClick={onClose}>
            Close
          </Button>
        </div>

        <section>
          <div className="flex items-center gap-1.5">
            <h3 className="text-lg font-semibold text-ink">Detector</h3>
            <Hint id={detectorNoteId}>
              Weights are cached on the server, so switching tiers is instant once downloaded.
            </Hint>
          </div>
          <div className="mt-3 space-y-2">
            {models.map((model) => {
              const selected = activeModel?.id === model.id;
              return (
                <div
                  key={model.id}
                  className={`flex items-start justify-between gap-3 rounded-md border px-4 py-3 ${
                    selected ? 'border-ink bg-paper' : 'border-rule bg-paper-raised'
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => model.installed && onSelectModel(model.id)}
                    disabled={!model.installed}
                    className="focusable flex-1 text-left disabled:cursor-not-allowed"
                  >
                    <span className="block text-base font-medium text-ink">{model.name}</span>
                    <span className="mt-0.5 block text-sm text-ink-soft">
                      {model.installed
                        ? `about ${model.ramMb} MB RAM`
                        : 'Not downloaded yet'}
                    </span>
                  </button>
                </div>
              );
            })}
          </div>
          <div className="mt-3">
            <Button variant="plain" onClick={onOpenDetectors}>
              Manage downloads
            </Button>
          </div>
        </section>

        <section className="space-y-5">
          <h3 className="text-lg font-semibold text-ink">Scoring</h3>

          <Field
            label="Window size"
            hint="Chunk size inside a paragraph: windows never cross a line break, so a short paragraph is always scored as one window. Larger chunks give the model more context and take longer."
          >
            <Slider
              value={settings.windowChars || activeModel?.windowChars || 560}
              min={250}
              max={1200}
              step={10}
              format={(value) => `${value} chars`}
              onCommit={(value) => onPatch({ windowChars: value })}
            />
          </Field>

          <Field
            label="Highlight threshold"
            hint={
              activeModel
                ? `Sentences at or above this score get underlined and listed. ${activeModel.name}'s default is the highest cut measured to keep false positives under 6% of human documents - every detector scores differently, so there is no universal value.`
                : 'Sentences at or above this machine-written score get underlined and listed.'
            }
          >
            <Slider
              value={Math.round(threshold * 100)}
              min={30}
              max={99}
              step={1}
              format={(value) => `${value}%`}
              onCommit={(value) => onPatch({ threshold: value / 100 })}
            />
            {overridden ? (
              <div className="mt-2">
                <Button variant="quiet" onClick={() => onPatch({ threshold: null })}>
                  Use {activeModel?.name ?? 'the model'} default ({Math.round(modelDefault * 100)}%)
                </Button>
              </div>
            ) : null}
          </Field>

          <Field
            label="Neighbour smoothing"
            hint="Averages each sentence score with this many sentences on either side."
          >
            <NumberInput
              value={settings.smoothing}
              min={0}
              max={3}
              onCommit={(value) => onPatch({ smoothing: value })}
              suffix={settings.smoothing === 1 ? 'sentence each side' : 'sentences each side'}
            />
          </Field>

          <Field
            label="Windows per batch"
            hint="How many windows are scored at once. Higher uses more CPU but finishes sooner."
          >
            <NumberInput
              value={settings.batchSize}
              min={1}
              max={16}
              onCommit={(value) => onPatch({ batchSize: value })}
              suffix="windows"
            />
          </Field>
        </section>

      </div>
    </div>
  );
}