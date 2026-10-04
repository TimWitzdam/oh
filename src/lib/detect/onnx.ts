import type { ModelSpec } from '../catalog';
import { MODELS_DIR } from '../paths';
import type { Detector, ScoreUpdate } from './types';

/**
 * Sequence-classifier detectors (BERT/RoBERTa family) run through
 * transformers.js on onnxruntime's CPU backend. Weights are read from the
 * volume that the download manager filled; remote fetches stay disabled so a
 * missing file fails loudly instead of silently reaching for the network.
 */

type TransformersModule = typeof import('@huggingface/transformers');
type DType = NonNullable<
  Parameters<TransformersModule['AutoModelForSequenceClassification']['from_pretrained']>[1]
> extends infer Options
  ? Options extends { dtype?: infer T }
    ? T
    : never
  : never;

let transformersPromise: Promise<TransformersModule> | null = null;

async function loadTransformers(): Promise<TransformersModule> {
  if (!transformersPromise) {
    transformersPromise = import('@huggingface/transformers').then((mod) => {
      mod.env.allowRemoteModels = false;
      mod.env.allowLocalModels = true;
      mod.env.localModelPath = MODELS_DIR;
      mod.env.useFSCache = false;
      return mod;
    });
  }
  return transformersPromise;
}

export class OnnxClassifierDetector implements Detector {
  private loading: Promise<void> | null = null;
  private pipeline: {
    model: Awaited<ReturnType<TransformersModule['AutoModelForSequenceClassification']['from_pretrained']>>;
    tokenizer: Awaited<ReturnType<TransformersModule['AutoTokenizer']['from_pretrained']>>;
  } | null = null;

  constructor(readonly spec: ModelSpec) {}

  ready(): Promise<void> {
    if (!this.loading) {
      this.loading = this.load().catch((error: unknown) => {
        this.loading = null;
        throw error;
      });
    }
    return this.loading;
  }

  private async load(): Promise<void> {
    const { AutoModelForSequenceClassification, AutoTokenizer } = await loadTransformers();
    const options = { dtype: this.spec.dtype as DType, device: 'cpu' as const };
    const tokenizer = await AutoTokenizer.from_pretrained(this.spec.id);
    const model = await AutoModelForSequenceClassification.from_pretrained(this.spec.id, options);
    this.pipeline = { model, tokenizer };
  }

  /**
   * Hands the onnxruntime session back. The arena those tensors live in is
   * native memory, so dropping the last JS reference frees nothing until the
   * session is released - without this the idle sweep is decoration.
   *
   * Clears the memoised load too: a released detector has nothing left to
   * serve, and `ready()` would otherwise resolve instantly against a session
   * that is gone. Callers drop the detector entirely, so the next run builds a
   * fresh one.
   */
  async release(): Promise<void> {
    const pipeline = this.pipeline;
    this.pipeline = null;
    this.loading = null;
    await pipeline?.model.dispose?.();
  }

  async *score(
    windows: string[],
    signal: AbortSignal,
    batchSize: number,
  ): AsyncGenerator<ScoreUpdate> {
    await this.ready();
    const pipeline = this.pipeline;
    if (!pipeline) throw new Error('detector failed to load');
    const { model, tokenizer } = pipeline;

    for (let start = 0; start < windows.length; start += batchSize) {
      throwIfAborted(signal);
      const slice = windows.slice(start, start + batchSize);
      const inputs = tokenizer(slice, {
        padding: true,
        truncation: true,
        max_length: this.spec.maxTokens,
      });
      const output = await model(inputs);
      const rows = toRows(output.logits);
      for (let i = 0; i < slice.length; i += 1) {
        yield { index: start + i, ai: softmaxAi(rows[i], this.spec.aiIndex) };
      }
    }
  }
}


type TensorLike = { tolist?: () => number[][] };

function toRows(value: unknown): number[][] {
  if (Array.isArray(value)) return value as number[][];
  const tensor = value as TensorLike;
  if (typeof tensor?.tolist === 'function') return tensor.tolist();
  throw new Error('unexpected tensor type from model output');
}

function softmaxAi(logits: number[], aiIndex: number): number {
  const max = Math.max(...logits);
  let ai = 0;
  let total = 0;
  for (let i = 0; i < logits.length; i += 1) {
    const value = Math.exp(logits[i] - max);
    total += value;
    if (i === aiIndex) ai = value;
  }
  return total > 0 ? ai / total : 0;
}

export function throwIfAborted(signal: AbortSignal): void {
  if (signal.aborted) throw new DOMException('aborted', 'AbortError');
}