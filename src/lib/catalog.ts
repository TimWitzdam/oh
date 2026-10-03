/**
 * Model catalog: the three selectable detector tiers.
 *
 * Every entry is a plain Hugging Face repo plus the file list needed to run it
 * offline. Weights are fetched once by the download manager into
 * DATA_DIR/models/<id>/ together with a manifest (see model.ts) that the
 * inference layer reads back, so nothing about a model lives in two places.
 */

export type Tier = 'lite' | 'balanced' | 'deep';
export type DetectorKind = 'onnx-classifier' | 'torch-classifier';

export interface CatalogFile {
  path: string;
  bytes: number;
}

export interface ModelSpec {
  id: string;
  tier: Tier;
  name: string;
  repo: string;
  license: string;
  kind: DetectorKind;
  /** ONNX precision for transformers.js. */
  dtype?: string;
  /** Logit index that means "machine written". */
  aiIndex: number;
  /** Torch tiers: quantise weights to int8 at load time (off for the deep tier). */
  quantize?: boolean;
  /** Torch tiers: where the classification head reads from. */
  readout?: 'variable-eos';
  /** Optional calibration: divide logits by this before softmax. */
  temperature?: number;
  /** Characters of text handed to the model per scoring pass. */
  windowChars: number;
  /** Measured scoring cost per window on CPU, in milliseconds. */
  msPerWindow: number;
  /** Token ceiling per window; windows are trimmed to this. */
  maxTokens: number;
  files: CatalogFile[];
  bytes: number;
  /** Approximate resident memory once loaded, for the picker. */
  ramMb: number;
  detail: string;
  tradeoff: string;
}

const MB = 1024 * 1024;

export const MODELS: ModelSpec[] = [
  {
    id: 'e5small-int8',
    tier: 'lite',
    name: 'Lite',
    repo: 'onnx-community/e5-small-lora-ai-generated-detector-ONNX',
    license: 'apache-2.0',
    kind: 'onnx-classifier',
    dtype: 'int8',
    aiIndex: 1,
    windowChars: 420,
    msPerWindow: 6,
    maxTokens: 384,
    files: [
      { path: 'config.json', bytes: 709 },
      { path: 'onnx/model_int8.onnx', bytes: 34157539 },
      { path: 'special_tokens_map.json', bytes: 695 },
      { path: 'tokenizer.json', bytes: 711396 },
      { path: 'tokenizer_config.json', bytes: 1301 },
      { path: 'vocab.txt', bytes: 231508 },
    ],
    bytes: 35103148,
    ramMb: 220,
    detail: '33M parameter BERT encoder, int8 quantised, loads in about a second.',
    tradeoff: 'Weakest tier: it gives up a lot on short passages and on AI text that was edited afterwards.',
  },
  {
    id: 'chatgpt-detector-roberta-int8',
    tier: 'balanced',
    name: 'Balanced',
    repo: 'onnx-community/chatgpt-detector-roberta-ONNX',
    license: 'apache-2.0',
    kind: 'onnx-classifier',
    dtype: 'int8',
    aiIndex: 1,
    windowChars: 560,
    msPerWindow: 15,
    maxTokens: 512,
    files: [
      { path: 'config.json', bytes: 914 },
      { path: 'merges.txt', bytes: 456318 },
      { path: 'onnx/model_int8.onnx', bytes: 125855419 },
      { path: 'special_tokens_map.json', bytes: 964 },
      { path: 'tokenizer.json', bytes: 3558745 },
      { path: 'tokenizer_config.json', bytes: 1358 },
      { path: 'vocab.json', bytes: 798293 },
    ],
    bytes: 130672011,
    ramMb: 420,
    detail: 'RoBERTa-large encoder trained across many generators, int8 quantised.',
    tradeoff: 'Still an encoder: it scores style, not authorship.',
  },
  {
    id: 'qwen3-06b-detector',
    tier: 'deep',
    name: 'Deep',
    repo: 'rasbt/ai-text-detector-qwen3-0.6b-variable',
    license: 'apache-2.0',
    kind: 'torch-classifier',
    aiIndex: 1,
    temperature: 1.4665638128271772,
    quantize: false,
    readout: 'variable-eos',
    windowChars: 900,
    msPerWindow: 180,
    maxTokens: 1023,
    files: [
      { path: 'config.json', bytes: 1583 },
      { path: 'detector-config.json', bytes: 994 },
      { path: 'model.safetensors', bytes: 1192139280 },
      { path: 'tokenizer.json', bytes: 11422749 },
      { path: 'tokenizer_config.json', bytes: 694 },
    ],
    bytes: 1203565300,
    ramMb: 2_700,
    detail: 'Qwen3-0.6B detector: a fine-tuned language model reading out at the last token, with temperature scaling.',
    tradeoff: 'Biggest and slowest tier by a wide margin: it reads each passage most carefully and is worth it when accuracy matters more than speed.',
  },
];

export const TIER_ORDER: Tier[] = ['lite', 'balanced', 'deep'];

export const DEFAULT_SETTINGS = {
  activeModelId: null as string | null,
  windowChars: 0,
  threshold: 0.5,
  smoothing: 1,
  maxWords: 12_000,
  batchSize: 8,
};

export type Settings = typeof DEFAULT_SETTINGS;

export function findModel(id: string | null | undefined): ModelSpec | undefined {
  if (!id) return undefined;
  return MODELS.find((model) => model.id === id);
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1000 * MB) return `${(bytes / (1000 * MB)).toFixed(1)} GB`;
  if (bytes >= MB) return `${Math.round(bytes / MB)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} kB`;
  return `${bytes} B`;
}