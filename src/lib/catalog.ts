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
  /** Torch tiers: quantise weights to int8 at load time. */
  quantize?: boolean;
  /** Torch tiers: where the classification head reads from. */
  readout?: 'variable-eos' | 'mean-pool-logit';
  /** Optional calibration: divide logits by this before softmax. */
  temperature?: number;
  /**
   * Default highlight threshold for this model, measured on labelled text the
   * model was not trained on (see scripts/fetch-eval-set.py). Detector scores
   * are not probabilities and sit in different places on every model, so one
   * shared default either floods a document with underlines or hides everything.
   * The user can override it; this is what an untouched install uses.
   */
  threshold: number;
  /** Characters of text handed to the model per scoring pass. */
  windowChars: number;
  /** Token ceiling per window; windows are trimmed to this. */
  maxTokens: number;
  files: CatalogFile[];
  bytes: number;
  /** Approximate resident memory once loaded, for the picker. */
  ramMb: number;
  detail: string;
}

const MB = 1024 * 1024;

export const MODELS: ModelSpec[] = [
  {
    id: 'chatgpt-detector-roberta-int8',
    tier: 'lite',
    name: 'Lite',
    repo: 'onnx-community/chatgpt-detector-roberta-ONNX',
    license: 'apache-2.0',
    kind: 'onnx-classifier',
    dtype: 'int8',
    aiIndex: 1,
    // Its scores pile up against 1.0, so the old shared 0.5 flagged a fifth of
    // unseen human documents. 0.99 holds the false-positive rate near 6%.
    threshold: 0.99,
    windowChars: 560,
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
    detail: 'RoBERTa-large encoder',
  },
  {
    id: 'qwen3-06b-detector',
    tier: 'balanced',
    name: 'Balanced',
    repo: 'rasbt/ai-text-detector-qwen3-0.6b-variable',
    license: 'apache-2.0',
    kind: 'torch-classifier',
    aiIndex: 1,
    temperature: 1.4665638128271772,
    quantize: false,
    readout: 'variable-eos',
    // Worth keeping as the one detector that reads the text as a language
    // model rather than a classifier, but it is not more accurate than the
    // Deep tier: measured on unseen domains it scores 0.68 against 0.99, and
    // it saturates so hard that half of human documents come back above 0.9.
    threshold: 0.5,
    windowChars: 900,
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
    detail: 'Qwen3-0.6B detector',
  },
  {
    id: 'desklib-deberta-v3-large',
    tier: 'deep',
    name: 'Deep',
    repo: 'desklib/ai-text-detector-v1.01',
    license: 'mit',
    kind: 'torch-classifier',
    aiIndex: 1,
    // Same backbone as the tier it replaces, but trained on RAID, and its head
    // is a mean pool plus one linear layer over a single logit - so there is no
    // softmax here and no temperature to tune.
    readout: 'mean-pool-logit',
    quantize: false,
    // Measured over 120 documents per set: at 0.9 this model catches 86-100% of
    // machine text while underlining 0-3.5% of human documents. At 0.5 the false
    // positives on unseen domains were 12%.
    threshold: 0.9,
    windowChars: 900,
    maxTokens: 512,
    files: [
      { path: 'config.json', bytes: 890 },
      { path: 'model.safetensors', bytes: 1736100972 },
      { path: 'special_tokens_map.json', bytes: 286 },
      { path: 'spm.model', bytes: 2464616 },
      { path: 'tokenizer.json', bytes: 8656624 },
      { path: 'tokenizer_config.json', bytes: 1315 },
    ],
    bytes: 1747224703,
    ramMb: 2_200,
    detail: 'DeBERTa-v3-large encoder',
  },
];

export const TIER_ORDER: Tier[] = ['lite', 'balanced', 'deep'];

export const DEFAULT_SETTINGS = {
  activeModelId: null as string | null,
  windowChars: 0,
  /** null means "use the active model's own measured default". */
  threshold: null as number | null,
  smoothing: 1,
  maxWords: 12_000,
  batchSize: 8,
};

export type Settings = typeof DEFAULT_SETTINGS;

/** The threshold in force: the user's override, or the model's measured default. */
export function effectiveThreshold(settings: Settings, spec: ModelSpec | undefined): number {
  return settings.threshold ?? spec?.threshold ?? DEFAULT_SETTINGS.threshold ?? 0.5;
}

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