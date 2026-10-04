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
   * Default highlight threshold for this model. Detector scores are not
   * probabilities and sit in different places on every model, so one shared
   * default either floods a document with underlines or hides everything.
   * The user can override it; this is what an untouched install uses.
   *
   * The defaults below are measured, not chosen. Each is the highest threshold
   * whose false-positive rate stays under 6% of human documents on the eval
   * sets from `scripts/fetch-eval-set.py`, which are graded by their own
   * domain-only baseline so a set that cannot support a comparison says so
   * before the comparison is believed. Re-measure with
   * `fetch-eval-set.py` then `eval_torch.py --sweep` / `npm run eval:detectors
   * -- --sweep` after changing a checkpoint. An earlier round of figures came
   * from a MAGE sample whose machine half was one domain, where a one-line
   * domain check scored AUROC 0.9375; those numbers are gone.
   */
  threshold: number;
  /** Characters of text handed to the model per scoring pass. */
  windowChars: number;
  /**
   * Measured words per second on the reference host (see the README tables).
   * The composer uses it to say how long a run will take before it starts,
   * which is the difference between waiting and wondering.
   */
  wordsPerSecond: number;
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
    // Measured on the int8 ONNX checkpoint this tier actually ships: 0.99 is
    // the highest cut that keeps the false-positive rate under 6% (5.9% on
    // RAID, 12.5% on MAGE). It is a quiet setting by design - this checkpoint
    // puts human text up against 1.0, so anything lower underlines a quarter of
    // an unseen human document.
    threshold: 0.99,
    windowChars: 560,
    wordsPerSecond: 3900,
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
    // MEASURED, and the measurement is not good. This is the only checkpoint
    // here with no in-domain contamination on any eval set (it was trained on
    // rasbt/human-vs-ai-50k), so its out-of-domain numbers are the honest ones:
    // AUROC 0.598 on the MAGE test split, against 0.968 for the Deep tier and
    // 0.786 for Lite. It is the weakest of the three by a wide margin, and
    // worse than the cheap tier it sits above in the picker.
    //
    // No threshold rescues it. The best point on the whole curve is 11.3% of
    // machine text at a 5.6% false-positive rate; at 0.5 it catches 7.8% while
    // flagging 2.1% of human documents. Three MAGE domains come out
    // *anti-correlated* - squad 0.229, sci_gen 0.236, tldr 0.248 - it ranks
    // human text above machine text. This is the same "no usable threshold
    // outside its training data" test that ruled out TMR, and this fails it.
    //
    // Kept at 0.5 because that is where the measurement puts it and moving it
    // buys nothing: 0.3 reaches 8.3%, 0.95 reaches 5.9% with the same 0%
    // false positives. It is here because it reads text as a language model
    // rather than a classifier, which is a different kind of wrong. Anyone
    // picking a tier for accuracy should not pick this one.
    threshold: 0.5,
    windowChars: 900,
    wordsPerSecond: 195,
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
    // Measured. 0.9 catches 87% of machine text on the MAGE test split while
    // underlining 4.2% of human documents, and 86% at 1.4% on RAID. Its 0.968
    // AUROC there is the reason this checkpoint is the Deep tier at all: the
    // textsight-v23 it replaced reads 0.815 on the same documents, so the swap
    // holds up - by 0.15 AUROC rather than the 0.35 a confounded eval set had
    // implied. At 0.5 the false-positive rate on unseen domains is 20%.
    threshold: 0.9,
    windowChars: 900,
    wordsPerSecond: 172,
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