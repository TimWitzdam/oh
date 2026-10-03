// Dev tool: rewrite the size fields in src/lib/catalog.ts from the hub, so the
// download verifier can compare against the real published file sizes.
// Usage: node scripts/sync-catalog-sizes.mjs
import { readFile, writeFile } from 'node:fs/promises';

const REPOS = {
  'e5small-int8': {
    repo: 'onnx-community/e5-small-lora-ai-generated-detector-ONNX',
    files: [
      'config.json',
      'onnx/model_int8.onnx',
      'special_tokens_map.json',
      'tokenizer.json',
      'tokenizer_config.json',
      'vocab.txt',
    ],
  },
  'chatgpt-detector-roberta-int8': {
    repo: 'onnx-community/chatgpt-detector-roberta-ONNX',
    files: [
      'config.json',
      'merges.txt',
      'onnx/model_int8.onnx',
      'special_tokens_map.json',
      'tokenizer.json',
      'tokenizer_config.json',
      'vocab.json',
    ],
  },
  'qwen3-06b-detector': {
    repo: 'rasbt/ai-text-detector-qwen3-0.6b-variable',
    files: [
      'config.json',
      'detector-config.json',
      'model.safetensors',
      'tokenizer.json',
      'tokenizer_config.json',
    ],
  },
  'desklib-deberta-v3-large': {
    repo: 'desklib/ai-text-detector-v1.01',
    files: [
      'config.json',
      'model.safetensors',
      'special_tokens_map.json',
      'spm.model',
      'tokenizer.json',
      'tokenizer_config.json',
    ],
  },
};

const catalogPath = new URL('../src/lib/catalog.ts', import.meta.url);
let source = await readFile(catalogPath, 'utf8');

for (const [id, { repo, files }] of Object.entries(REPOS)) {
  const response = await fetch(`https://huggingface.co/api/models/${repo}?blobs=true`);
  const data = await response.json();
  const sizes = new Map(data.siblings.map((s) => [s.rfilename, s.size ?? 0]));

  let total = 0;
  const entries = files.map((file) => {
    const size = sizes.get(file);
    if (!size) throw new Error(`${repo}: ${file} has no published size`);
    total += size;
    return `      { path: '${file}', bytes: ${size} },`;
  });

  const listPattern = new RegExp(
    `(id: '${id}',[\\s\\S]*?files: \\[)[\\s\\S]*?(\\n    \\],\\n    bytes: )[\\d_]+`,
  );
  if (!listPattern.test(source)) {
    // Candidates come and go; a repo that is no longer in the catalog is not an
    // error, it just has nothing to write.
    console.log(`${id}: not in the catalog, skipped`);
    continue;
  }
  source = source.replace(
    listPattern,
    (_match, head, tail) => `${head}\n${entries.join('\n')}${tail}${total}`,
  );
  console.log(`${id}: ${files.length} files, ${(total / 1024 / 1024).toFixed(1)} MB`);
}

await writeFile(catalogPath, source);