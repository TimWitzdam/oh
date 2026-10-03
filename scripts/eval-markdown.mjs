// Dev tool: how much does markdown formatting move the score?
// Usage: node scripts/eval-markdown.mjs
import { env, AutoTokenizer, AutoModelForSequenceClassification } from '@huggingface/transformers';

env.allowRemoteModels = true;

const CASES = [
  {
    kind: 'ai-plain',
    text: 'Local-first software keeps your data on your own machine. This is not a nice-to-have; it is the whole point. There is no upload latency, it works on a plane, and nobody else can read it.',
  },
  {
    kind: 'ai-markdown',
    text: `## Why Local-First Tools Matter

Local-first software keeps your data **on your own machine**. This is not a
nice-to-have; it is the whole point.

- No upload latency
- Works on a plane
- Nobody else can read it

### Getting Started

1. Install the package
2. Run the server
3. Open your browser

> Offline is a feature, not a fallback.`,
  },
  {
    kind: 'human-plain',
    text: "I tried the hosted version for about six weeks. It was fine, mostly, except the week my internet went out and I lost a day's work to an upload bar. No sync means editing on the train is fine. Backups are my problem now, which is its own problem. What I actually miss is the shared database.",
  },
  {
    kind: 'human-markdown',
    text: `## Why I moved off the hosted thing

I tried the hosted version for about six weeks. It was fine, mostly, except the
week my internet went out and I lost a day's work to an upload bar.

- No sync means editing on the train is fine
- Backups are my problem now, which is its own problem
- The config file is just a file

### What I actually miss

The shared database. Everything else I can live without.`,
  },
  {
    kind: 'ai-formatted-plain',
    text: 'Here are five proven strategies for improving your workflow. First, audit your current process. Second, eliminate the steps that add no value. Third, automate the repetitive parts. Fourth, review results monthly. Fifth, iterate on what changed.',
  },
  {
    kind: 'ai-formatted-markdown',
    text: `Here are five proven strategies for improving your workflow:

1. **Audit your current process** — map every step.
2. Eliminate the steps that add no value.
3. Automate the repetitive parts:
   - start small
   - measure before scaling
4. Review results *monthly*.
5. Iterate on what changed.

> Start with step one today.`,
  },
];

const MODEL = 'onnx-community/chatgpt-detector-roberta-ONNX';

export function stripFormatting(text) {
  return text
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/^#{1,6}\s*/gm, '')
    .replace(/^\s*>\s?/gm, '')
    .replace(/^\s*[-*+]\s+/gm, '')
    .replace(/^\s*(\d+)[.)]\s+/gm, '')
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1$2')
    .replace(/(^|\s)_([^_\n]+)_/g, '$1$2')
    .replace(/^\s*([-*_]\s*){3,}$/gm, ' ')
    .replace(/[ \t]+/g, ' ')
    .replace(/\s*\n+\s*/g, ' ')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

const tokenizer = await AutoTokenizer.from_pretrained(MODEL, { dtype: 'int8' });
const model = await AutoModelForSequenceClassification.from_pretrained(MODEL, {
  dtype: 'int8',
  device: 'cpu',
});

const softmax = (v) => {
  const max = Math.max(...v);
  const e = v.map((x) => Math.exp(x - max));
  return e.map((x) => x / e.reduce((a, b) => a + b, 0));
};

async function ai(text) {
  const inputs = tokenizer(text, { truncation: true, max_length: 512 });
  const out = await model(inputs);
  return softmax(out.logits.tolist()[0])[1];
}

console.log(`model: ${MODEL}\n`);
for (const item of CASES) {
  const raw = await ai(item.text);
  const stripped = await ai(stripFormatting(item.text));
  console.log(
    `${item.kind.padEnd(24)} raw ${(raw * 100).toFixed(1).padStart(6)}%   stripped ${(stripped * 100).toFixed(1).padStart(6)}%   delta ${((stripped - raw) * 100).toFixed(1).padStart(6)}`,
  );
}