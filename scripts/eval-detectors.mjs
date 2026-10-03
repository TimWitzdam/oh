// Dev tool: compare candidate CPU detectors on latency + signal quality.
// Prints P(class 0) and P(class 1) for every passage so label order can be
// verified by hand instead of trusting id2label in config.json.
// Usage: node scripts/eval-detectors.mjs [modelKey ...]
import { env, AutoTokenizer, AutoModelForSequenceClassification } from '@huggingface/transformers';

env.allowLocalModels = true;
env.allowRemoteModels = true;

const passages = [
  ['human-blog', "I lost my job in March and honestly? Part of me was relieved. Three years of sitting in that open plan office, pretending the 8am standup meant something. Anyway, I spent the summer walking the dog way too much and finally fixed the bike. The new job pays less but the commute is four minutes instead of fifty, so I'm counting it as a win. We'll see how long the honeymoon lasts before I start complaining again."],
  ['ai-assistant', "Losing a job can be a profoundly difficult experience, but it is also an opportunity for personal growth. By embracing the uncertainty and taking time to reflect on your career goals, you can emerge from this period with a clearer sense of what truly matters to you. Here are several strategies to help you navigate this transition and emerge stronger on the other side. First, allow yourself to grieve the loss."],
  ['human-academic', "Our results diverge from those of Kowalski et al. (2019), who report no significant effect once sample size is controlled for. We attribute this discrepancy to their exclusion of the pre-2017 cohort, which comprises roughly a third of our observations. Including that cohort, the effect size rises to 0.31, which is consistent with their supplementary analysis."],
  ['ai-marketing', "In today's fast-paced digital landscape, having a strong online presence is no longer optional — it's essential. Whether you're a small business owner or a budding entrepreneur, leveraging the power of social media can unlock unprecedented growth. In this guide, we'll walk you through five proven strategies that will help you connect with your audience and drive meaningful engagement."],
  ['human-short', "he said the door was locked but it wasn't. we went in anyway and the place smelled like wet concrete and old paper"],
  ['ai-short', "The door, though it appeared locked, was in fact ajar, and so we ventured inside, where the air carried the unmistakable scent of damp concrete and aged paper."],
  ['human-review', "Took the 7:42 into the city and it was standing room only the whole way. Some guy on the platform was playing a podcast out loud with no headphones, topically about coupons for meal kits, which somehow made the whole carriage very aware of itself. Got off two stops early and walked instead."],
  ['ai-essay', "Technology has fundamentally transformed the way humans communicate, work, and relate to one another. While these advancements offer remarkable opportunities, they also give rise to complex challenges that society must grapple with. It is our collective responsibility to ensure that innovation serves humanity rather than the other way around. A balanced approach will be essential as we navigate this rapidly evolving landscape."],
];

const MODELS = {
  e5small: { repo: 'onnx-community/e5-small-lora-ai-generated-detector-ONNX', dtype: 'int8' },
  hellosimple: { repo: 'onnx-community/chatgpt-detector-roberta-ONNX', dtype: 'int8' },
  hellosimple_fp: { repo: 'onnx-community/chatgpt-detector-roberta-ONNX', dtype: 'fp16' },
  modernbert: { repo: 'onnx-community/answerdotai-ModernBERT-base-ai-detector-ONNX', dtype: 'int8' },
  roberta_large: { repo: 'onnx-community/roberta-large-openai-detector-ONNX', dtype: 'int8' },
  openai_base: { repo: 'onnx-community/roberta-base-openai-detector-ONNX', dtype: 'q4f16' },
};

const keys = process.argv.slice(2).length ? process.argv.slice(2) : Object.keys(MODELS);

for (const key of keys) {
  const spec = MODELS[key];
  if (!spec) continue;
  const t0 = performance.now();
  try {
    const tokenizer = await AutoTokenizer.from_pretrained(spec.repo, { dtype: spec.dtype });
    const model = await AutoModelForSequenceClassification.from_pretrained(spec.repo, {
      dtype: spec.dtype,
      device: 'cpu',
    });
    const loadMs = performance.now() - t0;

    const t1 = performance.now();
    const rows = [];
    for (const [kind, text] of passages) {
      const inputs = tokenizer(text, { padding: true, truncation: true });
      const out = await model(inputs);
      rows.push([kind, softmax(tolist(out.logits[0]))]);
    }
    const inferMs = (performance.now() - t1) / passages.length;

    console.log(`\n${key}  ${spec.repo} [${spec.dtype}]  load ${(loadMs / 1000).toFixed(1)}s  ${inferMs.toFixed(0)}ms/passage`);
    for (const [kind, [p0, p1]] of rows) {
      console.log(`   ${kind.padEnd(14)} p0=${(p0 * 100).toFixed(1).padStart(6)}%  p1=${(p1 * 100).toFixed(1).padStart(6)}%`);
    }
    for (const aiIndex of [0, 1]) {
      const correct = rows.filter(([k, p]) => (k.startsWith('human') ? p[aiIndex] < 0.5 : p[aiIndex] >= 0.5)).length;
      console.log(`   -> aiIndex ${aiIndex}: ${correct}/${passages.length} correct`);
    }
  } catch (err) {
    console.log(`\n${key}  FAILED after ${((performance.now() - t0) / 1000).toFixed(1)}s: ${err.message}`);
  }
}

function tolist(x) {
  return typeof x?.tolist === 'function' ? x.tolist() : x;
}

function softmax(v) {
  const max = Math.max(...v);
  const e = v.map((x) => Math.exp(x - max));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}