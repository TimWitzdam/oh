// Dev tool: compare candidate CPU detectors on latency + labelled signal quality.
//
//   node scripts/eval-detectors.mjs                  # hand-written passages only
//   node scripts/eval-detectors.mjs --set raid       # + metrics over eval/raid.jsonl
//   node scripts/eval-detectors.mjs tmr_int8         # only these candidates
//
// The hand-written passages print P(class 0)/P(class 1) per passage so the label
// order can be verified by hand instead of trusting id2label in config.json.
// Over a labelled set the metrics are AUROC (ranking quality, threshold-free),
// TPR at 1% and 5% FPR (what a conservative reviewer setting needs), and
// accuracy at the 5% FPR threshold.
//
// Build the sets first: python3 scripts/fetch-eval-set.py
//
// Contamination: `raid` is the labelled split of the corpus the tmr, desklib
// and modernbert-mage candidates were trained on, and `control` is the MAGE test
// split, which modernbert-mage also trained on. `hc3` is clean for every
// candidate except hellosimple, whose checkpoint was fine-tuned on it. So each
// set flatters one family and only the set that flatters your own pick is worth
// believing least.
import { readFileSync, existsSync } from 'node:fs';
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
  // Shipped tiers, for a like-for-like reference point.
  hellosimple: { repo: 'onnx-community/chatgpt-detector-roberta-ONNX', dtype: 'int8', shipped: 'Lite' },
  // RAID-trained candidates.
  tmr_int8: { repo: 'onnx-community/tmr-ai-text-detector-ONNX', dtype: 'int8' },
  tmr_fp32: { repo: 'onnx-community/tmr-ai-text-detector-ONNX', dtype: 'fp32' },
  mage_int8: { repo: 'onnx-community/modernbert-ai-detection-raid-mage-ONNX', dtype: 'int8' },
  mage_fp32: { repo: 'onnx-community/modernbert-ai-detection-raid-mage-ONNX', dtype: 'fp32' },
  // Everything else that was measured before.
  e5small: { repo: 'onnx-community/e5-small-lora-ai-generated-detector-ONNX', dtype: 'int8' },
  hellosimple_fp: { repo: 'onnx-community/chatgpt-detector-roberta-ONNX', dtype: 'fp16' },
  modernbert: { repo: 'onnx-community/answerdotai-ModernBERT-base-ai-detector-ONNX', dtype: 'int8' },
  roberta_large: { repo: 'onnx-community/roberta-large-openai-detector-ONNX', dtype: 'int8' },
  openai_base: { repo: 'onnx-community/roberta-base-openai-detector-ONNX', dtype: 'q4f16' },
};

const args = process.argv.slice(2);
const sets = args.includes('--set')
  ? [args[args.indexOf('--set') + 1]].filter(Boolean)
  : ['raid', 'control', 'hc3'].filter((name) => existsSync(new URL(`../eval/${name}.jsonl`, import.meta.url)));
const keys = args.filter((arg) => !arg.startsWith('--') && MODELS[arg]);
const picked = keys.length ? keys : Object.keys(MODELS);

const loaded = sets.map(loadSet);
for (const set of loaded) {
  if (!set) continue;
  const humans = set.rows.filter((r) => r.label === 0).length;
  const domains = new Set(set.rows.map((r) => r.domain)).size;
  const baseline = domainBaseline(set);
  console.log(`eval/${set.name}.jsonl: ${set.rows.length} docs (${humans} human, ${domains} domains)`);
  console.log(`  domain-only AUROC ${baseline.toFixed(4)}${baseline > 0.75 ? '   <- set is domain-confounded' : ''}`);
}
if (sets.length && loaded.some((s) => !s)) {
  console.log('run: python3 scripts/fetch-eval-set.py');
}

for (const key of picked) {
  const spec = MODELS[key];
  const t0 = performance.now();
  let model = null;
  try {
    const tokenizer = await AutoTokenizer.from_pretrained(spec.repo, { dtype: spec.dtype });
    model = await AutoModelForSequenceClassification.from_pretrained(spec.repo, {
      dtype: spec.dtype,
      device: 'cpu',
    });
    const loadMs = performance.now() - t0;

    console.log(`\n=== ${key}  ${spec.repo} [${spec.dtype}]${spec.shipped ? `  <- shipped ${spec.shipped}` : ''}  load ${(loadMs / 1000).toFixed(1)}s`);

    const t1 = performance.now();
    const rows = [];
    for (const [kind, text] of passages) {
      const inputs = tokenizer(text, { padding: true, truncation: true });
      const out = await model(inputs);
      rows.push([kind, softmax(tolist(out.logits[0]))]);
    }
    console.log(`   hand-written passages, ${((performance.now() - t1) / passages.length).toFixed(0)}ms/doc`);
    for (const [kind, [p0, p1]] of rows) {
      console.log(`     ${kind.padEnd(14)} p0=${(p0 * 100).toFixed(1).padStart(6)}%  p1=${(p1 * 100).toFixed(1).padStart(6)}%`);
    }

    // Score every set before deciding anything, so the label order can be
    // settled once for the whole candidate instead of once per set.
    const caches = new Map();
    for (const set of loaded) {
      if (!set) continue;
      const started = performance.now();
      const scores = [];
      for (let i = 0; i < set.rows.length; i++) {
        const inputs = tokenizer(set.rows[i].text, { padding: true, truncation: true });
        const out = await model(inputs);
        scores.push(tolist(out.logits[0]));
        if ((i + 1) % 100 === 0) process.stdout.write(`   ...${i + 1}/${set.rows.length}\r`);
      }
      const ms = (performance.now() - started) / set.rows.length;
      process.stdout.write(' '.repeat(30) + '\r');
      caches.set(set, { scores, ms });
    }

    // Which logit means "machine written" is a property of the checkpoint, not
    // of the dataset, so it is decided once per candidate from every set at
    // once and then applied everywhere.
    //
    // This used to be decided per set, independently, by keeping whichever
    // index scored the higher AUROC. That reports max(auroc, 1 - auroc) for
    // each set on its own, so a candidate at 0.64 could never read below 0.5,
    // the number moved when the sets changed, and a model that simply reads
    // the domain cue in one set was rewarded for it.
    const totals = [0, 0];
    for (const set of loaded) {
      if (!set) continue;
      const { scores } = caches.get(set);
      for (const aiIndex of [0, 1]) {
        const value = auroc(set.rows.map((row, i) => [row.label, softmax(scores[i])[aiIndex]]));
        if (!Number.isNaN(value)) totals[aiIndex] += value;
      }
    }
    const aiIndex = totals[0] >= totals[1] ? 0 : 1;

    for (const set of loaded) {
      if (!set) continue;
      const { scores, ms } = caches.get(set);
      const result = report(set, scores, aiIndex);
      const flipped = auroc(set.rows.map((row, i) => [row.label, softmax(scores[i])[1 - aiIndex]]));
      const note = flipped > result.auroc + 0.02 ? `  (label order disagrees here: logit ${1 - aiIndex} would read ${flipped.toFixed(3)})` : '';
      console.log(`   ${set.name}: ${ms.toFixed(0)}ms/doc, ai = logit ${aiIndex}${note}`);
      // `thr` is where a 5% false-positive rate puts the cut. The app ships a
      // threshold per model now, so a score nowhere near it means the ranking
      // is fine but the number is not a probability.
      console.log(`     AUROC ${result.auroc.toFixed(4)}   TPR@1%FPR ${(result.tpr1 * 100).toFixed(1)}%   TPR@5%FPR ${(result.tpr5 * 100).toFixed(1)}%   acc@5%FPR ${(result.acc5 * 100).toFixed(1)}%   thr@5%FPR ${result.cutoff5.toFixed(4)}   human median p(ai) ${result.humanMedian.toFixed(3)}`);
      if (result.domainBaseline > 0.75) {
        console.log(`     domain-only baseline ${result.domainBaseline.toFixed(4)} - the halves differ in domain, so this figure is not a detector result`);
      }
      const worst = result.domains.slice(0, 3);
      if (worst.length) console.log(`     weakest domains: ${worst.map((d) => `${d.name} ${d.auroc.toFixed(3)} (n=${d.n})`).join('  ')}`);
      else if (result.domainBaseline <= 0.75) console.log('     no domain has 10 human and 10 machine documents');

      // The app ships a measured threshold per model, so a candidate is only
      // shippable if some threshold keeps the false-positive rate sane on every
      // set at once. AUROC alone does not say whether one exists.
      if (args.includes('--sweep')) {
        for (const threshold of [0.5, 0.7, 0.8, 0.9, 0.93, 0.95, 0.97, 0.99]) {
          const { fpr, tpr } = rates(set, scores, aiIndex, threshold);
          console.log(`       thr ${threshold.toFixed(2)}  TPR ${(tpr * 100).toFixed(1).padStart(5)}%  FPR ${(fpr * 100).toFixed(1).padStart(5)}%`);
        }
      }
    }
  } catch (err) {
    console.log(`\n=== ${key}  FAILED after ${((performance.now() - t0) / 1000).toFixed(1)}s: ${err.message}`);
  }
  model = null;
  globalThis.gc?.();
}

function loadSet(name) {
  const path = new URL(`../eval/${name}.jsonl`, import.meta.url);
  if (!existsSync(path)) return null;
  const rows = readFileSync(path, 'utf8')
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  return { name, rows };
}

function report(set, scores, aiIndex) {
  const pairs = set.rows.map((row, i) => [row.label, softmax(scores[i])[aiIndex]]);
  const humans = pairs.filter(([label]) => label === 0).map(([, p]) => p).sort((a, b) => a - b);
  const cutoff = (fpr) => quantile(humans, 1 - fpr);
  const at1 = cutoff(0.01);
  const at5 = cutoff(0.05);
  const correct = pairs.filter(([label, p]) => (label === 1 ? p >= at5 : p < at5)).length;

  const domains = new Map();
  for (const row of set.rows) {
    const bucket = domains.get(row.domain) ?? { human: 0, ai: 0 };
    bucket[row.label === 1 ? 'ai' : 'human']++;
    domains.set(row.domain, bucket);
  }
  const perDomain = [...domains]
    .filter(([, counts]) => counts.human >= 10 && counts.ai >= 10)
    .map(([name]) => {
      const subset = pairs.filter((_, i) => set.rows[i].domain === name);
      return { name, n: subset.length, auroc: auroc(subset) };
    })
    .sort((a, b) => a.auroc - b.auroc);

  return {
    aiIndex,
    auroc: auroc(pairs),
    tpr1: tprAt(pairs, at1),
    tpr5: tprAt(pairs, at5),
    acc5: correct / pairs.length,
    cutoff5: at5,
    humanMedian: quantile(humans, 0.5),
    domainBaseline: domainBaseline(set),
    domains: perDomain,
  };
}

/**
 * AUROC of the best one-domain guess - 'this came from domain D' - over D. No
 * weights, no inference. If this is high the two halves of the set differ in
 * domain as well as in authorship, every detector is handed the same shortcut,
 * and a model's AUROC here is partly a measurement of the sampling. Printed next
 * to every model's number so a set that cannot support a comparison says so
 * before the comparison is believed.
 */
function domainBaseline(set) {
  let best = 0;
  for (const name of new Set(set.rows.map((row) => row.domain))) {
    const pairs = set.rows.map((row) => [row.label, row.domain === name ? 1 : 0]);
    const value = auroc(pairs);
    if (!Number.isNaN(value) && value > best) best = value;
  }
  return best;
}

/** False- and true-positive rates at one fixed threshold. */
function rates(set, scores, aiIndex, threshold) {
  let ai = 0;
  let aiHit = 0;
  let human = 0;
  let humanHit = 0;
  for (let i = 0; i < set.rows.length; i++) {
    const score = softmax(tolist(scores[i]))[aiIndex];
    if (set.rows[i].label === 1) {
      ai++;
      if (score >= threshold) aiHit++;
    } else {
      human++;
      if (score >= threshold) humanHit++;
    }
  }
  return { tpr: ai ? aiHit / ai : NaN, fpr: human ? humanHit / human : NaN };
}

/** Rank-based AUROC with ties averaged, so saturated probabilities do not
 *  quietly read as perfect separation. */
function auroc(pairs) {
  const sorted = pairs
    .map(([label, score]) => ({ label, score }))
    .sort((a, b) => a.score - b.score);
  const positives = pairs.filter(([label]) => label === 1).length;
  const negatives = pairs.length - positives;
  if (!positives || !negatives) return Number.NaN;
  let rank = 1;
  let sum = 0;
  for (let i = 0; i < sorted.length; ) {
    let j = i;
    while (j + 1 < sorted.length && sorted[j + 1].score === sorted[i].score) j++;
    const avg = (rank + (rank + (j - i))) / 2;
    for (let k = i; k <= j; k++) if (sorted[k].label === 1) sum += avg;
    rank += j - i + 1;
    i = j + 1;
  }
  return (sum - (positives * (positives + 1)) / 2) / (positives * negatives);
}

function tprAt(pairs, threshold) {
  const ai = pairs.filter(([label]) => label === 1);
  return ai.length ? ai.filter(([, score]) => score >= threshold).length / ai.length : Number.NaN;
}

function quantile(sorted, q) {
  if (!sorted.length) return Number.NaN;
  const pos = (sorted.length - 1) * q;
  const low = Math.floor(pos);
  const high = Math.ceil(pos);
  if (low === high) return sorted[low];
  return sorted[low] + (sorted[high] - sorted[low]) * (pos - low);
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