"""Dev tool: compare candidate torch detectors through the same path the app uses.

    python scripts/eval_torch.py                     # every candidate
    python scripts/eval_torch.py textsight_v23 desklib

Two output shapes per candidate:

  * the hand-written passages, printed as raw per-class scores so the label
    order can be checked by hand instead of trusting id2label;
  * AUROC / TPR@FPR / threshold over the labelled sets from
    `python3 scripts/fetch-eval-set.py`, using the same scoring each model is
    shipped with - the textsight logit margin included, because its softmax
    saturates and is the reason the deep tier needs the special case at all.

Needs a torch install: `service/.venv`, or the image this repo ships, which has
one at /srv/venv.
"""

from __future__ import annotations

import collections
import json
import math
import pathlib
import random
import sys
import time

import torch
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

torch.set_num_threads(max(1, torch.get_num_threads() or 4))

EVAL_DIR = pathlib.Path("eval")

PASSAGES = [
    ("human-blog", "I lost my job in March and honestly? Part of me was relieved. Three years of sitting in that open plan office, pretending the 8am standup meant something. Anyway, I spent the summer walking the dog way too much and finally fixed the bike. The new job pays less but the commute is four minutes instead of fifty, so I'm counting it as a win. We'll see how long the honeymoon lasts before I start complaining again."),
    ("ai-assistant", "Losing a job can be a profoundly difficult experience, but it is also an opportunity for personal growth. By embracing the uncertainty and taking time to reflect on your career goals, you can emerge from this period with a clearer sense of what truly matters to you. Here are several strategies to help you navigate this transition and emerge stronger on the other side. First, allow yourself to grieve the loss."),
    ("human-academic", "Our results diverge from those of Kowalski et al. (2019), who report no significant effect once sample size is controlled for. We attribute this discrepancy to their exclusion of the pre-2017 cohort, which comprises roughly a third of our observations. Including that cohort, the effect size rises to 0.31, which is consistent with their supplementary analysis."),
    ("ai-marketing", "In today's fast-paced digital landscape, having a strong online presence is no longer optional — it's essential. Whether you're a small business owner or a budding entrepreneur, leveraging the power of social media can unlock unprecedented growth. In this guide, we'll walk you through five proven strategies that will help you connect with your audience and drive meaningful engagement."),
    ("human-short", "he said the door was locked but it wasn't. we went in anyway and the place smelled like wet concrete and old paper"),
    ("ai-short", "The door, though it appeared locked, was in fact ajar, and so we ventured inside, where the air carried the unmistakable scent of damp concrete and aged paper."),
    ("human-review", "Took the 7:42 into the city and it was standing room only the whole way. Some guy on the platform was playing a podcast out loud with no headphones, topically about coupons for meal kits, which somehow made the whole carriage very aware of itself. Got off two stops early and walked instead."),
    ("ai-essay", "Technology has fundamentally transformed the way humans communicate, work, and relate to one another. While these advancements offer remarkable opportunities, they also give rise to complex challenges that society must grapple with. It is our collective responsibility to ensure that innovation serves humanity rather than the other way around. A balanced approach will be essential as we navigate this rapidly evolving landscape."),
]

CANDIDATES = {
    # The shipped deep tier, plus the same weights scored the naive way, so the
    # logit-margin special case earns its place in the catalog.
    "textsight_v23": {
        "repo": "textsightai/textsight-detector-v23-custom",
        "readout": "cls",
        "temperature": 4.0,
        "variants": ["margin", "softmax"],
    },
    # Its single logit is already a logit, so temperature 1: the model was
    # trained to emit a probability and sigmoid() is its own answer.
    "desklib": {
        "repo": "desklib/ai-text-detector-v1.01",
        "readout": "desklib",
        "temperature": 1.0,
        "variants": ["margin"],
    },
    "desklib_academic": {
        "repo": "desklib/ai-text-detector-academic-v1.01",
        "readout": "desklib",
        "temperature": 1.0,
        "variants": ["margin"],
    },
    # The shipped balanced tier, as shipped (no quantisation) and quantised.
    "qwen06_fp32": {
        "repo": "rasbt/ai-text-detector-qwen3-0.6b-variable",
        "readout": "variable-eos",
        "temperature": 1.4665638128271772,
        "variants": ["softmax"],
    },
    "qwen06_int8": {
        "repo": "rasbt/ai-text-detector-qwen3-0.6b-variable",
        "readout": "variable-eos",
        "temperature": 1.4665638128271772,
        "quantize": True,
        "variants": ["softmax"],
    },
    "hellosimple_fp32": {
        "repo": "Hello-SimpleAI/chatgpt-detector-roberta",
        "readout": "cls",
        "variants": ["softmax"],
    },
}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0


def build(spec):
    """Returns (logit_fn, tokenizer) where logit_fn(text) -> raw class logits.

    One model load serves every variant, so textsight's margin and softmax come
    out of a single pass over the weights.
    """
    tokenizer = AutoTokenizer.from_pretrained(spec["repo"])
    readout = spec["readout"]

    if readout == "desklib":
        # desklib ships no modelling code: it is a deberta-v3-large encoder, mean
        # pooling over the attention mask, and a single linear logit. The
        # checkpoint nests the encoder under `model.` (as a
        # DebertaV2ForSequenceClassification would) and adds
        # `classifier.weight [1, 1024]`, so from_pretrained silently builds a
        # random encoder. The weights are loaded by hand instead.
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        config = AutoConfig.from_pretrained(spec["repo"])
        config.architectures = ["DebertaV2Model"]
        encoder = AutoModel.from_pretrained(spec["repo"], config=config, ignore_mismatched_sizes=True)
        blob = load_file(hf_hub_download(spec["repo"], "model.safetensors"))
        weight = blob["classifier.weight"].float()
        bias = blob["classifier.bias"].float()
        encoder_state = {
            key[len("model."):]: value
            for key, value in blob.items()
            if key.startswith("model.")
        }
        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
        if missing:
            raise RuntimeError(f"desklib encoder is missing {len(missing)} weights, e.g. {missing[:3]}")
        encoder.eval()
        del blob, encoder_state

        def logit_fn(text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.inference_mode():
                hidden = encoder(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                return [0.0, float((pooled @ weight.T + bias)[0])]

        return logit_fn, tokenizer

    model = AutoModelForSequenceClassification.from_pretrained(spec["repo"], dtype=torch.float32).eval()
    if spec.get("quantize"):
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    if readout == "variable-eos":
        # The variable-position models read their head off an appended EOS.
        def logit_fn(text):
            ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=1022)["input_ids"]
            ids = ids + [tokenizer.eos_token_id]
            inputs = {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            }
            with torch.inference_mode():
                return model(**inputs).logits[0].float().tolist()

        return logit_fn, tokenizer

    def logit_fn(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.inference_mode():
            return model(**inputs).logits[0].float().tolist()

    return logit_fn, tokenizer


def variant_scores(spec, variant, logits):
    """Raw logits -> [score_human, score_ai]."""
    if variant == "margin":
        # Unbounded gap between the two class logits, squashed for display. On
        # the textsight checkpoint the softmax saturates: it ties 77% of a
        # 2,520-document benchmark onto the same rounded value.
        value = sigmoid((logits[1] - logits[0]) / spec.get("temperature", 4.0))
        return [1.0 - value, value]
    top = max(logits)
    exp = [math.exp(x - top) for x in logits]
    total = sum(exp)
    return [x / total for x in exp]


def load_sets(names, limit=None):
    """--limit takes a seeded random slice, because the files are grouped by
    domain and a prefix would drop whole domains."""
    sets = []
    for name in names:
        path = EVAL_DIR / f"{name}.jsonl"
        if not path.is_file():
            print(f"(no eval/{name}.jsonl - run scripts/fetch-eval-set.py)")
            continue
        # split("\n"), not splitlines(): some source documents contain U+2028 and
        # friends, which splitlines() treats as line breaks and json does not escape.
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
        if limit and len(rows) > limit:
            rows = random.Random(11).sample(rows, limit)
        sets.append((name, rows))
    return sets


def auroc(pairs):
    ordered = sorted(pairs, key=lambda pair: pair[1])
    positives = sum(1 for label, _ in ordered if label == 1)
    negatives = len(ordered) - positives
    if not positives or not negatives:
        return float("nan")
    rank, total = 1, 0.0
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1][1] == ordered[index][1]:
            end += 1
        average = (rank + (rank + (end - index))) / 2
        for position in range(index, end + 1):
            if ordered[position][0] == 1:
                total += average
        rank += end - index + 1
        index = end + 1
    return (total - (positives * (positives + 1)) / 2) / (positives * negatives)


def quantile(values, q):
    if not values:
        return float("nan")
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (position - low)


def tpr_at(pairs, threshold):
    ai = [score for label, score in pairs if label == 1]
    return sum(1 for score in ai if score >= threshold) / len(ai) if ai else float("nan")


def report(name, rows, scores, ai_index):
    pairs = [(row["label"], scores[i][ai_index]) for i, row in enumerate(rows)]
    humans = sorted(score for label, score in pairs if label == 0)
    at1 = quantile(humans, 0.99)
    at5 = quantile(humans, 0.95)
    correct = sum(1 for label, score in pairs if (score >= at5) == (label == 1))

    domains = collections.defaultdict(lambda: [0, 0])
    for row in rows:
        domains[row["domain"]][row["label"] == 1] += 1
    per_domain = []
    for domain, (ai, human) in domains.items():
        if ai >= 10 and human >= 10:
            subset = [pair for pair, row in zip(pairs, rows) if row["domain"] == domain]
            per_domain.append((domain, auroc(subset), len(subset)))
    per_domain.sort(key=lambda item: item[1])

    return {
        "ai_index": ai_index,
        "auroc": auroc(pairs),
        "tpr1": tpr_at(pairs, at1),
        "tpr5": tpr_at(pairs, at5),
        "acc5": correct / len(pairs),
        "cutoff5": at5,
        "human_median": quantile(humans, 0.5),
        "domains": per_domain,
    }


def rates(rows, scores, ai_index, threshold):
    ai = human = ai_hit = human_hit = 0
    for row, pair in zip(rows, scores):
        score = pair[ai_index]
        if row["label"] == 1:
            ai += 1
            ai_hit += score >= threshold
        else:
            human += 1
            human_hit += score >= threshold
    return (human_hit / human if human else float("nan")), (ai_hit / ai if ai else float("nan"))


def main():
    argv = sys.argv[1:]
    sweep = "--sweep" in argv
    argv = [a for a in argv if a != "--sweep"]
    limit = None
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit") + 1])
        argv = [a for a in argv if a not in ("--limit", str(limit))]
    names = [a for a in argv if not a.startswith("--")] or list(CANDIDATES)

    sets = load_sets(["raid", "control", "hc3"], limit)
    for name, rows in sets:
        humans = sum(1 for row in rows if row["label"] == 0)
        print(f"eval/{name}.jsonl: {len(rows)} docs ({humans} human)")

    for name in names:
        spec = CANDIDATES.get(name)
        if not spec:
            continue
        started = time.perf_counter()
        try:
            logit_fn, tokenizer = build(spec)
            print(f"\n=== {name}  {spec['repo']}  load {time.perf_counter() - started:.1f}s", flush=True)

            raw = [(kind, logit_fn(text)) for kind, text in PASSAGES]
            for variant in spec["variants"]:
                for kind, logits in raw:
                    scores = variant_scores(spec, variant, logits)
                    pretty = "  ".join(f"p{i}={p * 100:5.1f}%" for i, p in enumerate(scores))
                    print(f"   [{variant}] {kind:<14} {pretty}")

            for set_name, rows in sets:
                infer_started = time.perf_counter()
                cache = []
                for i, row in enumerate(rows):
                    cache.append(logit_fn(row["text"]))
                    if (i + 1) % 100 == 0:
                        print(f"   ...{i + 1}/{len(rows)}", flush=True)
                per_doc = (time.perf_counter() - infer_started) / len(rows)
                for variant in spec["variants"]:
                    scores = [variant_scores(spec, variant, logits) for logits in cache]
                    results = [report(set_name, rows, scores, index) for index in (0, 1)]
                    best = max(results, key=lambda r: r["auroc"])
                    other = min(results, key=lambda r: r["auroc"])
                    note = "  (both look alike, verify by hand)" if abs(best["auroc"] - other["auroc"]) < 0.02 else ""
                    print(
                        f"   {set_name} [{variant}]: {per_doc * 1000:.0f}ms/doc, "
                        f"ai = score {best['ai_index']}{note}"
                    )
                    print(
                        f"     AUROC {best['auroc']:.4f}   TPR@1%FPR {best['tpr1'] * 100:.1f}%"
                        f"   TPR@5%FPR {best['tpr5'] * 100:.1f}%   acc@5%FPR {best['acc5'] * 100:.1f}%"
                        f"   thr@5%FPR {best['cutoff5']:.4f}   human median {best['human_median']:.3f}"
                    )
                    if best["domains"]:
                        weakest = "  ".join(f"{d} {a:.3f} (n={n})" for d, a, n in best["domains"][:3])
                        print(f"     weakest domains: {weakest}")
                    if sweep:
                        for threshold in (0.5, 0.7, 0.8, 0.9, 0.93, 0.95, 0.97, 0.99):
                            fpr, tpr = rates(rows, scores, best["ai_index"], threshold)
                            print(f"       thr {threshold:.2f}  TPR {tpr * 100:5.1f}%  FPR {fpr * 100:5.1f}%")
        except Exception as exc:  # noqa: BLE001
            print(f"\n=== {name}  FAILED after {time.perf_counter() - started:.1f}s: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()