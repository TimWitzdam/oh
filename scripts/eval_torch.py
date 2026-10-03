"""Dev tool: compare candidate CPU detectors through the same torch path the app uses.

Usage: /path/to/venv/bin/python scripts/eval_torch.py [model ...]
"""

import json
import pathlib
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))

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
    "hellosimple": {
        "repo": "Hello-SimpleAI/chatgpt-detector-roberta",
        "ai_index": 1,
        "quantize": True,
    },
    "hellosimple_fp32": {
        "repo": "Hello-SimpleAI/chatgpt-detector-roberta",
        "ai_index": 1,
        "quantize": False,
    },
    "distilbert": {
        "repo": "rasbt/ai-text-detector-distilbert",
        "ai_index": None,
        "quantize": True,
    },
    "distilbert_mica": {
        "repo": "rasbt/ai-text-detector-distilbert-mica",
        "ai_index": None,
        "quantize": True,
    },
    "e5small": {
        "repo": "MayZhou/e5-small-lora-ai-generated-detector",
        "ai_index": None,
        "quantize": True,
    },
    "qwen06": {
        "repo": "rasbt/ai-text-detector-qwen3-0.6b-variable",
        "ai_index": 1,
        "quantize": True,
        "temperature": 1.4665638128271772,
        "readout": "variable-eos",
    },
    "qwen06_fp32": {
        "repo": "rasbt/ai-text-detector-qwen3-0.6b-variable",
        "ai_index": 1,
        "quantize": False,
        "temperature": 1.4665638128271772,
        "readout": "variable-eos",
    },
}


def score(model, tokenizer, text, spec):
    if spec.get("readout") == "variable-eos":
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=1023)["input_ids"]
        ids = ids + [tokenizer.eos_token_id]
        inputs = {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.inference_mode():
        logits = model(**inputs).logits[0].float()
        if spec.get("temperature"):
            logits = logits / spec["temperature"]
        probs = torch.softmax(logits, dim=-1)
    return probs.tolist()


def main():
    names = sys.argv[1:] or list(CANDIDATES)
    for name in names:
        spec = CANDIDATES.get(name)
        if not spec:
            continue
        started = time.perf_counter()
        try:
            tokenizer = AutoTokenizer.from_pretrained(spec["repo"])
            model = AutoModelForSequenceClassification.from_pretrained(spec["repo"], dtype=torch.float32)
            model.eval()
            if spec.get("quantize"):
                model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            load_s = time.perf_counter() - started

            rows = []
            infer_started = time.perf_counter()
            for kind, text in PASSAGES:
                rows.append((kind, score(model, tokenizer, text, spec)))
            per = (time.perf_counter() - infer_started) / len(PASSAGES)

            print(f"\n{name}  {spec['repo']}  load {load_s:.1f}s  {per * 1000:.0f}ms/passage")
            for kind, probs in rows:
                pretty = "  ".join(f"p{i}={p * 100:5.1f}%" for i, p in enumerate(probs))
                print(f"   {kind:<14} {pretty}")
            for ai_index in {0, 1, spec["ai_index"]}:
                hits = sum(
                    1
                    for (kind, probs), in zip(
                        [(k, p) for k, p in rows]
                    )
                    if (probs[ai_index] >= 0.5) == (not kind.startswith("human"))
                )
                print(f"   -> ai_index {ai_index}: {hits}/{len(PASSAGES)} correct")
        except Exception as exc:  # noqa: BLE001
            print(f"\n{name}  FAILED after {time.perf_counter() - started:.1f}s: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()