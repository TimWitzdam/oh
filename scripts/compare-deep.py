"""Dev tool: head-to-head comparison of the two torch detector families.

Both are scored the way their authors prescribe: the TextSight card asks for the
logit margin (logits[AI] - logits[Human]) because its softmax saturates, and the
Qwen3 detector asks for a temperature-scaled softmax over a variable readout.

Usage: python scripts/compare-deep.py
"""

import gc
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PASSAGES = [
    ("human-blog", "I lost my job in March and honestly? Part of me was relieved. Three years of sitting in that open plan office, pretending the 8am standup meant something. Anyway, I spent the summer walking the dog way too much and finally fixed the bike."),
    ("ai-assistant", "Losing a job can be a profound opportunity for personal growth. By embracing the uncertainty and taking time to reflect on your career goals, you can emerge from this period with a clearer sense of what truly matters to you. Here are several strategies to help you navigate this transition and emerge stronger on the other side."),
    ("human-academic", "Our results diverge from those of Kowalski et al. (2019), who report no significant effect once sample size is controlled for. We attribute this discrepancy to their exclusion of the pre-2017 cohort, which comprises roughly a third of our observations."),
    ("ai-marketing", "In today's fast-paced digital landscape, having a strong online presence is no longer optional. Whether you're a small business owner or a budding entrepreneur, leveraging the power of social media can unlock unprecedented growth. In this guide, we'll walk you through five proven strategies."),
    ("human-short", "he said the door was locked but it wasn't. we went in anyway and the place smelled like wet concrete and old paper"),
    ("ai-short", "The door, though it appeared locked, was in fact ajar, and so we ventured inside, where the air carried the unmistakable scent of damp concrete and aged paper."),
    ("human-review", "Took the 7:42 into the city and it was standing room only the whole way. Some guy on the platform was playing a podcast out loud with no headphones, topically about coupons for meal kits. Got off two stops early and walked instead."),
    ("ai-essay", "Technology has fundamentally transformed the way humans communicate, work, and relate to one another. While these advancements offer remarkable opportunities, they also give rise to complex challenges that society must grapple with. It is our collective responsibility to ensure that innovation serves humanity rather than the other way around."),
]

MARKDOWN = [
    ("md-ai-list", "Here are five proven strategies for improving your workflow:\n\n"
                  "1. **Audit your current process** and map every step.\n"
                  "2. Eliminate the steps that add no value.\n"
                  "3. Automate the repetitive parts:\n   - start small\n   - measure before scaling\n"
                  "4. Review results *monthly*.\n5. Iterate on what changed."),
    ("md-human-note", "## Why I moved off the hosted thing\n\n"
                      "I tried the hosted version for about six weeks. It was fine, mostly, except the week my internet went out "
                      "and I lost a day's work to an upload bar.\n\n"
                      "- No sync means editing on the train is fine\n- Backups are my problem now\n- The config file is just a file\n\n"
                      "### What I actually miss\n\nThe shared database. Everything else I can live without."),
]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + pow(2.718281828, -x))


def bench(name: str, score_fn, cases) -> list[tuple[str, float, float]]:
    rows = []
    for kind, text in cases:
        started = time.perf_counter()
        value = score_fn(text)
        rows.append((kind, value, time.perf_counter() - started))
    warm = min(r[2] for r in rows)
    hits = sum(1 for kind, value, _ in rows if (value >= 0.5) == (not kind.startswith("human") and not kind.startswith("md-human")))
    print(f"\n{name}")
    for kind, value, ms in rows:
        print(f"  {kind:<14} {value:6.3f}   {ms * 1000:6.0f} ms")
    print(f"  separation@0.5: {hits}/{len(rows)}   fastest {warm * 1000:.0f} ms")
    return rows


def main() -> None:
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "4")))

    print("loading textsight...", flush=True)
    ts_tok = AutoTokenizer.from_pretrained("textsightai/textsight-detector-v23-custom")
    ts_model = AutoModelForSequenceClassification.from_pretrained(
        "textsightai/textsight-detector-v23-custom"
    ).eval()

    def ts_score(text: str) -> float:
        enc = ts_tok(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.inference_mode():
            logits = ts_model(**enc).logits.float()[0]
        return sigmoid((float(logits[1] - logits[0])) / 4.0)

    bench("textsight v23 (deberta-v3-large, margin/4)", ts_score, PASSAGES + MARKDOWN)

    del ts_model, ts_tok
    gc.collect()

    print("\nloading qwen3-0.6b detector...", flush=True)
    q_tok = AutoTokenizer.from_pretrained("rasbt/ai-text-detector-qwen3-0.6b-variable")
    q_model = AutoModelForSequenceClassification.from_pretrained(
        "rasbt/ai-text-detector-qwen3-0.6b-variable", dtype=torch.float32
    ).eval()

    def q_score(text: str) -> float:
        ids = q_tok(text, add_special_tokens=False, truncation=True, max_length=1023)["input_ids"]
        ids = ids + [q_tok.eos_token_id]
        inputs = {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }
        with torch.inference_mode():
            logits = q_model(**inputs).logits[0].float()
        return float(torch.softmax(logits / 1.4665638128271772, dim=-1)[1])

    bench("qwen3-0.6b variable readout (softmax/T)", q_score, PASSAGES + MARKDOWN)


if __name__ == "__main__":
    main()