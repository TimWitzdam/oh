"""Dev tool: evaluate the TextSight v23 detector and compare scoring modes.

The model card is explicit that the softmax probability saturates and that the
logit margin is the useful quantity, so both are reported here.

Usage: python scripts/eval_textsight.py
"""

import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = "textsightai/textsight-detector-v23-custom"

PLAIN = [
    ("human-blog", "I lost my job in March and honestly? Part of me was relieved. Three years of sitting in that open plan office, pretending the 8am standup meant something. Anyway, I spent the summer walking the dog way too much and finally fixed the bike."),
    ("ai-assistant", "Losing a job can be a profound opportunity for personal growth. By embracing the uncertainty and taking time to reflect on your career goals, you can emerge from this period with a clearer sense of what truly matters to you. Here are several strategies to help you navigate this transition and emerge stronger on the other side."),
    ("human-academic", "Our results diverge from those of Kowalski et al. (2019), who report no significant effect once sample size is controlled for. We attribute this discrepancy to their exclusion of the pre-2017 cohort, which comprises roughly a third of our observations. Including that cohort, the effect size rises to 0.31."),
    ("ai-marketing", "In today's fast-paced digital landscape, having a strong online presence is no longer optional. Whether you're a small business owner or a budding entrepreneur, leveraging the power of social media can unlock unprecedented growth. In this guide, we'll walk you through five proven strategies."),
    ("human-short", "he said the door was locked but it wasn't. we went in anyway and the place smelled like wet concrete and old paper"),
    ("ai-short", "The door, though it appeared locked, was in fact ajar, and so we ventured inside, where the air carried the unmistakable scent of damp concrete and aged paper."),
]

MARKDOWN_AI = """## Why Local-First Tools Matter

Local-first software keeps your data **on your own machine**. This is not a
nice-to-have; it is the whole point.

- No upload latency
- Works on a plane
- Nobody else can read it

### Getting Started

1. Install the package
2. Run the server
3. Open your browser

> Offline is a feature, not a fallback.
"""

MARKDOWN_HUMAN = """## Why I moved off the hosted thing

I tried the hosted version for about six weeks. It was fine, mostly, except the
week my internet went out and I lost a day's work to an upload bar.

- No sync means editing on the train is fine
- Backups are my problem now, which is its own problem
- The config file is just a file

### What I actually miss

The shared database. Everything else I can live without.
"""


def main() -> None:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()
    print(f"loaded in {time.perf_counter() - started:.1f}s")

    def margin(text: str) -> float:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.inference_mode():
            logits = model(**enc).logits.float()[0]
        return float(logits[1] - logits[0])

    started = time.perf_counter()
    rows = [(kind, margin(text)) for kind, text in PLAIN]
    per = (time.perf_counter() - started) / len(PLAIN)
    print(f"\nplain passages ({per * 1000:.0f} ms each, margin = logits[AI] - logits[Human])")
    for kind, value in rows:
        print(f"  {kind:<14} margin {value:+8.2f}   sigmoid(m/4) {1 / (1 + pow(2.718281828, -value / 4)):.3f}")
    for cut in (0.0, 2.0):
        hits = sum(1 for kind, v in rows if (v > cut) == (not kind.startswith("human")))
        print(f"  threshold {cut:+.1f}: {hits}/{len(rows)} correct")

    print("\nmarkdown formatting")
    for name, text in (("markdown-ai", MARKDOWN_AI), ("markdown-human", MARKDOWN_HUMAN)):
        raw = margin(text)
        stripped = margin(strip_markdown(text))
        print(f"  {name:<15} raw {raw:+8.2f}   stripped {stripped:+8.2f}   delta {stripped - raw:+.2f}")


def strip_markdown(text: str) -> str:
    import re

    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s*([-*_]\s*){3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


if __name__ == "__main__":
    main()