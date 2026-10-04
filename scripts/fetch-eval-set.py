"""Dev tool: build the labelled eval set the detector comparison runs on.

Two files land in `eval/`, both gitignored - the passages belong to other
people's datasets, not to this repo:

  raid.jsonl      stratified reservoir sample of RAID's `train_none.csv`, the
                  labelled no-adversarial split. This is the same source the
                  published textsight numbers were measured on, so the deep tier
                  stays comparable. RAID's own test split ships as
                  `id,generation` with the labels stripped, so the training
                  split is the only locally measurable one. Every candidate
                  trained on RAID therefore scores optimistic numbers here.
  control.jsonl   the MAGE *test* split (ACL 2024): 10 domains x 27
                  generators, labelled. Independent of RAID, but the
                  modernbert-mage candidate trained on MAGE, so treat its
                  numbers here as optimistic too. Sampled per (label, domain,
                  task) cell rather than to a flat total: the split is stored in
                  domain blocks, so a flat quota reads one block of human rows
                  and then one block of machine rows and calls it a control
                  set. `sample_control` prints a domain-only AUROC for the set
                  it just built, which is how that mistake gets caught.
  hc3.jsonl       HC3 human vs ChatGPT answers. Lite's own checkpoint was
                  fine-tuned on this, so only Lite's numbers are inflated -
                  a bias that works against the challengers, which is the
                  direction worth having. Short answers, not prose.

Usage:
    python3 scripts/fetch-eval-set.py [--human-per-domain 40] [--ai-per-group 6]
                            [--control-per-cell 8] [--hc3 200] [--seed 7]
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

csv.field_size_limit(10_000_000)

RAID_URL = "https://dataset.raid-bench.xyz/train_none.csv"
RAID_CSV = Path("data/train_none.csv")
EVAL_DIR = Path("eval")

MIN_CHARS = 400
MAX_CHARS = 20_000
ROWS_SERVER = "https://datasets-server.huggingface.co/rows"
CONTROL_DATASET = "yaful/MAGE"
HC3_DATASET = "Hello-SimpleAI/HC3"
USER_AGENT = "oh-detector-eval/1.0 (+local dev script)"
RETRY_CODES = {429, 500, 502, 503, 504}


def get_json(url: str, attempts: int = 5) -> dict:
    # The datasets server answers 422 to the stock urllib agent, and 429/5xx when
    # a script pages through it faster than it likes.
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRY_CODES or attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt * 5)
    raise RuntimeError("unreachable")


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def fingerprint(text: str) -> str:
    return hashlib.sha1(normalise(text).encode()).hexdigest()


def clip(text: str) -> str:
    text = text.strip()
    if len(text) > MAX_CHARS:
        cut = text[:MAX_CHARS]
        stop = max(cut.rfind(". "), cut.rfind("\n"))
        text = cut[: stop + 1] if stop > MAX_CHARS // 2 else cut
    return text


def make_entry(row: dict, human: bool) -> dict:
    return {
        "id": row.get("id"),
        "label": 0 if human else 1,
        "domain": row.get("domain", "?"),
        "generator": row.get("model", "?"),
        "text": clip(row.get("generation") or ""),
    }


def fetch_raid_csv() -> Path:
    if RAID_CSV.exists():
        size = RAID_CSV.stat().st_size / 1048576
        print(f"using {RAID_CSV} ({size:.0f} MB)")
        return RAID_CSV
    RAID_CSV.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {RAID_URL} -> {RAID_CSV} (765 MB)")
    tmp = RAID_CSV.with_suffix(".csv.part")
    with urllib.request.urlopen(RAID_URL) as response, tmp.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1048576:7.1f} / {total / 1048576:.0f} MB", end="")
    print()
    tmp.replace(RAID_CSV)
    return RAID_CSV


def sample_raid(human_per_domain: int, ai_per_group: int, seed: int) -> list[dict]:
    """One streaming pass over the CSV; buckets are capped reservoir samples.

    Human rows are 2.9% of the corpus, so they get their own per-domain quota:
    a threshold quoted at 5% FPR needs at least a few dozen human docs per
    domain before the number means anything.
    """
    path = fetch_raid_csv()
    rnd = random.Random(seed)
    buckets: dict[tuple, list[dict]] = collections.defaultdict(list)
    seen: dict[tuple, int] = collections.defaultdict(int)
    scanned = 0

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scanned += 1
            text = row.get("generation") or ""
            if len(text) < MIN_CHARS:
                continue
            human = row.get("model") == "human"
            key = (
                ("human", row.get("domain", "?"))
                if human
                else ("ai", row.get("domain", "?"), row.get("model", "?"))
            )
            seen[key] += 1
            quota = human_per_domain if human else ai_per_group
            bucket = buckets[key]
            # Reservoir sampling: every eligible doc gets an equal chance of
            # ending up in the sample, whatever order the CSV happens to be in.
            if len(bucket) < quota:
                bucket.append(make_entry(row, human))
            elif rnd.randrange(seen[key]) < quota:
                bucket[rnd.randrange(quota)] = make_entry(row, human)

    print(f"scanned {scanned} rows, {len(buckets)} buckets")
    return [entry for key in sorted(buckets) for entry in buckets[key]]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    humans = sum(1 for r in rows if r["label"] == 0)
    print(f"  {path}: {len(rows)} rows ({humans} human / {len(rows) - humans} ai)")


def split_src(src: str) -> tuple[str, str]:
    """Split MAGE's `src` into (domain, task).

    `src` is either `<domain>_human` or `<domain>_machine_<task>_<generator>`.
    Taking the last underscore-separated token, as this used to, yielded
    `human`, `30b` and `gpt-3.5-trubo` - generator names rather than domains -
    so every per-domain figure reported for this set was a per-generator figure
    with too few documents to clear the reporting threshold, and the set
    silently produced no domain breakdown at all.
    """
    if src.endswith("_human"):
        return src[: -len("_human")], "human"
    marker = "_machine_"
    if marker in src:
        domain, _, rest = src.partition(marker)
        return domain, rest.partition("_")[0]
    return src, "?"


def auroc(pairs: list[tuple[int, float]]) -> float:
    """Rank-based AUROC with ties averaged, so saturated scores cannot read as
    perfect separation. Duplicated from the eval scripts on purpose: this file
    has to be able to grade the set it just built without them."""
    ordered = sorted(pairs, key=lambda pair: pair[1])
    positives = sum(1 for label, _ in ordered if label == 1)
    negatives = len(ordered) - positives
    if not positives or not negatives:
        return float("nan")
    total, rank, index = 0.0, 1, 0
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


def domain_baseline(rows: list[dict]) -> tuple[float, str]:
    """AUROC of the best possible one-domain guess: 'this document came from
    domain D', maximised over D. No model, no weights, no network.

    A detector cannot be trusted on a set where this scores well, because the
    domain alone already separates the halves and the detector may only be
    reading that. It is the cheapest possible check that a labelled set is
    actually testing what it claims to test.
    """
    domains = sorted({row["domain"] for row in rows})
    best, best_domain = 0.0, "?"
    for domain in domains:
        score = auroc([(row["label"], 1.0 if row["domain"] == domain else 0.0) for row in rows])
        if score == score and score > best:  # score == score rejects NaN
            best, best_domain = score, domain
    return best, best_domain


def sample_control(per_cell: int, seed: int) -> list[dict]:
    """The MAGE *test* split (ACL 2024), labelled. `label` is 1 for human, so it
    is inverted here; eval rows use 1 for machine.

    Sampled per (label, domain, task) cell instead of to a flat total. The
    earlier version asked for 80 human and 80 machine documents and stopped at
    6,000 rows, but the split is stored in domain blocks: the first ~1,200 rows
    are human and the machine rows that follow are all ChangeMyView
    continuations from small models. The set it produced had 80 machine
    documents from a single domain against 64 human documents from eight, and a
    one-line `src.startswith("cmv")` scored AUROC 0.9375 on it. Every model
    measured against that number was mostly being measured on the layout.

    The whole split is walked, because stopping early is the whole bug: a cell
    only becomes reachable once the scan reaches its block. 60,743 rows at 100
    per request is a few hundred requests and a few minutes.
    """
    rnd = random.Random(seed)
    cells: dict[tuple[int, str, str], list[dict]] = collections.defaultdict(list)
    seen: collections.Counter[tuple[int, str, str]] = collections.Counter()
    fingerprints: set[str] = set()
    total_rows = 0
    offset = 0

    while True:
        query = urllib.parse.urlencode(
            {
                "dataset": CONTROL_DATASET,
                "config": "default",
                "split": "test",
                "offset": offset,
                "length": 100,
            }
        )
        try:
            page = get_json(f"{ROWS_SERVER}?{query}")
        except (urllib.error.URLError, ValueError) as exc:
            print(f"  control: page {offset} failed ({exc})", file=sys.stderr)
            break
        rows = page.get("rows") or []
        if not rows:
            break  # past the end of the split
        total_rows = int(page.get("num_rows_total") or total_rows)
        for item in rows:
            row = item["row"]
            text = (row.get("text") or "").strip()
            if len(text) < MIN_CHARS:
                continue
            src = str(row.get("src") or "?")
            label = 0 if str(row.get("label")) == "1" else 1
            domain, task = split_src(src)
            key = (label, domain, task)
            if fingerprint(text) in fingerprints:
                continue
            fingerprints.add(fingerprint(text))
            seen[key] += 1
            bucket = cells[key]
            # Reservoir sampling, as in sample_raid: every eligible document
            # gets an equal chance whatever order the split is stored in.
            if len(bucket) < per_cell:
                bucket.append(
                    {
                        "id": str(item["row_idx"]),
                        "label": label,
                        "domain": domain,
                        "generator": src,
                        "task": task,
                        "text": clip(text),
                    }
                )
            elif rnd.randrange(seen[key]) < per_cell:
                bucket[rnd.randrange(per_cell)] = {
                    "id": str(item["row_idx"]),
                    "label": label,
                    "domain": domain,
                    "generator": src,
                    "task": task,
                    "text": clip(text),
                }
        offset += len(rows)
        if offset % 10_000 == 0:
            print(f"  control: {offset}/{total_rows or '?'} rows, {len(cells)} cells", flush=True)

    rows = [entry for key in sorted(cells) for entry in cells[key]]
    rnd.shuffle(rows)
    return rows


def report_control(rows: list[dict]) -> None:
    """Print the shape of the control set and grade it against the cheapest
    possible detector. Run before trusting any AUROC measured on it."""
    humans = [row for row in rows if row["label"] == 0]
    machine = [row for row in rows if row["label"] == 1]
    print(f"  {len(rows)} docs: {len(humans)} human / {len(machine)} machine")
    for label, name in ((0, "human "), (1, "machine")):
        half = [row for row in rows if row["label"] == label]
        domains = collections.Counter(row["domain"] for row in half)
        tasks = collections.Counter(row["task"] for row in half)
        print(f"    {name}: {len(domains)} domains {dict(sorted(domains.items()))}")
        print(f"            tasks {dict(sorted(tasks.items()))}")

    shared = sorted({row["domain"] for row in humans} & {row["domain"] for row in machine})
    score, domain = domain_baseline(rows)
    print(f"    domains in both halves: {len(shared)} {shared}")
    print(f"    domain-only AUROC (no model, best single domain '{domain}'): {score:.4f}")
    if score > 0.75:
        print(
            f"    WARNING: a one-line domain guess scores {score:.2f} on this set. The halves are "
            "not comparable and any AUROC measured here mostly reflects the layout."
        )



def sample_hc3(target: int, seed: int, min_chars: int = 300) -> list[dict]:
    """HC3: human Reddit/ELI5 answers next to ChatGPT answers to the same
    question, as published in the raw release (two list-of-answers columns, no
    label column - the column is the label).

    It is the one set here that overlaps somebody's training data: Lite's
    checkpoint was fine-tuned on HC3, so Lite's numbers on it are optimistic.
    That bias runs against the challengers rather than for them, which is the
    useful direction - a candidate that still wins here did not win on
    memorisation. The cost is domain: short question answers, not prose.
    """
    rnd = random.Random(seed)
    seen_text: set[str] = set()
    halves: dict[int, list[dict]] = {0: [], 1: []}

    offset = 0
    while min(len(halves[0]), len(halves[1])) < target // 2 and offset < 1200:
        query = urllib.parse.urlencode(
            {
                "dataset": HC3_DATASET,
                "config": "all",
                "split": "train",
                "offset": offset,
                "length": 100,
            }
        )
        try:
            page = get_json(f"{ROWS_SERVER}?{query}")
        except (urllib.error.URLError, ValueError) as exc:
            print(f"  hc3: page {offset} failed ({exc})", file=sys.stderr)
            break
        rows = page.get("rows") or []
        if not rows:
            break
        for item in rows:
            row = item["row"]
            for label, column in ((0, "human_answers"), (1, "chatgpt_answers")):
                text = longest_answer(row.get(column))
                if len(halves[label]) >= target or len(text) < min_chars:
                    continue
                if fingerprint(text) in seen_text:
                    continue
                seen_text.add(fingerprint(text))
                halves[label].append(
                    {
                        "id": f"{item['row_idx']}-{label}",
                        "label": label,
                        "domain": str(row.get("source") or "qa"),
                        "generator": "chatgpt" if label else "human",
                        "text": clip(text),
                    }
                )
        offset += len(rows)

    print(f"  hc3 pool: {len(halves[0])} human / {len(halves[1])} ai ({offset} questions scanned)")
    rnd.shuffle(halves[0])
    rnd.shuffle(halves[1])
    half = max(1, target // 2)
    return halves[0][:half] + halves[1][: min(len(halves[1]), target - half)]


def longest_answer(raw) -> str:
    """The columns hold a stringified list of answers; take the longest one."""
    if isinstance(raw, list):
        candidates = [str(x) for x in raw]
    else:
        try:
            parsed = ast.literal_eval(str(raw))
        except (ValueError, SyntaxError):
            return str(raw or "")
        candidates = [str(x) for x in parsed] if isinstance(parsed, list) else [str(raw or "")]
    return max(candidates, key=len).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-per-domain", type=int, default=40)
    parser.add_argument("--ai-per-group", type=int, default=6)
    parser.add_argument(
        "--control-per-cell",
        type=int,
        default=8,
        help="documents per (label, domain, task) cell of the MAGE split; the "
        "total is whatever that produces, which is the point",
    )
    parser.add_argument("--hc3", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    print("raid:")
    write_jsonl(EVAL_DIR / "raid.jsonl", sample_raid(args.human_per_domain, args.ai_per_group, args.seed))
    if args.control_per_cell > 0:
        print("control (MAGE test split):")
        control = sample_control(args.control_per_cell, args.seed)
        report_control(control)
        write_jsonl(EVAL_DIR / "control.jsonl", control)
    if args.hc3 > 0:
        print("hc3 (question answers):")
        write_jsonl(EVAL_DIR / "hc3.jsonl", sample_hc3(args.hc3, args.seed))


if __name__ == "__main__":
    main()