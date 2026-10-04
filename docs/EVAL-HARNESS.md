# The eval harness, and what it changed

Written for whoever picks this up next. It records a measurement bug that
invalidated a shipped decision, the fix, and how to tell whether a running
container is actually serving your build.

## The short version

An eval set called `control` (the MAGE test split) was sampled in a way that made
it useless for comparing detectors, and it was the sole stated basis for swapping
the Deep tier and for the copy describing the Balanced tier. The set is rebuilt,
every checkpoint is re-measured through the shipped scoring path, and the numbers
now in `src/lib/catalog.ts` come from sets that are graded before use.

The Deep swap turned out to be correct anyway. The reasoning behind it was not.

## Why the old set was worthless

`sample_control()` asked the Hugging Face datasets-server for 80 human and 80
machine documents and stopped at `offset < 6000`. Two things about that split
made the result meaningless:

- It is **stored in domain blocks**, so the first ~1,200 rows are all human and
  the machine rows that immediately follow are all `cmv_machine_continuation_*`
  — small models continuing half-written Reddit comments.
- It has **no per-domain quota across the page**, and the old code capped each
  `src` at 8 documents (`per_source[src] >= 8`), which does not help when only
  one domain is reachable.

The set that came out had 80 machine documents from a single domain against 64
human documents from eight others. A detector with no weights at all:

```python
auroc([(r["label"], 1.0 if r["generator"].startswith("cmv") else 0.0) for r in rows])
# 0.9375
```

So the headline "0.99 against 0.64" was comparing a model that reads the domain
cue against one that declines to. A second bug compounded it: `domain` was taken
as `src.rsplit("_", 1)[-1]`, which yields `human`, `30b`, `gpt-3.5-trubo` —
generator names. Every "domain" then had fewer than 10 human documents, so the
per-domain AUROC column was silently empty for this set.

## What replaced it

`sample_control()` samples per `(label, domain, task)` cell and walks the whole
60,743-row split, because a cell is only reachable once the scan reaches its
block. Two naming conventions had to be handled — MAGE uses both:

```
cmv_human                        -> cmv,   human
cmv_machine_continuation_gpt-4   -> cmv,   continuation
cnn_human                        -> cnn,   human
cnn_gpt4                         -> cnn,   gpt4
cnn_human_para                   -> cnn,   human      # label is MACHINE
```

`cnn_human_para` is labelled machine: "human" names the paragraph the model was
prompted with, not the author of the output. Missing this was a bug in the first
version of the fix — it sent 96 machine documents down a fallback branch where
each distinct `src` became its own "domain". Check `split_src()` against that
list before trusting it.

Result: 517 documents, 144 human / 373 machine, 12 domains in both halves,
**domain-only AUROC 0.5483** (was 0.9375). `hswag` and `roct` ship no human
documents in this split and are reported as machine-only so their TPR is not
mistaken for a paired comparison.

The builder now grades its own output and warns above a 0.75 baseline. All three
sets currently: raid 0.500, control 0.548, hc3 0.500.

## Why this is the set to decide on

Training provenance, from the model cards. The starred columns are in-domain and
optimistic:

| tier (checkpoint) | trained on | raid* | control | hc3* |
| --- | --- | --- | --- | --- |
| Deep — desklib v1.01 | `liamdugan/raid` | 0.9849 | **0.9682** | 1.0000 |
| ex-Deep — textsight-v23 | unstated (benchmarks RAID `train_none`) | 0.8562 | 0.8154 | 1.0000 |
| Lite — chatgpt-detector-roberta (int8, shipped) | HC3, **no held-out** | 0.7384 | 0.7949 | 1.0000 |
| Balanced — qwen3-0.6b-variable | `rasbt/human-vs-ai-50k` | 0.8071 | **0.5977** | 0.9792 |

Each row is one checkpoint measured one way. Lite is the int8 ONNX export the
tier actually ships, via `eval-detectors.mjs`; its fp32 PyTorch weights read
0.7400 / 0.7862 / 1.0000, so quantisation costs nothing measurable here — but
they are a different checkpoint and the two must not be mixed in a row. An
earlier draft of this table did exactly that.

`control` is the only set clean for all four. Balanced is the only checkpoint
with no contamination anywhere, so its out-of-domain numbers are its honest ones.

## Results and decisions

**Deep stays desklib.** 0.9682 vs 0.8154 for the textsight it replaced — the swap
holds by 0.15 AUROC, not the 0.35 the broken set implied. The decision rule was
fixed before measuring: revert only if textsight ≥ desklib on the rebuilt
control set.

**Tier order was backwards.** The picker implied Lite < Balanced < Deep. Measured
Lite 0.79, Balanced 0.60, Deep 0.97 — Balanced is worse than the tier below it.

**Balanced has no usable threshold**, which is the test that ruled out TMR. Best
point anywhere on the curve is 11.3% TPR at 5.6% FPR; at the shipped 0.5 it
catches 7.8% while flagging 2.1% of human documents. Three MAGE domains are
*anti-correlated* — squad 0.229, sci_gen 0.236, tldr 0.248. It stays at 0.5
because moving it buys nothing. **It was not removed** — that is a product
decision, not a side effect of a measurement fix.

**Lite's 0.99 is confirmed against the int8 ONNX checkpoint the tier actually
ships**, not the fp32 one: 5.9% FPR on RAID, 12.5% on MAGE.

**Deep's 0.9 is confirmed**: 87.4% TPR at 4.2% FPR on control.

## Two harness bugs worth knowing about

**Label order was chosen per set.** Both eval scripts kept whichever logit index
scored the higher AUROC *independently per set*, which reports
`max(auroc, 1 - auroc)`. A candidate at 0.64 could never read below 0.5, the
figure moved when the sets changed, and a model reading the domain cue was
rewarded for it. Now decided once per candidate from all sets pooled. Reported
AUROC can legitimately be below 0.5 now.

**The sweep grid started at 0.5.** Qwen piles its human text up against 0.000,
so the old grid reported it as flagging nothing at any threshold rather than as
unmeasurable below the floor. `SWEEP_THRESHOLDS` now spans 0.01–0.99.

Also: `eval_torch.py` read desklib's 1.74 GB checkpoint twice (`from_pretrained`,
then `load_file`), building the encoder from config instead.

## The settings migration

`store.ts` used to discard any stored `threshold` equal to `0.5` as "legacy", on
every read and write. Since the slider writes `value / 100`, that made **0.5
unstorable**: the write was thrown away, the setting came back `null`, and the
control snapped to the model default with no error and no way back. On Lite
(0.99) and Deep (0.9) the user simply could not set 50%.

The migration now runs once, on read, keyed off a `version` field written into
`settings.json`. An untouched install still migrates; a deliberate `0.5` sticks.

## Proving a container is serving your build

The thresholds are **not** evidence. `0.99 / 0.5 / 0.9` shipped in `f4624ae`,
which was on `main` before any of this — the old image reports the same numbers.
Same trap for the idle-reaper in `/api/health` (`4bda554`, also already on
`main`).

Grep the served bundle for copy that differs either side of the merge:

```bash
check() {
  n=$(docker exec oh sh -c "grep -ro '$1' /app/.next/static 2>/dev/null | wc -l")
  if [ "$n" -gt 0 ]; then echo "  PRESENT ($n)  $2"; else echo "  absent        $2"; fi
}
check "steadier on unseen writing"                   # new DetectorPicker note
check "highest cut measured to keep false positives" # new SettingsPanel hint
check "weakest on unseen writing"                    # pre-merge: must be absent
check "is measured to sit around"                    # pre-merge: must be absent
```

Do **not** write this as `grep ... | head -3 || echo absent` — piping to `head`
masks grep's exit status, so the `||` never fires and the check passes whether or
not the string is there. Count instead, as above.

A second, independent signal: a successful settings write stamps
`"version": 2` into `/data/settings.json`. Nothing before this change wrote that
field.

## Reproducing

```bash
python3 scripts/fetch-eval-set.py          # ~20 min, walks the MAGE split
python3 scripts/fetch-eval-set.py --control-per-cell 0 --hc3 0   # raid only

docker build --target python-deps -t oh-pydeps .   # torch, no app build
docker run --rm -v "$PWD:/work" -v /tmp/hfcache:/hf -e HF_HOME=/hf \
  -e OMP_NUM_THREADS=6 -w /work oh-pydeps \
  /srv/venv/bin/python scripts/eval_torch.py textsight_v23 desklib qwen06_fp32 --limit 600 --sweep

HF_HOME=/tmp/hfcache-onx node scripts/eval-detectors.mjs hellosimple --sweep
```

Keep the HF cache on a mounted volume or every run re-downloads ~5 GB.
`eval/` and `data/` are gitignored; the passages belong to other people's
datasets. `--limit` slices per set with a fixed seed, so control needs ~600 to
stay whole.

`docker run --rm` **without** `-d`/`nohup` gets killed if the calling tool call is
interrupted. Do not `pgrep -f "eval-detectors.mjs"` in a wait loop — the pattern
matches the loop's own command line and it never exits.

## Loose ends

- **Deep has never run in the UI on this install.** Only Lite and Balanced are in
  `/data/models`; `desklib-deberta-v3-large` reports `installed: false`. Its
  scoring path is verified — `main.py` reproduced the harness exactly (0.9682,
  thr@5%FPR 0.8695, human median 0.081, every sweep point) — but that is the
  service, not the browser.
- **The container runs ahead of `main`.** Built from
  `fix/eval-set-and-threshold`; `main` (checked out at
  `~/Desktop/b1g programming/oh`) does not contain it. Rebuilding from there
  reverts to the pre-measurement build.
- `scripts/compare-deep.py` and `scripts/eval_textsight.py` predate
  `eval_torch.py`, are superseded by it, and are still listed in the README.
- RAID was built from the labelled **training** split, because RAID's test split
  ships without labels. In-domain for desklib and probably textsight.
- textsight's card says its weights want a unicode-normalisation pass
  (0.6087 → 0.7014 across perturbations). Neither the app nor the harness does
  it. It targets adversarial perturbations, so it should not move clean-prose
  numbers — but it would matter if a perturbation set is ever added.
