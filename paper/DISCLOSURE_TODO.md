# Before you submit: what only you can fill in

The paper has red `[TO FILL: ...]` markers. They are deliberately impossible
for an assistant to complete honestly. Everything below is the concrete,
fast path to filling them from genuine personal verification (budget: ~2 h).

## 1. Author name and repository link (title page)

Push the repo (without `results/*/exp*/` if size is a problem — keep the
`summary.json` files and `paper/`) and paste the URL into the `\thanks{}`
footnote. Replace `[TO FILL: author name]`.

## 2. Personal verification — do these, then write one line each

```bash
cd MATS/research-eval-circuit/direction-to-circuit && source .venv/bin/activate

# (a) Look at raw prompts and their recorded scores. Do 8; note anything odd.
python scripts/sample_audit.py --model qwen15b --n 8 --seed 1

# (b) Recompute the headline numbers yourself, from per-item CSVs only.
#     Compare to Table 1 / the executive summary. They should match to rounding.
python scripts/recompute_headline.py --model qwen15b
python scripts/recompute_headline.py --model qwen7b

# (c) Read the two code paths the claims depend on (about 250 lines total):
#     - evalaware/interventions.py      : what "patching" literally does
#     - evalaware/experiments/exp10_span_directions.py : the positional 2x2
#     Check for yourself that `patch_end_range_hook` patches [-a, -b) from the
#     end, and that the `final` region is (1, 0) = the last input position.
```

Then fill the section-8 fields in your own words:

- **Designed yourself**: which of these were your calls — the 2×2 completion,
  the locked split, the positional split, the register decomposition, the
  100-direction null? Be specific about which were suggested by critiques.
- **Code paths read end to end**: name the files.
- **Prompts inspected by hand**: the number, and whether alignments/scores
  looked right (say so if something looked wrong and what you did).
- **Numbers independently recomputed**: list which (from step b).
- **What remains unchecked**: e.g. the Olmo/Phi-3 runs, the attribution code,
  the register direction construction — whatever you did not personally read.
- **Surprise-at-error, one line per component** (corpus, patching, steering,
  statistics): "very surprised / somewhat / not surprised" and why.
- **Total active hours** and a 5–8 line decision timeline (dates: when the
  control matched the effect, when the 2×2 was run, when the positional
  split reversed the interpretation, when the attrition bug was found).
- **Other contributors**: "none" if none.

## 3. LLM-use disclosure — already drafted, check it is accurate

Section 8 currently states: Claude (code, orchestration, drafting, audits)
and ChatGPT/Codex (the written critiques whose points this version addresses).
If any other tool was used, add it.

## 4. Final regeneration (after any results change) -- v2 paper

The submission paper is now `paper/main_v2.tex`, built from `results_v2/`
(the re-run with the second-turn belief readout; see paper/CRITIQUE.md).

```bash
python scripts/make_paper_assets.py    --results results_v2 --model qwen15b --compare qwen05b,qwen15b,qwen7b,phi3mini,olmo7b
python scripts/make_paper_assets_v2.py --results results_v2 --model qwen15b --compare qwen05b,qwen15b,qwen7b,phi3mini,olmo7b
python scripts/regen_paper_figs_v2.py  --results results_v2 --model qwen15b
cd paper && tectonic -X compile main_v2.tex --outdir .
grep -n "TO FILL\|TODO" main_v2.tex numbers.tex numbers_v2.tex   # must print nothing
```

Two `\todo{}` verdict sentences in main_v2.tex (steering across models;
formality across models) are to be written FROM the generated tables once all
five models have finished; do not type numbers into prose.

## 5. Two more things only you can do

* `git init && git add -A && git commit -m "v2 readout + audit fixes"` before
  reading the v2 results, so "pre-specified" is checkable.
* Read and approve the wording of `REGISTER_FORMAL` / `REGISTER_CASUAL` in
  `evalaware/data/templates.py` (written 2026-09-01, unreviewed).
