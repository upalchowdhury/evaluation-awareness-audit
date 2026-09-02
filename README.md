# Does the model know it is being tested?

Reading, asking, and steering "evaluation awareness" in five small open
language models (Qwen2.5-0.5B/1.5B/7B-Instruct, Phi-3-mini-4k-instruct,
Olmo-3-7B-Instruct), on one controlled corpus, with the same held-out items
for every instrument. Runs locally on a 96 GB Apple-silicon Mac with
TransformerLens.

The submission write-up is `paper/main_v2.tex`. The research critique that
led to it, including the instrument failure it fixes, is `paper/CRITIQUE.md`.

## What is measured

| instrument | what it is | where |
|---|---|---|
| reading | linear probe / mass-mean direction `d` on the residual stream at the last task token, every layer | exp1 |
| asking | the model's own Yes/No answer to "is this a test?", asked in a **second user turn** after a placeholder assistant turn, reply prefilled with "Answer:" (two polarities averaged) | `readouts.belief_prompt` |
| steering | `d` added at one layer, locked layer selection, dose response, 100 matched-norm random directions | exp2, exp8 |
| patching | source residuals copied into a destination prompt over the shared suffix; full source 2x2 (class x content), positional split, distance matching | exp2, exp7, exp10, exp11 |
| formality | in-corpus style direction and a neutral formality direction with no eval/deploy vocabulary | exp8 |
| behaviour | sandbagging 2x2 (framing x incentive), sycophancy, refusal, self-report | exp3 |

## Why there is a v2

The first readout appended the belief question to the task turn. Qwen-7B,
Olmo and Phi-3 obeyed the task's "respond with only the letter" and never
answered it (`scripts/readout_validity.py`); Phi-3's tokenizer also put the
bare-space token in both the Yes and the No group. Every cross-model causal
number from that readout was invalid. Five formats were tested
(`scripts/readout_variants.py`); the second-turn format is answered by all
five models. Results with the old readout stay in `results/`; the re-run with
the new readout lands in `results_v2/`. Never mix them in one table.

## Run

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt && uv pip install -e .
export PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

python -u run.py --config configs/smoke.yaml --exps 0,1,2,3,7,8,10,11     # ~3 min wiring check

# v2 queue, one model at a time (~12 h for all five)
EXPS=0,1,2,7,8,10,11 scripts/seq_v2.sh qwen15b qwen05b qwen7b olmo7b phi3mini
EXPS=3               scripts/seq_v2.sh qwen15b qwen05b qwen7b olmo7b phi3mini

# paper
python scripts/make_paper_assets.py    --results results_v2 --model qwen15b --compare qwen05b,qwen15b,qwen7b,phi3mini,olmo7b
python scripts/make_paper_assets_v2.py --results results_v2 --model qwen15b --compare qwen05b,qwen15b,qwen7b,phi3mini,olmo7b
python scripts/regen_paper_figs_v2.py  --results results_v2 --model qwen15b
cd paper && tectonic -X compile main_v2.tex --outdir .
```

`tectonic` (`brew install tectonic`) runs the bibtex passes itself.

## Verify it yourself

```bash
python scripts/readout_validity.py qwen15b            # does the model answer Yes/No at all?
python scripts/sample_audit.py --results results_v2 --model qwen15b --n 8 --seed 1
python scripts/recompute_headline.py --results results_v2 --model qwen15b
```

## Repo layout

```
configs/            default + smoke + v2 overlay + per-model overlays
evalaware/
  data/             cue families, contents, behaviours, corpus builder
  model_io.py       loading, chat templates (incl. the two-turn belief prompt), logit utils
  readouts.py       belief (two polarities), validity check, MC accuracy, sycophancy, refusal
  probes.py         grouped-CV probes, mass-mean directions, permutation test
  interventions.py  steer / ablate / end-aligned patch primitives
  stats.py          cluster bootstrap, Holm, Cohen's d
  experiments/      exp0..exp11 drivers + run_all
paper/              main_v2.tex (submission), main.tex (v1 estimand paper), CRITIQUE.md, DISCLOSURE_TODO.md
scripts/            asset builders, figure builders, verification scripts, queue scripts
results/            v1 (old readout)      results_v2/   v2 (second-turn readout)
```

MIT licence. The corpus and behaviour items are original and synthetic.
