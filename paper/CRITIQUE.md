# Research critique of the evaluation-awareness project (1 September 2026)

This is a critique of the project as research, written for the author before a
MATS submission. It is deliberately blunt. The short version comes first, then
the evidence, then what was fixed, what is re-running, and what only the author
can do.

## The short version

**The project is stronger than the paper, but one instrument problem invalidates
half of the paper's headline table, and it had to be fixed before anything else.**

1. **The belief question was never answered by three of the five models.** The
   whole paper measures "what the model says about being evaluated" by appending
   a Yes/No question after the task and reading the first answer token. Only the
   two smallest Qwen models answer it. Qwen-7B and Olmo answer the multiple-choice
   task instead in 65 to 85 percent of prompts. Phi-3 never answers it (Yes+No
   probability mass 0.3 percent). So every "causal" row for those three models,
   including the paper's five-model "landscape" claim and Phi-3's "readable but
   not steerable" story, was measured on tokens the model never produces.
   On top of that, Phi-3's tokenizer put the bare-space token in both the Yes and
   the No group, which cancels the score by construction. **This is fixed** (new
   readout format tested on all five models; tokenizer bug fixed) and the affected
   experiments are re-running into `results_v2/`.
2. **Every number in the current PDF reproduces exactly from the raw files.** An
   independent re-derivation of about 90 values found zero arithmetic or
   transcription errors. The problems are in what was measured and how it is
   framed, not in the arithmetic.
3. **The paper leads with its weakest contribution.** "The paired-minus-unpaired
   control cannot estimate a class effect" is true and worth one section, but it
   is textbook bad-control bias (Cinelli, Forney and Pearl 2022), the paper shows
   nobody in the literature made the mistake, and the title advertises a
   confession. The strongest simple findings are elsewhere (below).
4. **No effect is calibrated.** The paper never reports the model's own natural
   eval-versus-deploy difference in the readout, so a reader cannot tell whether
   a 0.88-logit patch effect is a full flip or a rounding error. Computed from
   the existing files, the intervention effects are about the same size as the
   natural gap on every model, and the "scaling with model size" claim disappears
   once effects are expressed in those units. This is now computed by the
   pipeline (`natural_gap` in exp1) and the rewrite reports everything against it.
5. **It reads as machine-hardened rather than written.** Bolded epithet on every
   paragraph, 240 numbers in eight pages, 56 confidence intervals in running
   prose, no example prompt, no raw data figure, no introduction, no "what next",
   and the disclosure section that would prove a human did the work is a blank
   form. A plain-language rewrite is being produced from the re-run.

## 1. How the critique was done

- A fresh-eyes reviewer with no prior involvement read the paper as a MATS
  application reviewer.
- A code audit read every intervention, readout, and experiment module against
  what the paper claims they do.
- An independent recomputation re-derived every headline number and table
  from the raw per-item CSVs with freshly written code, and stress-tested the
  statistics.
- I loaded all five models myself and checked the readout instrument directly
  (`scripts/readout_validity.py`, `scripts/readout_variants.py`).

## 2. The blocker in detail

Validity of the v1 belief readout (belief question appended inside the task
turn; 20 eval + 20 deploy held-out prompts per model):

| model | P(Yes)+P(No) at the answer position | greedy first token is Yes/No | what it emits instead |
|---|---|---|---|
| Qwen2.5-0.5B | 1.00 | 100% | — |
| Qwen2.5-1.5B (primary) | 0.88 to 0.97 | 90 to 100% | "No" to both polarities |
| Qwen2.5-7B | 0.17 to 0.34 | 15 to 35% | the MC letter |
| Olmo-3-7B | 0.28 to 0.48 | 25 to 55% | the MC letter |
| Phi-3-mini | 0.003 to 0.025 | 0 to 5% | the MC letter, "Ice", "The" |

Cause: the task text ends with "Respond with only the letter of the correct
answer", and the larger instruction-tuned models obey that instruction rather
than the question appended below it. On open-ended items the same models answer
the belief question fine, which confirms the mechanism.

Even on the primary model the readout means less than the paper implies: the
model's greedy answer is "No" to *both* "is this a test?" and "is this a real
user?" in 18 to 20 of 20 prompts. The "verbalized belief" is a graded shift in
how strongly the model says No, not a report of being evaluated. The paper
never says this.

Five alternative formats were tested on all five models. The only one every
model answers is: task in a user turn, a fixed placeholder assistant turn,
the belief question in a second user turn, and the assistant reply prefilled
with "Answer:". Under it the greedy token is Yes/No in 95 to 100 percent of
prompts on every model, and the two-polarity score separates eval from deploy
prompts with AUC 0.82 to 1.00 on 40 prompts. Qwen-7B now actually says "Yes,
this is a test" under eval framing. This format is the v2 readout.

## 3. Every other problem found, with status

Severity: **B** blocker, **M** major, **m** minor. Status: fixed / re-running /
withdrawn / to write / user-only.

| # | Problem | Sev | Status |
|---|---|---|---|
| 1 | Belief question not answered by 7B, Olmo, Phi-3 (above) | B | fixed, re-running |
| 2 | Phi-3 bare-space token (id 29871) in both Yes and No groups, and in all four MC letter groups; digit groups identical for all nine digits so the self-report readout was constant 5.0 on Phi-3 (sd 0.000) | B | fixed (whitespace pieces filtered, overlap removed); self-report claim withdrawn |
| 3 | Paper says the readout is "two-polarity"; every patching experiment (exp2, 5, 6, 7, 10, 11) uses polarity A only | M | done: disclosed in the rewrite's Setup |
| 4 | Four `subtle_exam_format` templates put text after the task, so 9 of 60 patching pairs shared only the belief question and no task tokens; patched span varied from 51 to 134 tokens | M | fixed: such renderings are excluded from all patching pairs |
| 5 | The distance-matched "pre-question span" collapsed to a single glue token on 98 percent of destinations on four of five models, yet the paper reports it as evidence | M | withdrawn from the paper |
| 6 | Interaction contrast was on twice the scale of the main effects, and was compared to them directly | m | fixed (halved) in exp10, exp11, asset script, equivalence script |
| 7 | exp7 wrote the class contrast into a column named `content`, with a silent fallback to paired-minus-unpaired, the very estimator the paper says is invalid | M | fixed: renamed `class_samecontent`; fallback now raises |
| 8 | No effect calibrated against the natural eval-vs-deploy gap of the readout | M | fixed in pipeline (`natural_gap`); rewrite reports effects in those units |
| 9 | Probe AUC after one layer is already 0.91 to 0.93 on every model, so the encoding result is nearly surface-decodable; the paper presents it as answering the format-sensitivity critique | M | done: reframed (Finding 1) |
| 10 | Style direction effect at 1.5B sits at the 83rd percentile of the random null (not distinguishable); the paper claims 100 random directions but Phi-3 and Olmo used 6 | M | re-running with 100 on all models; plus a neutral register direction not built from the deploy families |
| 11 | 119 confidence intervals reported, none corrected; `holm()` exists in `stats.py` and is never called; 4 of 14 "significant" positional cells fail Holm, including Olmo's sign-inverted final cell that a paragraph is built on | M | done: Holm markers in the positional table |
| 12 | Percentile bootstrap CIs under-cover (about 92 percent at n=30 by simulation); locked estimates use 15 clusters | m | done: stated in Limitations |
| 13 | Pooled sandbagging gap gives Qwen-7B 87 percent of the weight; the pooled CI excludes the 0.5B point estimate | m | withdrawn; report per-model |
| 14 | `sandbag_gap` is the diagonal contrast deploy/neutral minus eval/incentive, not a main effect (all decompositions are null anyway) | m | done: framing and incentive main effects reported |
| 15 | The all-layer ablation (F3) is "unstable" with no diagnosis; the random control moves the outcome 4x more than the treatment | m | withdrawn from claims; hypothesis stated in the rewrite (generic damage collapses the readout's range) |
| 16 | Steering effect at 1.5B is a single-layer spike (layer 10; layers 9 and 11 near zero) while 7B shows a smooth band; the direction changes smoothly across layers (cos 0.6 to 0.7 between neighbours), so the spike is a property of the layer, not the vector | m | resolved: under the valid readout the 1.5B profile is a mid-network band, not a spike |
| 17 | Dose figure shows 3 random directions; the text beside it cites the 100-direction null | m | done: captioned honestly |
| 18 | Manifests recorded the config string "auto" instead of the real dtype | m | fixed |
| 19 | Patch hooks with span 1 would fire on KV-cached generation steps (latent) | m | fixed |
| 20 | `bootstrap_mean_ci` crashed on NaN with clusters (latent) | m | fixed |
| 21 | Estimand framing: does not cite bad-control literature (Cinelli et al.), interchange-intervention work (Geiger et al., Vig et al.), or Mueller 2024; claims to be "orthogonal" to Zhang and Nanda 2024 when it is a special case of "choose the corruption deliberately" | M | done: cited in Related work |
| 22 | "Class transfer sits at the final pre-answer position at deep layers" is close to definitional (that residual is what the answer is read from); the interesting null upstream is confounded because the destination's cue tokens are never patched | M | done: stated as a limitation; first continuation item |
| 23 | No example prompt, no belief question text, no raw-data figure, no introduction, no "what I would do next" | M | done: all in the rewrite |
| 24 | Disclosure section is a blank form; author and repository are `[TO FILL]`; no version control, so "pre-specified" and "effect-blind" are self-attested | B for the application | user-only |

## 4. What is solid and should be kept

- All 90-odd headline values reproduce exactly from the raw CSVs, including
  the locked estimates, the 2x2 effects, the positional split, the matched arms,
  and the sandbagging cells.
- The class effect at the final position is not driven by outliers: 27 of 30
  content clusters have the expected sign in both directions, mean and median
  agree, and trimming strengthens it. The three wrong-sign clusters are the
  same items in both directions (a real item property, not noise).
- The discovery/estimation split is genuinely leak-free.
- The 2x2 completion is a correct and useful methods point. It just is not a
  headline.
- The probe controls (no-lexicon family, topic controls, cross-family
  transfer) are well designed even though the layer-0 result limits what
  they show.
- The readout scoring itself (two polarities, logsumexp over surface variants)
  is good practice; it only needed the format change and the tokenizer fix.

## 5. What is re-running and how the paper will be rebuilt

Re-running (`scripts/seq_v2.sh`, results in `results_v2/`): exp0 corpus, exp1
probes plus readout validity and natural gap, exp2 patch/steer sweeps with the
locked split, exp7 source 2x2, exp8 register decomposition with the neutral
style direction and 100 random directions on every model, exp10 positional 2x2,
exp11 distance-matched 2x2, then exp3 behaviour. Order: Qwen-1.5B, Qwen-0.5B,
Qwen-7B, Olmo-3-7B, Phi-3-mini. The 7B sweep now uses layer stride 2.

The rewrite (`paper/main_v2.tex`) is organised around the findings a
non-specialist can hold in one sentence each:

1. Every model's activations encode eval-versus-deploy almost perfectly, but
   after one layer a linear probe is already at 0.91 to 0.93, so this is mostly
   a surface property of the prompts.
2. Whether the model *says* it is being evaluated depends on how you ask.
   With the question appended to the task, three of five models never answer.
   Asked in a separate turn, all five answer, and (results pending) the report
   tracks the internal encoding to a degree that differs by model.
3. The report can be steered along the eval direction (results pending), and
   the effect must be read against two yardsticks the old paper omitted: the
   model's own natural gap and the matched-norm random-direction null.
4. A formality direction is heavily entangled with the eval direction
   (cosine 0.46 to 0.68 in all five models in v1), and a neutral formality
   direction built with no eval/deploy vocabulary now tests whether that
   entanglement survives.
5. Patching shows that what transfers between prompts is the answer state at
   the readout, not the task tokens, and the way that was discovered (the
   control that matched the effect, then the 2x2) is told as the research
   story rather than as the title.
6. None of it reaches behaviour on these tasks (short, with the caveat that
   trivia questions give the models no reason to sandbag).

Every number in the rewrite comes from a macro regenerated by
`scripts/make_paper_assets.py --results results_v2`, and the text carries the
calibration columns (natural gap, random p95, Holm-corrected flags).

## 6. What only the author can do

1. Fill the disclosure section honestly: which designs were yours, which code
   you read end to end, how many prompts you inspected, which numbers you
   recomputed (`scripts/recompute_headline.py`), what you did not check.
   `paper/DISCLOSURE_TODO.md` lists the steps; budget about two hours.
2. Put the repository under version control and commit before reading the v2
   results, so "pre-specified" means something a reader can check.
3. Read `evalaware/model_io.py` (`chat_wrap_belief`), `evalaware/readouts.py`,
   and one patching experiment end to end. They are about 400 lines together
   and every claim in the paper depends on them.
4. Decide whether the neutral register direction's wording is acceptable to you
   (`REGISTER_FORMAL` and `REGISTER_CASUAL` in `evalaware/data/templates.py`);
   it was written today and has not been reviewed by anyone else.

## 7. How the project continues after the submission

- **Patch the cue, not only the suffix.** The upstream null is only interpretable
  once the destination's own cue tokens are patched or attention from the
  readout back to the cue is knocked out. This is the first mechanistic
  experiment worth running.
- **Which heads write the answer state?** The archived exp6 attribution results
  (MLP15, L17H0, MLP18; partial mediation) are the "mech" in mech interp and
  were cut from the paper. Re-run them under the v2 readout.
- **More nuisance axes.** Politeness, length, presence of a system prompt,
  imperative mood: each is a candidate for the same entanglement test as
  formality.
- **Tasks with a reason to sandbag.** Trivia gives the models nothing to hide;
  dangerous-capability style tasks or explicit incentives with consequences do.
- **Survey the literature's controls.** Coding 15 to 25 published patching
  papers on what their control source varies would turn the estimand point from
  an anecdote into a finding.
- **Bigger models.** The 14B config exists; the v2 readout was designed so that
  larger instruction-tuned models answer it.

## 8. What the re-run found (2 September 2026)

All five models completed the core experiments under the new readout
(`scripts/report_v2.py` prints the digest; `paper/main_v2.tex` is built from
it). The picture changed in four places and held in three.

| model | answers Yes/No (old → new) | verbal AUC (old → new) | eval direction vs 100 random directions | locked steering, eval prompts (units of natural gap) |
|---|---|---|---|---|
| Qwen-0.5B | 100% → 100% | 0.84 → 0.82 | percentile 81 | −0.63 logits (3.2 gaps; gap only 0.20) |
| Qwen-1.5B | 90–100% → 100% | 0.78 → 0.94 | percentile 88 | −0.41 (0.56) |
| Qwen-7B | 15–35% → 100% | 0.89 → 0.94 | percentile 99 | −3.05 (0.76) |
| Phi-3-mini | 0–5% → 100% | 0.70 → 0.90 | percentile 88, sign-inverted | +0.34 (0.85), wrong sign |
| Olmo-3-7B | 25–55% → 100% | 0.64 → 0.85 | percentile 93 | −0.75 (0.63) |

**Changed.**
- Phi-3 is not "readable but not steerable". Its readout was dead. Under the
  valid readout it is the one model whose evaluation direction steers the
  answer the wrong way at the locked layer, with tight intervals and a sign
  that alternates across layers. The "instruments disagree on the same model"
  observation survives, but on Phi-3, not Olmo.
- Olmo's sign-inverted steering is gone. It was an artefact of the
  unanswered readout. Its locked steering is normal-signed.
- "Steering is selective" holds on one model. Only Qwen-7B's direction beats
  the 95th percentile of 100 random directions of the same size; the other
  four sit between percentiles 81 and 93. The old paper's 98th to 100th
  percentiles on the small Qwens were measured with a readout that was
  answered but with a "No"-to-everything bias.
- The formality finding is now clean and is the paper's best result: on
  Qwen-7B a formality direction built with no evaluation or deployment
  wording moves the "am I being tested?" answer more than the evaluation
  direction itself (6.3 versus 5.4 logits across the dose range; 100th
  percentile of the random null), and removing it from the evaluation
  direction leaves 65 percent of the effect. On the four other models the
  neutral formality direction is inside the random null.

**Held.**
- The unrelated-prompt control reproduces the matched patch curve on every
  model; the four-cell repair recovers the class effect everywhere; the task
  tokens carry nothing (at most 0.05 logits on any model); the effect sits at
  the final position on the four Qwen/Phi models and on the question tokens
  on Olmo.
- The probe is shallow everywhere: after one layer the held-out AUC is 0.82
  to 0.93.
- The stated answer tracks the probe better than before (verbal AUC 0.82 to
  0.94; prompt-level rank correlation 0.38 to 0.68), which is what a valid
  readout should do.
- Behaviour: the framing effect on accuracy is null on four models; on
  Olmo-3-7B accuracy is 0.07 lower under test framing [0.02, 0.14]. Steering
  the stated answer by 0.35 logits on the primary model changes accuracy by
  0.001.

**Positional cells:** 30; 20 exclude zero uncorrected; 19 survive Holm.

**Fact audit of the rewrite (2 September):** an independent agent checked
every hand-typed number and universal in `main_v2.tex` against the result
files; 21 findings, all prose-level (one real inversion: the topic-control
sentence had its label flipped), all fixed. The two verdict sentences and
the behaviour claims are generated from macros.
