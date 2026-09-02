"""evalaware: a causal decomposition of evaluation awareness in open-weight LMs.

Pipeline stages (see evalaware/experiments/):
  exp0  build the contrastive corpus and behavior suites
  exp1  cache activations; linear probes by layer & position; directions
  exp2  Experiment A - causal control of the model's verbalized belief
        (activation patching + steering, layer sweep)
  exp3  Experiment B - downstream behavior under intervention
        (sandbagging MC, sycophancy, over-refusal, self-report)
  exp4  Experiment C - stage localization & dissociation analysis
  exp5  Experiment D - token-position construction map
  exp6  Experiment E - E1 attribution patching -> E2 causal verification
        -> E3 connection of components to the direction d
"""

__version__ = "0.1.0"
