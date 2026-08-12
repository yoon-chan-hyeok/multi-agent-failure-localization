# Reproducibility notes

## What is included

- runnable experiment driver;
- model-backend abstraction;
- prompt and output parsers;
- strict agent/step metrics;
- deterministic smoke data and mock backend;
- aggregate result tables from completed experiments;
- utilities for paired significance, bootstrap confidence intervals, A2P prompt
  auditing, and MP-Bench multi-annotation evaluation.

## What is not included

- paid API keys;
- full benchmark data;
- multi-gigabyte raw prediction directories;
- model checkpoints;
- Who&When Pro results, because a usable public release was not available during the
  project;
- native TraceElephant full-observability benchmark results.

## Main run controls

- GPT-4o TSR-Loc used temperature `0.0`.
- The A2P repository-exact condition left the temperature request field unset to
  preserve the audited repository request semantics.
- The A2P audit targeted commit
  `7953d780c85054721a7b4bf246bcf60a16bb28af`.
- The same strict local evaluator was used for the main Who&When comparison.
- Requirement lists were frozen before localization in the canonical TSR-Loc runs.
- Errors can be recorded with `--continue-on-error`; sharded runs were merged only
  after failed offsets were rerun.

## Statistical interpretation

- Direct vs No-GT TSR-Loc exact-step McNemar: `p = 5.77e-12`.
- No-GT TSR-Loc vs A2P repo-exact: `p = 0.2954`; do not claim significant
  superiority.
- Matched requirement-block controls were directionally positive but not significant.
- Human evaluation covered 337 generated requirements: 97.63% and 96.74% validity by
  two raters, Cohen's kappa `0.838`.

## Execution safeguards

Long traces and hosted-model failures required process-level sharding, resumable
execution, null-case repair, input caps for local models and explicit output-path
handling. Failed offsets were rerun before shards were merged.

## Claim boundaries

Do not use this artifact to claim:

- universal or benchmark-wide SOTA;
- statistically significant superiority over A2P;
- lower token cost than all-at-once;
- formal verification or intervention-verified causality;
- an official AgentRx reproduction;
- a universal benefit from chunking.

