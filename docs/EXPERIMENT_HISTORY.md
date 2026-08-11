# Experiment history

## Initial local Llama baseline, Who&When HC 58

| Method | Agent | Exact step |
|---|---:|---:|
| All-at-once | 18.97% | 12.07% |
| Step-by-step | 29.31% | 1.72% |
| Binary search | 0.00% | 0.00% |
| Paper hybrid | 18.97% | 5.17% |
| Chunk vote 10 | 51.72% | 13.79% |
| Paper hybrid 10 | 55.17% | 12.07% |
| CCV10 | 41.38% | 8.62% |
| MVBS10 | 39.66% | 5.17% |

These experiments established that chunk-level methods could improve agent selection
under a constrained local model, but they did not establish a universal exact-step
gain.

## GPT-4o fixed ten-chunk conditions, Who&When 184

| Method | Agent | Exact step |
|---|---:|---:|
| Chunk vote 10 | 39.13% | 15.22% |
| Paper hybrid 10 | 39.13% | 16.85% |
| CCV10 | 45.11% | 25.54% |
| CCV beam 10 | 44.57% | 25.00% |

## Controlled HC-long chunk experiments

With top-k fixed at three, target chunk budgets of 500, 1,000, 1,500, 2,000,
2,500, and 3,000 proxy tokens produced non-monotonic exact-step performance.
Varying top-k from three through eight was also non-monotonic. Step-boundary-preserving
runs did not eliminate this instability.

This negative result changed the main research direction: chunking remained an
implemented exploratory tool, while the final method focused on pre-trace task
interpretation and full-trace localization.

