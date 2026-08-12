# Data and evaluation

## JSONL schema

The loader accepts flexible Who&When-style records. A minimal record is:

```json
{
  "id": "case-001",
  "problem": "Task description",
  "ground_truth": "Optional reference answer",
  "final_answer": "Optional failed system answer",
  "failure_log": [
    {"agent": "Planner", "content": "..."},
    {"agent": "WebSurfer", "content": "..."}
  ],
  "label": {"agent": "WebSurfer", "step": 2}
}
```

Explicit step fields are preserved. Otherwise the loader generates step identifiers
starting from `data.generated_step_base` in the TOML configuration.

## Benchmarks used

| Benchmark | Cases used | Step range | Attribution labels |
|---|---:|---:|---|
| Who&When AG | 126 | 5 to 10 | named agent, exact step, reason |
| Who&When HC | 58 | 5 to 130 | canonical agent, exact step, reason |
| MP-Bench Automatic | 120 | 5 to 10 | multiple agent/step annotations |
| MP-Bench Manual | 169 | 5 to 130 | multiple agent/step annotations |
| TraceElephant compact | 218 | 5 to 94 | responsible actor, failure boundary |

The commonly reported HC-long slice contains 23 Who&When HC trajectories with more
than 50 steps. It is a repeatedly inspected post-hoc slice.

## Metrics

- **Agent accuracy:** exact canonical-agent match.
- **Exact-step accuracy:** exact global step match.
- **Step ±3 / ±5:** prediction falls within the corresponding distance.
- **MAD:** mean absolute distance between predicted and gold step.
- **MP-Bench Any:** prediction matches at least one expert annotation.
- **MP-Bench Majority / Unanimous:** prediction satisfies the corresponding human
  agreement criterion.

Joint agent-step-error accuracy is not the primary metric in this repository.

## Annotation cautions

- Who&When AG commonly uses zero-based step positions, while HC records often expose
  one-based explicit step identifiers.
- One audited HC case showed a possible `+1` mismatch between gold agent role and gold
  step; the next step matched the role.
- A failed attempt that is later corrected should not be selected as the decisive
  failure.
- TraceElephant's recovery-aware failure boundary is not automatically identical to
  Who&When's earliest decisive agent action.
- MP-Bench supports several human attributions, so deterministic top-1 evaluation is
  accompanied by Any/Majority/Unanimous metrics.

