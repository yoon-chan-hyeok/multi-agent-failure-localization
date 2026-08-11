# Verified aggregate results

These CSV files contain manually curated aggregate values from completed experiment
artifacts. They are presentation tables, not substitutes for case-level predictions.

| File | Contents |
|---|---|
| `who_and_when_main.csv` | Main GPT-4o comparison on all 184 Who&When cases and the exploratory HC-long slice |
| `model_factorial_2x2.csv` | Llama/GPT requirement-compiler × localizer experiment |
| `external_transfer.csv` | MP-Bench Manual, Automatic, and MAST-source evaluation |

## Interpretation rules

- Accuracy values are fractions in the CSV files and percentages in the README.
- Empty cells indicate values not retained in the final verified result ledger.
- `HC-long-posthoc` denotes the repeatedly inspected 23-case subset with more than
  50 steps.
- `A2P-repo-exact` denotes a prompt-level reimplementation audited against public
  commit `7953d780c85054721a7b4bf246bcf60a16bb28af`, not the authors' original API run.
- MP-Bench `agent_or_role_any` uses benchmark-dependent role/agent matching; compare
  conditions within the same split.

The full interpretation and claim boundaries are documented in
[../docs/REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md).

