# Core package map

This package preserves the executable research harness used across the completed
experiments. It favors backward compatibility with historical method names and output
artifacts over a minimal production API.

## Start here

| File | Responsibility |
|---|---|
| `schema.py` | `Case`, `LogStep`, and `Prediction` data structures |
| `io.py` | Who&When-style JSONL loading and output serialization |
| `llm.py` | mock, local Hugging Face, OpenAI-compatible, Anthropic, Ollama, and llama.cpp backends |
| `metrics.py` | agent accuracy, exact/±k step metrics, MAD, and usage summaries |
| `chunking.py` | fixed-count and adaptive token/step chunk allocation |
| `methods.py` | method registry, execution pipelines, parsers, and historical aliases |
| `prompts.py` | prompt registry for baselines, TSR-Loc, A2P, ECHO-style, CCV, and chunk methods |
| `shared_cache.py` | frozen requirement/constraint cache validation |

## Main entry points

- `run_method(...)` in `methods.py`: central method dispatcher.
- `run_who_when_official_all_at_once(...)`: official-style Direct condition.
- `run_a2p_repo_exact(...)`: public-repository prompt reimplementation.
- `run_ccv_information_ablation(...)` with both ground-truth flags set to `False`:
  canonical task-only / No-GT TSR-Loc condition, historically exposed as
  `ccv_ablation_no_gt`.
- `run_ccv_full_trace(...)`: GT-assisted full-trace condition.
- `run_mvbs10(...)`: implemented Multi-View Beam Log Search experiment.

## Why the registries are large

The project evaluated many frozen prompt and allocation conditions while preserving
old output compatibility. Consequently, `methods.py` and `prompts.py` include both
canonical methods and exploratory ablations. For a production rewrite, the recommended
split would be:

```text
methods/
  baselines.py
  tsr_loc.py
  a2p.py
  chunk_search.py
prompts/
  baselines.py
  tsr_loc.py
  external_methods.py
```

That refactor is intentionally not performed in this research snapshot because it
would make the released code differ from the implementation that produced the stored
experiment artifacts.

