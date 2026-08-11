# TSR-Loc: Multi-Agent Failure Attribution

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Evaluation](https://img.shields.io/badge/Evaluation-Agent%20%2B%20Exact%20Step-176B87)
![Training](https://img.shields.io/badge/Fine--tuning-None-5B6573)

[한국어 프로젝트 설명](docs/PORTFOLIO_KO.md) ·
[방법론](docs/METHOD.md) ·
[결과표](results/README.md) ·
[재현성 및 한계](docs/REPRODUCIBILITY.md) ·
[학습·엔지니어링 로드맵](docs/LEARNING_ROADMAP.md)

TSR-Loc is a training-free research prototype for identifying **which agent failed**
and **the exact step where the decisive error first occurred** in a multi-agent
execution trajectory.

The central design separates task interpretation from trace inspection:

1. compile the task into a frozen, natural-language success specification;
2. inspect the trajectory against that specification and localize the earliest
   unrecovered error as an `(agent, step)` pair.

The project began as a long-trace chunk-search study and grew into a full experimental
harness covering direct judging, step-wise search, binary search, agent-first methods,
A2P, ECHO-style analysis, constraint-guided localization, multi-view beam search, and
TSR-Loc.

> **Portfolio status:** research artifact, not a production monitoring service.
> Benchmark-wide SOTA and formal causal-verification claims are intentionally avoided.


## Results at a glance

All numbers below come from completed runs with the strict local evaluator.

### Who&When, 184 trajectories, GPT-4o

| Method | Agent accuracy | Exact-step accuracy |
|---|---:|---:|
| Who&When official-style Direct | 51.63% | 8.15% |
| A2P repository-exact reimplementation | **63.04%** | 33.15% |
| TSR-Loc, task-only / No-GT | 57.61% | 38.59% |
| TSR-Loc, GT-assisted | 60.33% | **39.67%** |

The Direct-to-No-GT exact-step difference is `+30.43 pp` with paired McNemar
`p = 5.77e-12`. The No-GT advantage over A2P is `+5.43 pp`, but it is **not
statistically significant** (`p = 0.2954`).

On the exploratory HC-long subset of 23 trajectories, exact-step accuracy was
`0.00%` for Direct, `4.35%` for A2P, `26.09%` for No-GT TSR-Loc, and `30.43%`
for GT-assisted TSR-Loc. This subset was repeatedly inspected and is not used for
a general long-trace SOTA claim.


### Compiler and localizer capacity

The model-factorial experiment separates the requirement compiler from the trajectory
localizer. Replacing the compiler had little effect in this model pair, while replacing
the localizer changed exact-step accuracy by roughly 19 percentage points.


## Research contributions represented in this repository

- A unified evaluator for agent accuracy, exact-step accuracy, tolerance accuracy,
  mean absolute step distance, call count, and token usage.
- A task-success-requirement interface that freezes task interpretation before trace
  evidence is exposed to the localizer.
- Reproducible implementations of direct, step-by-step, binary-search, agent-first,
  A2P-style, ECHO-style, CCV, and custom chunk-search conditions.
- Multi-View Beam Log Search (`mvbs10`): four-view chunk scoring, top-k expansion,
  window localization, and LLM pairwise candidate ranking.
- Cross-model compiler/localizer factorial experiments and multi-annotation MP-Bench
  evaluation.
- Engineering support for local Hugging Face models, OpenAI-compatible APIs,
  Anthropic APIs, OpenRouter routing, retries, sharding, resume, and usage accounting.

## Method sketch

```text
Task description
      |
      v
Requirement compiler
  - no execution trace
  - no gold attribution
  - no reference answer in No-GT mode
      |
      v
Frozen natural-language success requirements
      |
      v
Full-trajectory causal localizer
  - inspect steps in temporal order
  - ignore errors that were later recovered
  - prefer the earliest error whose minimal correction changes the outcome
      |
      v
Responsible agent + exact step
```

A worked example is available in [docs/METHOD.md](docs/METHOD.md).

## Repository layout

```text
failure_attribution/   Core schemas, prompts, methods, model backends, and metrics
configs/               Secret-free example configurations
data/                  One synthetic smoke-test case only
results/               Verified aggregate result tables
scripts/               Dataset, audit, significance, and reporting utilities
tests/                 Parser and prompt-contract tests
docs/                  Korean portfolio notes, method, results, and reproducibility
assets/                Paper figures used for project presentation
run_experiment.py      Unified experiment runner
```

The method and prompt registries preserve the complete research history and are larger
than production modules. Start with
[failure_attribution/README.md](failure_attribution/README.md) for a code map instead
of reading those registries from top to bottom.

## Quick start

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
powershell -ExecutionPolicy Bypass -File scripts\run_smoke.ps1 -Python python
```

The smoke test uses a deterministic mock backend and a synthetic trajectory. It does
not call a paid API.

Run one real OpenAI-compatible experiment after setting `OPENAI_API_KEY`:

```powershell
python run_experiment.py `
  --config configs\openai.example.toml `
  --data path\to\who_and_when.jsonl `
  --methods who_when_official_all_at_once,a2p_repo_exact,ccv_ablation_no_gt `
  --out outputs\who_when_main
```

In the code and historical outputs, `ccv_ablation_no_gt` is the canonical task-only
TSR-Loc condition. The historical method name is retained so old experiment artifacts
remain readable.

## Supported model backends

- deterministic `mock` backend for tests
- Hugging Face Transformers `local_hf`
- Ollama
- OpenAI-compatible HTTP APIs, including OpenAI, LM Studio, vLLM, and OpenRouter
- Anthropic Messages API
- `llama.cpp` CLI

API keys are loaded from environment variables. No credential is stored in this
portfolio repository.

## Data

Benchmark files are not redistributed. Use the authors' releases and convert them to
the JSONL schema documented in [docs/DATA_AND_EVALUATION.md](docs/DATA_AND_EVALUATION.md).
The project used:

- Who&When: 126 algorithm-generated and 58 hand-crafted trajectories
- MP-Bench: 120 Automatic and 169 Manual trajectories after local conversion
- TraceElephant: a 218-case compact Who&When-style conversion for exploratory analysis

## Reproducibility and claim boundaries

Read [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) before interpreting the tables.
The most important boundaries are:

- TSR-Loc is evaluated as a complete two-stage algorithm. The project does not claim
  that requirement text alone explains the entire gain.
- Exact-step gains over A2P on Who&When were not statistically significant.
- HC-long has only 23 post-hoc cases.
- MP-Bench evaluates compatibility with multiple expert annotations, not an exact
  reproduction of the benchmark authors' attribution system.
- TraceElephant compact results are not native full-observability benchmark results.
- Token efficiency and universal SOTA are not claimed.

## Author role

Chanhyuk Yoon led the research framing, experimental decisions, failure analysis,
evaluation protocol, and iterative validation. Codex was used as an implementation
and analysis collaborator. See [docs/PORTFOLIO_KO.md](docs/PORTFOLIO_KO.md) for the
Korean project narrative and a strict separation of original ideas, prior work, and
implementation details.

추가 학습 및 운영 확장 계획은 [docs/LEARNING_ROADMAP.md](docs/LEARNING_ROADMAP.md)에 정리했습니다.

## Citation and prior work

This repository evaluates and adapts ideas from Who&When, A2P, ECHO, AgentRx, MP-Bench,
and TraceElephant. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Cite the original
papers and datasets when using their prompts, protocols, or data.
