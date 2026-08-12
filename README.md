![TSR-Loc project hero](assets/project-hero.svg)

<div align="center">

**Multi-agent execution trace에서 task requirements를 먼저 고정하고, final failure로 이어진 earliest unrecovered error를 agent·step 수준에서 localization하는 training-free 평가 프레임워크**

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Evaluation](https://img.shields.io/badge/Evaluation-Agent%20%2B%20Exact%20Step-7C3AED)
![Training](https://img.shields.io/badge/Fine--tuning-None-5B6573)
![CI](https://github.com/yoon-chan-hyeok/multi-agent-failure-localization/actions/workflows/ci.yml/badge.svg)

[핵심 결과](#핵심-결과) · [방법](#tsr-loc의-구조) · [빠른 실행](#빠른-실행) · [한국어 설명](docs/PORTFOLIO_KO.md)

</div>

---

## 이 프로젝트는

Multi-agent system이 최종 답변에 실패했을 때 전체 trace에는 계획, 도구 호출, 다른 agent의 수정 시도와 최종 응답이 한꺼번에 남습니다. 마지막에 잘못 말한 agent만 찾거나 trace 전체를 한 번에 요약하면 실제로 결과를 바꾼 최초 오류를 놓치기 쉽습니다. TSR-Loc은 task의 성공 조건을 먼저 고정한 뒤 trace를 시간순으로 읽어, 이후 단계에서도 복구되지 않은 가장 이른 오류를 찾습니다.

출력은 responsible agent와 exact step입니다. 별도 fine-tuning 없이 기존 model backend를 사용하며, requirement compiler와 temporal localizer를 분리해 task 해석과 trace evidence가 섞이지 않도록 구성했습니다. 결과표에는 agent-level accuracy와 exact-step accuracy를 따로 기록합니다.

### 시작한 이유

Agent 시스템을 고치려면 "실패했다"는 판정보다 어느 agent의 어떤 step을 다시 설계해야 하는지가 필요합니다. 그런데 기존 평가는 최종 성공 여부나 agent attribution에 머무는 경우가 많았습니다. 재현 가능한 trace-level 평가와 실제 수정 지점을 연결하기 위해 이 프로젝트를 만들었습니다.

## 상세 설명

| 구분 | 내용 |
|---|---|
| **Input** | task description과 agent별 action, observation이 포함된 execution trajectory |
| **Requirement compiler** | attribution label이나 trace를 보지 않고 성공 조건을 task-only requirement로 고정 |
| **Temporal localizer** | trace를 시간순으로 검사해 이후 step에서 복구되지 않은 가장 이른 오류를 선택 |
| **Output** | responsible agent와 exact failure step, requirement-level rationale |
| **Evaluation** | Who&When 184 trajectories에서 strict local evaluator와 paired significance test를 적용 |

TSR-Loc은 production monitoring service가 아니라 재현 가능한 evaluation artifact입니다. benchmark-wide SOTA나 causal attribution은 주장하지 않습니다.

## Exact-step attribution이 필요한 이유

“이 실행은 실패했다”만으로는 agent system을 고치기 어렵습니다. 운영자가 필요한 것은 다음 세 가지입니다.

1. 어떤 agent가 책임 있는가?
2. 어느 step에서 회복되지 않은 오류가 처음 발생했는가?
3. 그 판단이 task 요구조건과 어떤 관계가 있는가?

직접 전체 trace를 한 번에 판정하면 task 해석과 trace evidence가 섞이기 쉽습니다. TSR-Loc은 두 단계를 분리합니다.

## TSR-Loc의 구조

```mermaid
flowchart LR
    T["Task description"] --> C["Requirement compiler"]
    C --> R["Frozen success<br/>requirements"]
    X["Execution trajectory"] --> L["Temporal localizer"]
    R --> L
    L --> A["Responsible agent"]
    L --> S["Earliest decisive step"]
    A --> E["Strict evaluator"]
    S --> E
```

- **Compiler:** execution trace와 gold attribution을 보지 않고 성공 요구조건을 작성
- **Localizer:** trace를 시간순으로 읽고, 나중에 복구된 오류는 제외
- **Decision rule:** 최소 수정으로 최종 결과가 바뀌는 가장 이른 미복구 오류를 선택

## 핵심 결과

### Who&When · 184 trajectories · GPT-4o

| Method | Agent accuracy | Exact-step accuracy |
|---|---:|---:|
| Who&When official-style Direct | 51.63% | 8.15% |
| A2P repository-exact reimplementation | **63.04%** | 33.15% |
| **TSR-Loc task-only / No-GT** | 57.61% | **38.59%** |
| TSR-Loc GT-assisted | 60.33% | **39.67%** |

- Direct → task-only TSR-Loc exact-step: **+30.43%p**, paired McNemar `p = 5.77e-12`
- A2P → task-only TSR-Loc: **+5.43%p**, 통계적으로 유의하지 않음 `p = 0.2954`
- HC-long 23건은 반복 관찰된 post-hoc subset이므로 일반적 long-trace 우위를 주장하지 않음

### Factorial experiment에서 확인한 점

compiler와 localizer 모델을 교차한 실험에서 이 모델 조합에서는 compiler 교체 영향이 작았고, localizer 교체가 exact-step accuracy를 약 19%p 변화시켰습니다. 즉, 이 결과에서는 자연어 요구조건 생성 자체보다 **trace localization capacity**가 더 큰 병목 후보였습니다.

## 구현 범위

- agent accuracy, exact-step, tolerance, step distance, call count, token usage 통합 evaluator
- Direct, step-wise, binary-search, agent-first, A2P, ECHO-style, CCV, MVBS 조건
- OpenAI-compatible, Anthropic, OpenRouter, Hugging Face, Ollama, llama.cpp backends
- retry, sharding, resume, usage accounting
- Who&When·MP-Bench·TraceElephant 변환 및 평가 utilities
- deterministic mock smoke test와 GitHub Actions

## 저장소 구성

```text
failure_attribution/   schemas, prompts, methods, backends, metrics
configs/               secret-free example configurations
data/                  one synthetic smoke-test case
results/               verified aggregate tables
scripts/               dataset, audit, significance, reporting tools
tests/                 parser and prompt-contract tests
docs/                  method, results, reproducibility, Korean narrative
assets/                portfolio hero artwork
run_experiment.py      unified experiment runner
```

처음 코드를 읽는다면 [failure_attribution/README.md](failure_attribution/README.md)의 code map부터 보는 것이 가장 빠릅니다.

## 빠른 실행

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
powershell -ExecutionPolicy Bypass -File scripts\run_smoke.ps1 -Python python
```

smoke test는 deterministic mock backend와 synthetic trajectory를 사용하며 유료 API를 호출하지 않습니다.

실제 실험 예시는 [configs/openai.example.toml](configs/openai.example.toml)과 [데이터·평가 문서](docs/DATA_AND_EVALUATION.md)를 참고하세요.

## 해석 범위

- TSR-Loc은 compiler+localizer의 전체 2-stage algorithm으로 평가했습니다.
- A2P 대비 exact-step 차이는 통계적으로 유의하지 않았습니다.
- HC-long은 23개 post-hoc case입니다.
- MP-Bench 평가는 multiple expert annotation 호환성이지 저자 시스템의 완전 재현이 아닙니다.
- TraceElephant compact 결과는 native full-observability 결과가 아닙니다.
- universal SOTA와 token efficiency는 주장하지 않습니다.

## 기여 범위

연구 framing, 실험 설계, baseline 선정, failure analysis, evaluation protocol과 반복 검증을 맡았습니다. 선행연구와 자체 구현의 경계는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)와 [한국어 프로젝트 설명](docs/PORTFOLIO_KO.md)에 구분했습니다.

## 문서

[Method](docs/METHOD.md) · [Results](results/README.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Roadmap](docs/LEARNING_ROADMAP.md)


