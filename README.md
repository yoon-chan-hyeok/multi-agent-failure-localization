![TSR-Loc](assets/project-hero.svg)

<div align="center">

**여러 AI agent가 함께 수행한 작업이 실패했을 때, 책임 agent와 수정해야 할 정확한 step을 찾습니다.**

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Evaluation](https://img.shields.io/badge/Evaluation-Agent%20%2B%20Exact%20Step-7C3AED)
![Training](https://img.shields.io/badge/Fine--tuning-None-5B6573)
![CI](https://github.com/yoon-chan-hyeok/multi-agent-failure-localization/actions/workflows/ci.yml/badge.svg)

[결과](#결과) · [방법](#tsr-loc) · [실행](#실행) · [상세 문서](#상세-문서)

</div>

## 문제

Multi-agent 실행 기록에는 계획, 도구 호출, 수정 시도와 최종 응답이 함께 남습니다. 첫 오류만 고르면 뒤에서 복구된 사건을 원인으로 지목할 수 있고, 최종 응답만 보면 오류가 시작된 위치를 놓칩니다.

목표는 responsible agent뿐 아니라 최종 실패로 이어진 가장 이른 미복구 step을 찾는 것입니다. 그래야 긴 trace를 다시 전부 읽지 않고 수정할 위치를 정할 수 있습니다.

## 방향을 바꾼 이유

처음에는 긴 trace를 잘 나누면 된다고 생각했습니다. Fixed chunk, adaptive chunk, top-k reread와 reranking을 비교했지만 작은 chunk는 원인과 이후 맥락을 분리했고, 큰 chunk는 다시 long-context 문제가 됐습니다. Agent 선택은 일부 나아졌지만 exact step은 안정적으로 개선되지 않았습니다.

그래서 "로그를 어떻게 자를까" 대신 "이 과업이 성공하려면 무엇을 끝까지 지켜야 할까"를 먼저 묻도록 바꿨습니다.

## TSR-Loc

~~~mermaid
flowchart LR
    T["Task description"] --> C["Requirement compiler"]
    C --> R["Frozen success<br/>requirements"]
    X["Execution trace"] --> L["Recovery-aware<br/>localizer"]
    R --> L
    L --> A["Responsible agent"]
    L --> S["Earliest unrecovered step"]
~~~

1. Requirement compiler는 trace를 보기 전에 task description만으로 성공 조건을 만듭니다.
2. Localizer는 trace를 시간순으로 읽고 각 조건의 위반과 복구 여부를 확인합니다.
3. 끝까지 복구되지 않은 오류 중 가장 이른 step과 해당 agent를 반환합니다.

평가 과정에서는 recovered error 선택, downstream symptom 선택, step indexing과 agent/step 의미 혼동도 다시 확인했습니다. 같은 판정 규칙을 모든 비교 방법에 적용했습니다.

## 결과

Who&When 184 trajectories, GPT-4o, strict local evaluator 조건입니다.

| Method | Agent accuracy | Exact-step accuracy |
|---|---:|---:|
| Direct | 51.63% | 8.15% |
| A2P reimplementation | **63.04%** | 33.15% |
| **TSR-Loc task-only / No-GT** | 57.61% | **38.59%** |

TSR-Loc은 Direct 대비 exact-step accuracy가 30.43%p 높았고 paired McNemar p-value는 5.77e-12였습니다. A2P 대비 차이는 통계적으로 유의하지 않았습니다(p = 0.2954). 따라서 A2P보다 우수하거나 benchmark 전체의 SOTA라고 주장하지 않습니다.

## 실행

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
powershell -ExecutionPolicy Bypass -File scripts\run_smoke.ps1 -Python python
~~~

Smoke test는 synthetic trajectory와 deterministic mock backend를 사용합니다. 실제 benchmark data와 유료 API key는 포함하지 않습니다.

## 저장소 구성

~~~text
failure_attribution/   methods, backends, schemas, metrics
configs/               secret-free example configs
data/                  synthetic smoke case
results/               verified aggregate tables
scripts/               audit and reporting tools
tests/                 parser and prompt-contract tests
~~~

## 상세 문서

- [Method](docs/METHOD.md): algorithm과 worked example
- [Data and evaluation](docs/DATA_AND_EVALUATION.md): dataset schema와 평가 기준
- [Results](results/README.md): baseline과 ablation 결과
- [Experiment history](docs/EXPERIMENT_HISTORY.md): 실패한 접근과 방향 전환
- [Reproducibility](docs/REPRODUCIBILITY.md): 실행 조건

TSR-Loc은 fine-tuning 없이 compiler와 localizer를 결합한 평가 방법입니다. Formal causality, 모든 dataset으로의 일반화, 모든 baseline 대비 비용 우위는 검증하지 않았습니다.
