# 포트폴리오용 프로젝트 설명

## 한 줄 소개

멀티에이전트 시스템의 실행 로그에서 **실패에 책임이 있는 Agent와 최초의 결정적
오류 Step을 자동 탐지**하는 LLM 기반 연구·실험 시스템을 설계하고 구현했다.

## 문제

멀티에이전트 시스템은 여러 Agent와 도구가 긴 실행 궤적을 만들기 때문에 최종
오답만으로는 어디에서 실패가 시작되었는지 알기 어렵다. 긴 로그를 한 번에 읽히면
중요한 오류가 묻힐 수 있고, 무조건 잘게 나누면 원인과 결과의 연결이 끊어진다.
또한 최초 실수, 회복된 실수, downstream symptom을 구분해야 한다.

## 최종 접근

TSR-Loc은 로그를 보기 전에 task description을 자연어 성공 요구사항으로 변환하고
이를 고정한다. 두 번째 Judge는 이 요구사항을 기준으로 전체 로그를 시간순으로
검사하여 다음을 출력한다.

1. 책임 Agent
2. 최초의 미회복 결정적 오류 Step
3. 위반한 성공 요구사항과 인과적 이유

No-GT 조건에서 requirement compiler는 정답, 최종 시스템 오답, 실행 로그,
gold Agent/Step을 받지 않는다.

## 직접 결정하고 수행한 핵심 작업

- Who&When의 All-at-once, Step-by-step, Binary Search, Hybrid를 동일 runner에 통합
- Llama-3.1-8B 로컬 GPU와 GPT-4o API를 같은 평가 인터페이스로 연결
- 10-chunk, adaptive chunking, top-k beam, selected reread 실험 설계
- 4-view chunk scoring과 pairwise reranking을 포함한 MVBS 실험 설계
- 복합 점수 대신 단일 ordinal blame score를 사용하는 조건 설계
- Agent accuracy와 exact Step accuracy를 분리하고 `±3`, `±5`, MAD를 추가
- A2P 공개 저장소 prompt와 evaluator를 별도로 감사하고 repo-exact 조건 구축
- MP-Bench 다중 annotation을 Any/Majority/Unanimous 기준으로 평가
- requirement compiler와 localizer를 Llama/GPT-4o로 교차한 2×2 실험 수행
- OOM, API null, 긴 Windows 경로, 잘못된 step indexing을 진단하고 resume·shard 실행 체계 구축

## 대표 결과

Who&When 전체 184개, GPT-4o, strict exact-step evaluator:

| 방법 | Agent | Exact Step |
|---|---:|---:|
| Who&When official-style Direct | 51.63% | 8.15% |
| A2P repo-exact | 63.04% | 33.15% |
| TSR-Loc No-GT | 57.61% | 38.59% |
| TSR-Loc GT-assisted | 60.33% | 39.67% |

Direct 대비 No-GT TSR-Loc의 exact-step 차이는 `+30.43%p`였고 McNemar
검정은 `p=5.77e-12`였다. A2P 대비 차이는 `+5.43%p`였으나 통계적으로
유의하지 않았다.

HC-long 23개에서 Exact Step은 Direct 0%, A2P 4.35%, No-GT TSR-Loc
26.09%, GT-assisted 30.43%였다. 이 subset은 사후적으로 반복 관찰했으므로
장기 로그 전체에 대한 SOTA 증거로 사용하지 않는다.

## 모델 역량 분석

| Requirement compiler | Localizer | Exact Step |
|---|---|---:|
| Llama-3.1-8B | Llama-3.1-8B | 18.48% |
| GPT-4o | Llama-3.1-8B | 19.02% |
| Llama-3.1-8B | GPT-4o | 37.50% |
| GPT-4o | GPT-4o | 38.59% |

이 결과는 requirement 생성기는 경량 모델로 대체할 가능성이 있지만, 긴 로그에서
이를 활용해 exact step을 고르는 localizer 역량이 더 중요하다는 점을 보여준다.
Llama compiler + GPT localizer는 all-GPT exact의 97.2%를 유지했지만,
이는 통계적 동등성 증명이 아니라 기술 통계다.

## 시행착오와 방향 수정

초기 핵심 가설은 “긴 로그를 적절히 chunking하고 beam search하면 보편적으로
성능이 오른다”였다. 실제로는 A2P/Who&When 일부 조건에서 이득이 있었지만 CCV,
ECHO, Paper Hybrid에서는 global routing이 같거나 악화되기도 했다. chunk size와
top-k를 통제한 HC-long 실험도 비단조적이었다.

따라서 보편적 chunking 주장을 폐기하고, task interpretation을 trace 이전에
고정하는 end-to-end 진단 인터페이스로 연구 중심을 변경했다. 실패한 결과를 숨기지
않고 연구 질문을 수정한 것이 프로젝트의 중요한 의사결정이었다.

## 선행연구와 본 프로젝트의 구분

**선행연구에서 가져온 것**

- Who&When의 benchmark와 Direct/Step-by-step/Binary/Hybrid
- A2P의 Abduct-Act-Predict scaffold
- ECHO의 multi-analyst consensus
- AgentRx의 constraint 기반 진단 아이디어
- MP-Bench의 다중 annotation 평가

**프로젝트에서 제안한 것**

- 10-way Multi-View Beam Log Search와 pairwise candidate reranking
- task-derived success requirements를 trace 이전에 동결하는 TSR-Loc 인터페이스
- compiler/localizer 분리와 cross-model 2×2 분석
- exact-step 정의, 회복 여부, indexing 및 모델 역량을 함께 감사하는 평가 절차

**Codex가 보조한 구현 세부**

- Python runner, LLM backend, parser, chunking 함수, 통계·리포트 스크립트
- 반복·shard·resume 실행과 긴 경로/API 오류 대응
- 실험 문서, 논문 표, 그림 정리

개별 코드 줄의 작성자를 더 세밀하게 분리하는 것은 불가능하므로, 포트폴리오에서는
연구 기획·실험 판단·검증을 주도하고 Codex를 구현 협업 도구로 사용했다고 기술한다.

## 이 프로젝트가 보여주는 역량

- 불명확한 연구 문제를 평가 가능한 Agent/Step localization 문제로 형식화
- 논문 방법 재현과 evaluator 감사
- 로컬 GPU와 상용 API를 연결한 실험 인프라 구축
- 비단조적·부정적 결과를 분석하여 가설과 연구 방향 수정
- paired significance, 다중 annotation, human audit를 포함한 실증 평가

