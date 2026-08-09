# Learning & Engineering Roadmap

## 1. 로그 데이터 엔지니어링

- agent, step, tool call, observation, timestamp를 갖는 typed event schema 설계
- 손상된 JSONL, 중복 event, 누락된 parent를 검출하는 validation
- 긴 로그를 streaming으로 읽고 재시작 가능한 ingestion 구현

**완료 증거:** JSON Schema, validation 테스트, 1GB 합성 로그 부하 실험

## 2. 평가·통계

- agent accuracy, step accuracy, joint accuracy, top-k recall 분리
- bootstrap confidence interval과 paired significance test
- judge 순서·표현 변화에 대한 민감도 테스트

**완료 증거:** 한 명령으로 생성되는 결과 표와 error taxonomy 리포트

## 3. 관측 가능성과 서비스화

- OpenTelemetry span을 평가 trace로 변환
- async worker와 idempotent job 설계
- FastAPI 기반 분석 endpoint와 CLI 제공
- CI에서 작은 고정 benchmark 회귀 테스트

**완료 증거:** trace 업로드부터 원인 후보 출력까지 end-to-end demo

