# 미구현 확장 계획

아래 항목은 현재 프로젝트 결과가 아니라 이후 학습과 구현 후보입니다.

## 로그와 평가 운영화

- 실행·step·agent·tool event schema와 대용량 trace 저장
- batch 평가 queue, retry, idempotency와 failed-case 격리
- agent와 step attribution의 calibration, confidence interval과 error taxonomy
- API, worker, metrics, trace와 dashboard

완료 기준은 공개 synthetic trace로 재현되는 ingestion과 evaluation pipeline, CI, API와 운영 runbook입니다.
