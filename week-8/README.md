# 8주차 실습 과제: AI Agent Observability

## 배경

7주차에는 6주차 설계를 바탕으로 실제로 동작하는 AI Agent를 구현했습니다.

8주차에는 새 Agent를 다시 만드는 것이 아니라, 7주차 Agent가 실제로 어떻게 동작했는지 관찰할 수 있도록 trace와 실행 로그를 남기고 분석합니다.

이번 주의 핵심은 아래 두 가지입니다.

- Agent가 어떤 Tool을 어떤 순서로 호출했는지 확인할 수 있어야 합니다.
- 정상 케이스와 실패 또는 예외 케이스를 비교해 Agent의 동작 특성을 설명할 수 있어야 합니다.

## 과제 목표

7주차에 구현한 Agent에 observability를 추가합니다.

최소한 아래를 확인할 수 있어야 합니다.

1. 사용자 입력
2. 선택된 Tool
3. Tool 인자
4. Tool 결과 또는 에러
5. 최종 응답
6. 종료 시점 또는 중단 이유

## 제출 방식

### 1. 개인 프로젝트 repository

개인 프로젝트 repository에는 실제 코드와 trace/log 저장 기능이 포함되어 있어야 합니다.

권장 포함 항목:

- 실행 코드
- 로그 저장 코드
- trace 예시 2개 이상
- 정상 케이스 1개
- 실패 또는 예외 케이스 1개
- 민감정보 처리 방식

### 2. ai-agent-repo 제출

이 repository에는 구현 코드를 올리지 않고, 아래 경로에 요약 README만 제출합니다.

```text
week-8/{github-id}/README.md
```

README에는 아래 내용을 정리합니다.

- 개인 프로젝트 repository 링크
- 어떤 observability 방식을 사용했는지
- 대표 trace 2개 이상
- trace 분석
- metrics 또는 latency 요약
- 민감정보 처리 방식

## 권장 기록 항목

로그는 정교한 모니터링 시스템일 필요는 없지만, 아래 정도는 확인 가능하면 좋습니다.

```text
user input
selected tool
tool arguments
tool result
tool error
final answer
stop reason
latency
```

## 제출 현황

| github-id | 제출 README |
|-----------|-------------|
| `monkana` | [README.md](/Users/jms/Desktop/project/aiagent-repo/week-8/monkana/README.md) |
