# 8주차 AI Agent Observability

## 프로젝트 링크

- Repository: https://github.com/monkama/AiagentForHandlinOfCivilPetitionsService
- 7주차 제출 README: https://github.com/monkama/aiagent-repo/blob/main/week-7/monkana/README.md

## 구현한 Observability

- 사용한 방식: JSON log
- trace 저장 위치: `logs/openai_sdk/<run-id>/run.json`, `logs/openai_sdk/<run-id>/tool_calls/*.json`
- 기록하는 항목:
  - `user_input`
  - `conversation_turns`
  - `tool_call_sequence`
  - `tool_arguments`
  - `tool_result`
  - `tool_error`
  - `started_at`, `completed_at`
  - `final_answer`

추가로 최근 로거 개편에서 `stop_reason`, `total_latency_ms`, `tool latency_ms`를 명시적으로 남기도록 보강하고 있습니다. 아래 예시는 실제로 남겨진 대표 trace를 기준으로 요약했습니다.

## Agent 실행 흐름

- Agent 이름: 행정 민원 내비게이터 Agent
- 주요 Tool:
  - `RequestClassifierTool`
  - `RegionNormalizeTool`
  - `AgencyRoutingTool`
  - `PublicServiceSearchTool`
  - `RequirementAndDraftTool`
- 종료 조건:
  - 최종 답변 생성
  - 추가 질문 필요 시 질문 생성 후 세션 유지
  - 같은 Tool/같은 입력 반복 호출 방지
  - Tool 실패 시 일반 안내 후 종료
  - `max_turns = 6`

## 정상 케이스 Trace

입력:

```text
이사했는데 전입신고는 어디서 어떻게 하나요?
```

대표 trace:

- `logs/openai_sdk/20260515_173805_이사했는데-전입신고는-어디서-어떻게-하나요/run.json`

실행 요약:

| Step | Type | Name | 주요 입력 | 결과 |
|------|------|------|-----------|------|
| 1 | tool_call | `RequestClassifierTool` | `이사했는데 전입신고는 어디서 어떻게 하나요?` | `administrative_procedure`, `residence`, `needs_region=true`, `region_text=null` |
| 2 | tool_call | `AgencyRoutingTool` | `category=residence`, `keywords=[전입신고, 이사, 주민등록]`, `region_text=서울특별시 강남구` | `정부24 온라인 신청`, `강남구 관할 행정복지센터 방문` 채널 반환 |
| 3 | tool_call | `RequirementAndDraftTool` | `category=residence`, `agency_name=서울특별시 강남구청`, `user_input=서울특별시 강남구에서 전입신고하고 싶습니다.` | 준비물, 누락 정보, 안내 문안 초안, 주의사항 반환 |

최종 답변:

```text
서울시 강남구에서 전입신고를 하시려면 아래 방법 중 하나를 이용하실 수 있습니다:

- 온라인: 정부24(www.gov.kr)에서 전입신고 신청
- 오프라인: 강남구 관할 행정복지센터(주민센터) 방문

준비물:
- 신분증(원본)
- 전입할 정확한 주소
- 가족관계증명서(필요시, 가족과 함께 전입하는 경우)
- 신청서(센터 또는 구청 홈페이지에서 다운로드 가능)
```

## 실패 또는 예외 케이스 Trace

입력:

```text
회사에서 퇴직금을 안 줬는데 어디에 신고해야 하나요?
```

대표 trace:

- `logs/openai_sdk/20260515_164240_회사에서-퇴직금을-안-줬는데-어디에-신고해야-하나요/run.json`

실행 요약:

| Step | Type | Name | 주요 입력 | 결과 |
|------|------|------|-----------|------|
| 1 | tool_call | `RequestClassifierTool` | `회사에서 퇴직금을 안 줬는데 어디에 신고해야 하나요?` | `issue_resolution`, `labor`, `urgency=high` |
| 2 | tool_call | `AgencyRoutingTool` | `category=labor`, `keywords=[퇴직금, 신고, 임금체불, 노동청]` | `ROUTING_API_FAILED`, `기관 라우팅 API 호출 중 오류가 발생했습니다: 'list' object has no attribute 'get'` |

실패 처리:

- `AgencyRoutingTool` 에러가 발생한 뒤 같은 입력으로 재시도하지 않았습니다.
- 오케스트레이터는 일반 안내 fallback으로 전환해 `고용노동부 근로감독관서(노동청)`, `1350`, `온라인 신고`를 안내했습니다.
- 반복 호출 방지 규칙 때문에 같은 Tool을 다시 호출하는 흐름은 나타나지 않았습니다.

## Trace 분석

- 예상한 흐름:
  - 전입신고: `RequestClassifierTool -> 추가 질문 -> AgencyRoutingTool -> RequirementAndDraftTool -> final`
  - 퇴직금: `RequestClassifierTool -> AgencyRoutingTool -> final`
- 실제 흐름:
  - 전입신고 trace는 첫 턴에서 지역 추가 질문을 만들고, 사용자가 `서울시 강남구`를 답한 뒤 같은 세션에서 `AgencyRoutingTool`과 `RequirementAndDraftTool`이 이어서 호출됐습니다.
  - 실패 trace는 `RequestClassifierTool`까지는 정상 동작했지만 `AgencyRoutingTool` 파싱 오류로 중단됐고, fallback 일반 안내로 종료됐습니다.
- 잘 동작한 부분:
  - tool 이름, 인자, 결과, 에러를 JSON으로 남겨서 어느 단계에서 실패했는지 바로 확인할 수 있었습니다.
  - follow-up 질문이 별도 새 요청이 아니라 같은 세션 흐름으로 이어졌는지 `conversation_turns`로 재확인할 수 있었습니다.
  - 실패 시 같은 Tool을 반복 호출하지 않고 종료하는 동작이 로그로 확인됐습니다.
- 문제 또는 개선할 부분:
  - 대표 trace 시점에는 `stop_reason`, `step latency`가 별도 필드로 저장되지 않아 README에서 추론/계산해야 했습니다.
  - `AgencyRoutingTool`은 외부 API 파싱 형식 변화에 민감했고, 실제로 `'list' object has no attribute 'get'` 예외가 발생했습니다.
  - `AgencyRoutingTool`의 기관명 정규화가 과도하게 세부 하위조직명으로 치우친 사례가 있어 후처리 보정이 더 필요합니다.

## Metrics

| 항목 | 값 | 설명 |
|------|----|------|
| total latency (normal) | 약 `18.6s` | `20260515_173805` trace의 `started_at`~`completed_at` 기준 |
| step count (normal) | `3` | classifier, routing, draft |
| tool error count (normal) | `0` | 정상 케이스에서는 Tool 에러 없음 |
| total latency (failure) | 약 `9.1s` | `20260515_164240` trace의 `started_at`~`completed_at` 기준 |
| step count (failure) | `2` | classifier 후 routing에서 실패 |
| tool error count (failure) | `1` | `AgencyRoutingTool`에서 `ROUTING_API_FAILED` 발생 |

## 민감정보 처리

- 저장하지 않은 정보:
  - `.env`, API key, 개인 token
  - 주민등록번호, 결제 정보, 상세 주소, 연락처
- masking한 정보:
  - 샘플 trace에는 `user_id`, `order_id` 같은 식별자가 없어서 별도 masking 대상은 없었습니다.
  - 향후 식별자가 생기면 `masked_user_id` 같은 필드로 저장할 계획입니다.
- trace 공유 시 주의할 점:
  - 예시 질의는 공개 가능한 샘플 문장만 사용했습니다.
  - 실제 사용자 입력을 넣는 경우 구체 주소, 계좌, 연락처는 제거하거나 마스킹해야 합니다.

## 고도화 평가

| 평가 항목 | 구현 여부 | 결과 |
|-----------|-----------|------|
| correctness | 미구현 | 별도 evaluator 없음 |
| groundedness | 미구현 | tool result와 final answer의 자동 비교 미구현 |
| tool completeness | 미구현 | 수동 trace 분석만 수행 |

## 배운 점

- 최종 답변만 보면 괜찮아 보여도, trace를 보면 특정 Tool 파싱 오류나 잘못된 기관명 정규화 같은 문제가 바로 드러났습니다.
- follow-up 질문이 있는 Agent는 단순 1턴 로그보다 `conversation_turns`까지 남겨야 실제 동작을 재현할 수 있었습니다.
- 실패 trace 1개만 있어도, 어떤 Tool이 가장 취약한지와 fallback이 제대로 동작하는지 훨씬 분명하게 확인할 수 있었습니다.
