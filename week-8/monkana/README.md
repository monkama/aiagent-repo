# 8주차 AI Agent Observability

## 프로젝트 링크

- Repository: https://github.com/monkama/AiagentForHandlinOfCivilPetitionsService
- 7주차 제출 README: https://github.com/monkama/aiagent-repo/blob/main/week-7/monkana/README.md

## 구현한 Observability

- 사용한 방식: JSON log
- trace 저장 위치:
  - `logs/openai_sdk/<run-id>/run.json`
  - `logs/openai_sdk/<run-id>/summary.md`
- 대표적으로 기록하는 항목:
  - `user_input`
  - `conversation_turns`
  - `tool_call_sequence`
  - `tool_calls[].selected_tool`
  - `tool_calls[].tool_arguments`
  - `tool_calls[].tool_result`
  - `final_answer`
  - `started_at`
  - `completed_at`

현재 프로젝트의 최신 로거는 `stop_reason`, `total_latency_ms`도 함께 남기도록 확장 중입니다. 이번 README에서는 실제로 Tool 호출 흐름이 분명하게 남아 있는 대표 trace를 기준으로 분석했습니다.

## Agent 실행 흐름

- Agent 이름: 행정 민원 내비게이터 Agent
- 주요 Tool:
  - `RequestClassifierTool`
  - `RegionNormalizeTool`
  - `AgencyRoutingTool`
  - `PublicServiceSearchTool`
  - `RequirementAndDraftTool`
- 종료 규칙:
  - 추가 질문이 필요하면 같은 세션 안에서 follow-up 질문 생성
  - 같은 Tool과 같은 입력 반복 호출 방지
  - Tool 실패 시 fallback 안내 후 종료
  - `max_turns = 6`

## 정상 케이스 Trace

입력:

```text
이사했는데 전입신고는 어디서 어떻게 하나요?
```

대표 trace:

- `logs/openai_sdk/20260514_043412_이사했는데-전입신고는-어디서-어떻게-하나요/run.json`

실행 요약:

| Step | Type | Name | 주요 입력 | 결과 |
|------|------|------|-----------|------|
| 1 | tool_call | `RequestClassifierTool` | `이사했는데 전입신고는 어디서 어떻게 하나요?` | `administrative_procedure`, `residence`, `needs_region=true`, `region_text=null` |
| 2 | follow-up | assistant question | `새로 이사한 지역을 알려주세요` | 사용자 답변 `서울시 강남구` 획득 |
| 3 | tool_call | `AgencyRoutingTool` | `category=residence`, `keywords=[전입신고, 이사, 행정절차, 주민등록]`, `region_text=서울시 강남구` | `정부24 온라인 신청`, `전입지 관할 주민센터 방문` 반환 |
| 4 | tool_call | `RequirementAndDraftTool` | `agency_name=행정복지센터`, `agency_unit=정부24 또는 전입지 주민센터` | 준비 정보, 누락 정보, 초안, 주의사항 반환 |

최종 답변 요약:

- 정부24 온라인 신청 가능
- 강남구 관할 행정복지센터 방문 가능
- 준비물로 신분증, 새 주소, 전입일, 세대주 여부 등이 안내됨

이 trace에서 확인한 점:

- 추가 질문이 별도 새 요청이 아니라 같은 `conversation_turns` 안에서 이어졌습니다.
- 실제 Tool 순서가 `classify -> route -> draft`로 남았습니다.
- 사용자 지역 응답 이후 불필요한 재분류 없이 다음 Tool로 진행했습니다.

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
| 2 | tool_call | `AgencyRoutingTool` | `category=labor`, `keywords=[퇴직금, 신고, 임금체불, 노동청]` | `ok=false`, `ROUTING_API_FAILED` |
| 3 | fallback | assistant answer | Tool 실패 결과 관찰 | 고용노동부/1350/온라인 신고 일반 안내 후 종료 |

실패 처리:

- `AgencyRoutingTool`이 `ROUTING_API_FAILED`를 반환했습니다.
- 오케스트레이터는 같은 Tool을 같은 입력으로 재호출하지 않았습니다.
- 대신 일반 fallback 안내로 전환해 사용자가 바로 다음 행동을 할 수 있게 마무리했습니다.

이 trace에서 확인한 점:

- Tool 에러가 구조화된 형태로 기록됐습니다.
- Tool 실패 뒤 무한 재시도 없이 종료됐습니다.
- 실패해도 최종 답변은 비어 있지 않고 대체 안내를 제공합니다.

## Trace 분석

- 설계서에서 예상한 흐름은 `decide -> tool call -> observe -> decide -> final` 이었습니다.
- 정상 trace에서는 실제로 `RequestClassifierTool -> AgencyRoutingTool -> RequirementAndDraftTool` 순서가 확인됐습니다.
- 전입신고 케이스는 follow-up 질문이 있어도 세션이 끊기지 않고 이어졌습니다.
- 실패 trace에서는 `AgencyRoutingTool` 오류가 바로 드러났고, 어떤 Tool에서 막혔는지 로그만 보고 바로 찾을 수 있었습니다.
- 특히 실패 trace는 외부 API 파싱이 취약점이라는 점을 보여줬고, 이후 라우팅 로직을 다시 손보는 근거가 됐습니다.

## Metrics

| 항목 | 값 | 설명 |
|------|----|------|
| total latency (normal) | 약 `13.5s` | `20260514_043412` trace의 `started_at`~`completed_at` 기준 |
| step count (normal) | `3` | classifier, routing, draft |
| tool error count (normal) | `0` | 정상 케이스에서는 Tool 에러 없음 |
| total latency (failure) | 약 `9.1s` | `20260515_164240` trace의 `started_at`~`completed_at` 기준 |
| step count (failure) | `2` | classifier 후 routing에서 실패 |
| tool error count (failure) | `1` | `AgencyRoutingTool`에서 `ROUTING_API_FAILED` 발생 |

## 민감정보 처리

- 저장하지 않은 정보:
  - `.env`, API key, 개인 token
  - 주민등록번호, 계좌번호, 연락처
- 공개 가능한 예시 문장만 사용:
  - 퇴직금 미지급 문의
  - 전입신고 문의
- 실제 사용자 입력을 로그로 남길 때는 상세 주소, 연락처, 식별번호를 제거하거나 마스킹해야 합니다.

## 고도화 평가

| 평가 항목 | 구현 여부 | 결과 |
|-----------|-----------|------|
| correctness | 미구현 | 별도 evaluator 없음 |
| groundedness | 부분 구현 | Tool 결과와 final answer를 수동 trace 분석으로 비교 |
| tool completeness | 부분 구현 | Tool 호출 순서, 인자, 결과, 에러는 남기지만 자동 평가는 없음 |

## 배운 점

- 최종 답변만 보면 정상처럼 보여도, trace를 보면 어떤 Tool이 어떤 이유로 실패했는지 훨씬 선명하게 보였습니다.
- follow-up 질문이 있는 Agent는 `conversation_turns`를 함께 남겨야 실제 사용자 경험을 복원할 수 있었습니다.
- 실패 trace 1개만 있어도 라우팅 로직의 취약점과 fallback 품질을 동시에 점검할 수 있었습니다.
