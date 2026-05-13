# 행정 민원 내비게이터 Agent

## 프로젝트 링크

- Repository: https://github.com/monkama/AiagentForHandlinOfCivilPetitionsService
- 6주차 설계 문서: https://github.com/monkama/aiagent-repo/blob/main/week-6/design.md

## 구현한 Agent

- Agent 이름: 행정 민원 내비게이터 Agent
- 해결하려는 문제: 사용자가 자신의 상황에 맞는 소관 기관, 접수 채널, 준비 정보를 빠르게 찾기 어렵다는 문제를 해결합니다.
- 타깃 사용자: 행정 민원, 생활 불편 신고, 전입신고 같은 절차를 진행하려는 일반 사용자


## 프로젝트 개요

이 프로젝트는 6주차 `design.md`를 바탕으로 만든 행정 민원 내비게이터 Agent의 7주차 MVP 구현입니다.

사용자의 자연어 요청을 입력받아 다음과 같은 흐름으로 처리합니다.

- 요청 유형 분류
- 소관 가능성이 높은 기관 탐색
- 접수 채널 안내
- 준비 정보와 민원 문안 초안 정리

현재 구현은 `퇴직금 미지급`, `전입신고`, `도로 붕괴 위험` 시나리오를 중심으로 동작합니다.

## 6주차 설계와의 연결

### 유지한 설계

- `RequestClassifierTool -> AgencyRoutingTool -> RequirementAndDraftTool` 흐름을 기본 축으로 유지했습니다.
- 지역 정보가 필요한 경우에는 `RegionNormalizeTool`을 거치거나, 지역이 없으면 추가 질문을 생성하도록 구성했습니다.
- 긴급도가 `emergency`인 경우 일반 민원 안내보다 즉시 대응 채널 안내를 우선하도록 했습니다.
- 최종 응답에는 기관 후보, 접수 채널, 준비 정보, 주의사항을 포함하도록 맞췄습니다.

### 변경한 설계 - 1 

처음에는 `RequestClassifierTool`을 규칙 기반 코드로 구현하는 방향도 고려했습니다.  
예를 들어 특정 키워드가 들어오면 `issue_resolution`, `labor`, `emergency` 같은 값을 if/else 로 분기하는 식입니다. 하지만 현재 구현에서는 이 분류 단계를 단순 코드 기반이 아니라 **LLM 기반 분류**로 변경했습니다.

### 변경 이유 - 1

- 실제 사용자 입력은 표현이 매우 다양해서, 같은 의도라도 키워드가 항상 동일하게 들어오지 않습니다.
- “퇴직금을 못 받았어요”, “회사에서 돈을 안 줘요”, “노동청에 신고해야 하나요?”처럼 비슷한 의도를 동적으로 인식해야 합니다.
- 행정 민원 도메인은 `요청 유형`, `행정 분야`, `긴급도`, `지역 필요 여부`를 함께 판단해야 하므로 단순 키워드 분기보다 LLM이 더 자연스럽게 처리할 수 있습니다.
- 7주차 과제의 핵심이 “LLM이 입력을 보고 tool 사용 여부와 순서를 판단하는 구조”에 있기 때문에, 분류 단계 역시 AI 기반으로 두는 편이 전체 설계 의도와 더 잘 맞는다고 판단했습니다.

즉, 원래는 코드 베이스 분류기로도 시작할 수 있었지만,  
현재는 **사용자 입력을 더 유연하게 이해하고 동적으로 구분하기 위해 AI 기반 분류 구조로 변경**했습니다.

### 변경한 설계 - 2

또한 6주차 설계에는 원래부터 “필수 정보가 부족하면 추가 질문을 생성한다”는 개념이 있었지만,
현재 구현에서는 그 추가 질문이 한 번 출력되고 끝나는 방식보다 **같은 세션에서 사용자의 답을 받아 다음 단계로 이어지는 방식**으로 구체화했습니다.

### 변경 이유 - 2
- 7주차 과제의 핵심이 “LLM이 입력을 보고 tool 사용 여부와 순서를 판단하는 구조”에 있기 때문에, 분류 단계 역시 AI 기반으로 두는 편이 전체 설계 의도와 더 잘 맞는다고 판단했습니다.
- 추가 질문을 한 번 던지고 종료하면 사용 흐름이 끊기기 때문에, 실제 사용성 측면에서는 후속 답변을 같은 대화 맥락에서 이어서 처리하는 방식이 더 자연스럽다고 판단했습니다.


## 현재 구현 상태

### 구현된 Tool

| Tool 이름 | 실제/API/mock | 역할 |
|-----------|---------------|------|
| `RequestClassifierTool` | OpenAI API | 사용자 요청을 `response_type`, `category`, `urgency`, `needs_region` 등 분류 필드로 구조화 |
| `RegionNormalizeTool` | 공공 OpenAPI | 지역명을 표준 행정구역명과 법정동 코드로 정규화 |
| `AgencyRoutingTool` | mock | 분류 결과를 바탕으로 소관 가능성이 높은 기관 후보와 접수 채널 반환 |
| `RequirementAndDraftTool` | mock | 준비 정보, 누락 정보, 민원 문안 초안, 주의사항 반환 |

- `RequestClassifierTool`
  - OpenAI 모델을 사용해 사용자 요청을 분류합니다.
  - 반환 필드: `response_type`, `category`, `urgency`, `keywords`, `needs_region`, `confidence`, `region_text`

- `RegionNormalizeTool`
  - 지역명을 표준 행정구역 형태로 정리합니다.
  - 행정안전부 `행정표준코드_법정동코드` OpenAPI를 사용합니다.
  - 공공데이터포털 서비스키가 없으면 실패를 반환합니다.

- `AgencyRoutingTool`
  - 분류 결과를 바탕으로 소관 가능성이 높은 기관과 채널을 반환합니다.
  - 현재는 mock 데이터 기반입니다.

- `RequirementAndDraftTool`
  - 준비할 정보, 누락 정보, 민원 문안 초안을 생성합니다.
  - 현재는 mock 데이터 기반입니다.

### 현재 AI 사용 구조

- 바깥 실행 레이어는 `OpenAI Agents SDK`를 사용합니다.
- `RequestClassifierTool` 내부에서도 OpenAI 모델을 사용해 분류합니다.
- 분류 결과만으로 부족한 정보가 있으면, 추가 질문은 바깥 오케스트레이터가 생성합니다.

즉 현재는 “에이전트 실행 레이어”와 “분류 단계” 모두 AI를 활용하는 구조입니다.

## 실행 패턴

- 선택한 패턴:
  `OpenAI Agents SDK` 기반 단일 오케스트레이터 + Tool 호출 구조

- 이유:
  6주차 설계의 핵심이 `LLM이 Tool 사용 여부와 순서를 판단하는 구조`였기 때문에, 직접 고정 루프를 짜기보다 SDK 기반 오케스트레이션이 설계 의도와 더 잘 맞는다고 판단했습니다.

- 간단한 흐름:
  `decide -> RequestClassifierTool -> observe -> 필요 시 추가 질문 또는 RegionNormalizeTool / AgencyRoutingTool / RequirementAndDraftTool -> final`

## 종료 조건

현재 구현에서는 설계서 원문을 직접 수정하지 않고, 실제 실행 기준의 종료 조건만 별도로 단순화했습니다.

- `max_turns = 6`에 도달하면 실행을 종료합니다.
- 실행 시간이 60초를 초과하면 종료합니다.
- 같은 Tool을 같은 입력으로 반복 호출하려는 경우 실행을 종료합니다.
- Tool 실패가 발생하면 같은 입력으로 재시도하지 않고, 현재 가능한 일반 안내를 제공한 뒤 실행을 종료합니다.

## 프로젝트 구조

```text
project-root/
  README.md
  requirements.txt
  logs/
  examples/
    input_1.txt
    input_2.txt
    input_3.txt
  src/
    main.py
    tools/
      RequestClassifierTool.py
      RegionNormalizeTool.py
      AgencyRoutingTool.py
      RequirementAndDraftTool.py
      schemas.py
```

## 예시 입력 / 출력

- `examples/input_1.txt`
  - 퇴직금 미지급 문의

- `examples/input_2.txt`
  - 전입신고 절차 문의

- `examples/input_3.txt`
  - 도로 붕괴 위험 신고

### 예시 1. 퇴직금 미지급 문의

입력:

```text
회사에서 퇴직금을 안 줬는데 어디에 신고해야 하나요?
```

출력 요약:

```text
퇴직금을 지급하지 않는 회사는 노동관계법 위반에 해당할 수 있습니다. 아래와 같이 신고하실 수 있습니다.

• 소관 기관: 고용노동부 관할 지방고용노동관서
• 접수 방법:
  - 고용노동부 고객상담센터(1350) 전화
  - 고용노동부 홈페이지에서 온라인 민원/진정 접수
  - 관할 지방고용노동관서 직접 방문

진정 접수 시 퇴직일, 임금 내역, 사업장 정보 등 기본적인 사실관계를 준비해 가시면 도움이 됩니다.
```

### 예시 2. 전입신고 절차 문의

입력:

```text
이사했는데 전입신고는 어디서 어떻게 하나요?
```

추가 질문과 최종 출력 예시:

```text
전입신고는 이사하신 지역의 행정복지센터(동주민센터) 또는 온라인(정부24)에서 할 수 있습니다.

정확한 안내를 위해 이사하신 지역(시/군/구와 동/읍/면 이름)을 알려주시면, 해당 지역 소관 기관과 접수 방법을 안내해드릴 수 있습니다. 이사한 지역을 입력해 주세요!
> 서울시 강남구
서울시 강남구로 이사하셨다면 전입신고는 다음 두 가지 방법으로 하실 수 있습니다.

1. 정부24(www.gov.kr) 온라인 신청: 공인인증서(공동인증서) 등 본인 인증이 필요합니다.
2. 강남구 내 전입하신 주소지 관할 주민센터(행정복지센터) 직접 방문: 신분증(주민등록증, 운전면허증 등), 이사 사실을 증빙할 서류(임대차계약서 등)를 준비하시면 됩니다.

유의사항:
- 이사 후 14일 이내에 전입신고를 해야 하며, 미이행 시 과태료가 부과될 수 있습니다.
- 가족이 함께 이사했을 경우 한 명이 일괄 신청할 수 있습니다.

더 궁금한 점이 있으시면 추가로 문의해 주세요!
```

## 검증

6주차 성공 판정 기준 중 최소 3개를 실제 실행 결과로 확인했습니다.

| 케이스 | 확인한 내용 |
|--------|-------------|
| 정상 케이스 | `examples/input_1.txt`에서 `RequestClassifierTool -> AgencyRoutingTool` 흐름으로 최종 답변이 생성되는지 확인 |
| Tool 실패 케이스 | `RegionNormalizeTool`에 서비스키가 없을 때 `REGION_API_FAILED` 구조화 에러를 반환하는지 확인 |
| 종료 조건 케이스 | `examples/input_2.txt`처럼 추가 질문이 필요한 경우에도 같은 세션에서 이어서 처리하되, 전체 실행이 `max_turns = 6`과 실행 시간 제한 안에서 종료되는지 확인 |

- 정상 케이스:
  `회사에서 퇴직금을 안 줬는데 어디에 신고해야 하나요?` 입력 시 `RequestClassifierTool`이 `issue_resolution / labor`로 분류하고, `AgencyRoutingTool`이 고용노동부 채널을 반환한 뒤 최종 답변이 생성되었습니다.

  실행 명령:

  ```bash
  ./.venv/bin/python -B src/main.py examples/input_1.txt
  ```

  출력 예시:

  ```text
  퇴직금을 지급하지 않는 회사에 대해 신고하려면 아래 방법을 이용하실 수 있습니다.

  - 소관 기관: 고용노동부(관할 지방고용노동관서)
  - 주요 접수 채널:
    - 고용노동부 고객상담센터 1350
    - 고용노동부 홈페이지 민원/진정 접수
    - 지방고용노동관서 방문 접수
  ```

- Tool 실패 케이스:
  `RegionNormalizeTool`은 공공데이터포털 서비스키가 없을 때 mock로 넘어가지 않고 `REGION_API_FAILED`를 반환하도록 구현했습니다. 즉 Tool 실패가 발생해도 성공 결과처럼 처리하지 않고, 구조화된 에러로 명시합니다.

  실행 명령:

  ```bash
  ./.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from tools.RegionNormalizeTool import RegionNormalizeTool; print(RegionNormalizeTool('서울 강남구'))"
  ```

  출력 예시:

  ```text
  {'ok': False, 'data': None, 'error': {'code': 'REGION_API_FAILED', 'message': '공공데이터포털 서비스키가 없어 RegionNormalizeTool API를 호출할 수 없습니다.'}}
  ```

- 종료 조건 케이스:
  `이사했는데 전입신고는 어디서 어떻게 하나요?` 입력 시 오케스트레이터가 지역 추가 질문을 생성하고, 사용자가 `서울시 강남구`를 입력하면 같은 세션에서 다음 단계가 이어집니다. 이 흐름은 `max_turns = 6`, 실행 시간 60초, 반복 Tool 호출 방지 규칙 안에서 종료되도록 구성했습니다.

  실행 명령:

  ```bash
  printf '서울시 강남구\n' | ./.venv/bin/python -B src/main.py examples/input_2.txt
  ```

  출력 예시:

  ```text
  전입신고는 통상적으로 새로운 거주지 관할 행정복지센터(주민센터)나 전자정부 민원포털(정부24)에서 할 수 있습니다.

  혹시 이사하신 동네나 주소를 말씀해주실 수 있나요?
  > 서울시 강남구로 이사하신 경우, 전입신고는 아래 두 가지 방법 중 선택해서 하실 수 있습니다.

  ① 정부24(온라인) 신청
  ② 전입지(강남구) 주민센터 방문
  ```

- 추가 확인:
  `"도와주세요"` 같은 모호한 입력에서는 하나로 단정하지 않고 추가 질문을 생성하는 것도 확인했습니다. 이 동작은 6주차 성공 판정 기준 중 “확신도가 낮으면 추가 질문 생성”과 연결됩니다.

## 로그 분석

- 설계서에서 예상한 흐름은 `RequestClassifierTool -> 필요 시 Region 관련 질문 또는 정규화 -> AgencyRoutingTool -> RequirementAndDraftTool -> final` 이었습니다.
- 실제 로그에서도 `input_1`은 `RequestClassifierTool -> AgencyRoutingTool`, `input_2`는 `RequestClassifierTool -> 추가 질문 -> AgencyRoutingTool` 흐름으로 동작해 설계와 크게 다르지 않았습니다.
- 예상과 다른 점은 `input_1`에서 `RequirementAndDraftTool`까지 가지 않고 `AgencyRoutingTool` 단계에서 바로 최종 답변이 생성된 경우가 있었다는 점입니다. 현재 오케스트레이터는 질문 의도와 준비 정보 필요성을 보고 Tool 호출 수를 줄이는 방향으로도 동작합니다.
- 불필요한 반복 Tool 호출은 현재 로그 기준으로 나타나지 않았고, 같은 Tool/argument 반복 호출 방지 로직도 넣어두었습니다.
- Tool 실패가 발생하면 재시도만 반복하지 않고 일반 안내로 마무리하도록 설계했고, 종료 조건으로 `max_turns`, `time limit`, `repeated tool call`, `tool failure`를 두었습니다.

## 실행 방법

### 1. 의존성 설치

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일에 OpenAI API 키를 넣습니다.

```env
OPENAI_API_KEY=sk-proj-...
PUBLIC_DATA_SERVICE_KEY=...
```

### 3. 실행

```bash
./.venv/bin/python -B src/main.py examples/input_1.txt
```

추가 질문이 필요한 경우에는 실행이 끝나지 않고 같은 콘솔에서 `>` 프롬프트가 나타납니다.  
그 아래에 답을 입력하면 같은 대화 맥락으로 다음 단계가 이어집니다.

## 현재 한계

- `AgencyRoutingTool`, `RequirementAndDraftTool`는 아직 실제 공공 API 연동이 아니라 mock 데이터 기반입니다.
- `PublicServiceSearchTool`은 아직 구현하지 않았습니다.
- 현재는 MVP 단계이므로 공공서비스 검색보다 민원 라우팅과 절차 안내에 집중하고 있습니다.
