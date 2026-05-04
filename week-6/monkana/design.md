# 행정 민원 내비게이터 Agent 설계서

## 1. 개요·목적

### 해결하려는 문제

대한민국의 행정 업무는 중앙부처, 지방자치단체, 공공기관, 교육청 등으로 나뉘어 있어 일반 사용자가 자신의 문의, 신고, 신청을 어느 기관에 해야 하는지 판단하기 어렵다. 이 Agent는 사용자의 자연어 요청을 분석해 소관 가능성이 높은 기관, 접수 채널, 필요한 정보, 민원 문안 초안을 안내한다.

예를 들어 사용자가 “퇴직금을 못 받았는데 어디에 신고해야 하나요?”라고 물으면 노동 민원으로 분류하고, 고용노동부 또는 관할 지방고용노동관서에 문의할 가능성이 높다고 안내한다. 반대로 “도로에 큰 구멍이 났어요”라고 하면 지역 정보와 긴급성을 확인한 뒤, 지자체 도로관리부서 또는 긴급 신고 채널을 우선 안내한다.

### 타깃 사용자

행정기관 구조, 민원 접수 채널, 공공서비스 신청 절차를 잘 모르는 일반 시민.

### 왜 Agent여야 하는가

이 문제는 단일 LLM 호출이나 단순 RAG만으로 해결하기 어렵다. 사용자의 요청마다 필요한 판단 순서와 Tool 조합이 달라지기 때문이다.

예를 들어 “회사에서 퇴직금을 안 줬어요”는 노동 민원으로 분류한 뒤, 소관 기관 후보를 찾고, 사업장명·근무기간·퇴사일·미지급 금액 같은 필요 정보를 확인한 다음 민원 문안 초안을 작성해야 한다. 반면 “동네 도로에 큰 구멍이 생겼어요”는 지역 정보와 긴급성을 먼저 확인하고, 지자체 도로관리 부서 또는 긴급 신고 채널을 안내해야 한다. 또 “아이가 태어났는데 받을 수 있는 지원이 있나요?”는 신고 민원이 아니라 공공서비스·정부 혜택 탐색이 핵심이므로 공공서비스 검색 Tool을 호출해야 한다.

따라서 모든 요청을 고정된 `분류 → 검색 → 답변` 순서로 처리하면 부족하다. 이 Agent는 요청을 보고 어떤 Tool이 필요한지, 어떤 순서로 실행할지, 추가 질문이 필요한지를 동적으로 결정해야 한다.

---

## 2. 사용자 시나리오

### Persona

- 이름: 김민지
- 역할: 29세 직장인
- 목적: 행정기관에 문의하거나 민원을 넣어야 하는 상황에서 어디에, 어떤 방식으로, 어떤 자료를 준비해 요청해야 하는지 알고 싶다.
- 특징: 법률 용어와 부처명을 잘 모르고, 국민신문고·정부24·지자체 민원 창구의 차이를 명확히 알지 못한다.

### 대표 요청 1

> “회사에서 퇴직금을 안 줬는데 어디에 신고해야 하나요?”

단일 Tool 한 번으로 끝나지 않는 이유: 노동 민원인지 분류해야 하고, 소관 기관을 찾아야 하며, 사용자가 준비해야 할 증빙자료와 민원 문안까지 정리해야 한다.

예상 흐름:

```text
RequestClassifierTool → AgencyRoutingTool → RequirementAndDraftTool
```

### 대표 요청 2

> “우리 동네 도로에 큰 구멍이 생겼어요. 어디에 말해야 하나요?”

단일 Tool 한 번으로 끝나지 않는 이유: 도로 안전 민원인지 판단해야 하고, 위치 정보를 표준화해야 하며, 긴급 위험 여부에 따라 일반 민원 접수와 긴급 신고 안내가 달라진다.

예상 흐름:

```text
RequestClassifierTool → RegionNormalizeTool → AgencyRoutingTool → RequirementAndDraftTool
```

### 대표 요청 3

> “아이가 태어났는데 받을 수 있는 정부 지원이 있나요?”

단일 Tool 한 번으로 끝나지 않는 이유: 이 요청은 신고 민원이 아니라 공공서비스 탐색 요청이다. 사용자의 지역, 가족 상태, 자녀 출생 여부에 따라 검색해야 할 서비스와 안내 내용이 달라진다.

예상 흐름:

```text
RequestClassifierTool → RegionNormalizeTool → PublicServiceSearchTool → RequirementAndDraftTool
```

---

## 3. 기능 요구사항

### Must-have

1. 사용자의 자연어 요청을 입력받아 민원 또는 공공서비스 요청 유형을 분류한다.
   - 입력: “퇴직금을 못 받았어요”, “도로가 파였어요”, “출산 지원금이 있나요?”
   - 출력: `노동`, `도로안전`, `복지·출산지원`, `식품위생`, `주거`, `기타/불명확`

2. 요청 유형과 키워드를 바탕으로 소관 가능성이 높은 기관 후보를 제시한다.
   - 입력: 민원 유형, 핵심 키워드, 지역 정보
   - 출력: 기관명, 담당 단위, 추천 이유, 접수 채널, 확신도

3. 지역 정보가 필요한 요청이면 사용자가 입력한 지역명을 표준 행정구역 정보로 변환한다.
   - 입력: “강남구 역삼동”, “성남 분당”, “부산 해운대”
   - 출력: 시도, 시군구, 읍면동, 법정동 코드 후보

4. 사용자가 민원 제출에 필요한 정보를 빠뜨렸으면 바로 단정하지 않고 추가 질문을 생성한다.
   - 예: 퇴직금 민원에는 사업장명, 근무기간, 퇴사일, 미지급 금액, 급여명세서 여부를 확인한다.

5. 최종 응답에는 요청 유형에 따라 필요한 핵심 정보를 포함한다.
   - 공통: 민원 유형 또는 서비스 유형, 접수·신청 채널, 준비할 정보
   - 민원형 요청: 소관 기관 후보, 민원 문안 초안
   - 공공서비스 탐색 요청: 서비스명, 제공기관, 신청 조건 요약

### Nice-to-have

1. 관련 법령이나 공공서비스 근거를 참고 정보로 보여준다.

2. 사용자의 이전 상담 이력을 바탕으로 같은 지역이나 같은 기관에 대한 후속 질문을 더 짧게 처리한다.

3. 민원 문안 초안을 사용자의 말투에 맞춰 공손한 공식 문체로 변환한다.

---

## 4. Agent 패턴 선택과 근거

### 선택한 패턴

**Plan-and-Execute + ReAct 혼합**

### 선택 근거

이 Agent는 먼저 사용자의 요청을 보고 전체 처리 계획을 세워야 한다. 예를 들어 민원 신고인지, 단순 문의인지, 공공서비스 검색인지, 긴급 안전 상황인지부터 구분해야 한다. 이 부분은 Plan-and-Execute에 가깝다.

이후 각 Tool의 결과를 보고 다음 행동을 바꿔야 한다. 예를 들어 소관 기관 후보가 2개 이상이면 추가 질문을 해야 하고, 지역 정보가 없으면 지역 확인 질문을 해야 하며, 긴급 안전 민원으로 판단되면 일반 민원 안내보다 즉시 신고 안내를 우선해야 한다. 이 부분은 ReAct의 `Thought → Action → Observation` 흐름에 가깝다.

### 루프 구조

```text
1. 사용자 요청 수신
2. RequestClassifierTool로 요청 유형, 의도, 긴급성, 필요한 정보 추출
3. Agent가 처리 계획 수립
   - 지역 정보가 필요한가?
   - 공공서비스 검색이 필요한가?
   - 민원 문안 작성이 필요한가?
   - 긴급 안내가 필요한가?
4. 필요한 경우 RegionNormalizeTool 호출
5. 민원 라우팅이면 AgencyRoutingTool 호출
6. 혜택·지원금 탐색이면 PublicServiceSearchTool 호출
7. RequirementAndDraftTool로 필요 정보와 민원 문안 초안 생성
8. 정보가 부족하면 추가 질문 생성
9. 충분하면 최종 응답 생성
10. max_steps 또는 stop 조건에 도달하면 종료
```

---

## 5. 동작 명세

### 입력 스키마

입력은 자연어 문자열을 기본으로 받는다. 선택적으로 구조화 필드를 함께 받을 수 있다.

```json
{
  "user_message": "회사에서 퇴직금을 안 줬는데 어디에 신고해야 하나요?",
  "user_region": "서울 강남구",
  "user_profile": {
    "age": null,
    "job_status": "퇴사자",
    "household": null
  },
  "options": {
    "want_draft": true,
    "allow_memory": false
  }
}
```

### 출력 스키마

최종 응답은 자연어 설명과 구조화 정보를 함께 제공한다.

```json
{
  "summary": "퇴직금 미지급 관련 노동 민원으로 보입니다.",
  "response_type": "complaint_guide",
  "category": "labor",
  "urgency": "normal",
  "agency_candidates": [
    {
      "name": "고용노동부",
      "local_unit": "지방고용노동관서",
      "reason": "퇴직금, 임금체불, 근로관계 분쟁 관련 민원 가능성이 높음",
      "confidence": 0.86
    }
  ],
  "services": [],
  "channels": ["국민신문고", "고용노동부 민원", "지방고용노동관서 상담"],
  "required_info": ["사업장명", "근무기간", "퇴사일", "미지급 금액", "근로계약서 또는 급여명세서"],
  "missing_info": ["사업장명", "퇴사일", "미지급 금액"],
  "next_steps": [
    "사업장명과 퇴사일을 정리한다.",
    "근로계약서, 급여명세서, 입금내역 등 증빙자료를 준비한다.",
    "고용노동부 민원 또는 국민신문고를 통해 진정을 접수한다."
  ],
  "draft": "안녕하세요. 저는 ○○회사에서 ○년 ○월부터 ○년 ○월까지 근무하였으나 퇴사 후 퇴직금을 지급받지 못했습니다. 관련 사실 확인 및 필요한 조치를 요청드립니다."
}
```

`agency_candidates`, `services`, `draft`는 요청 경로에 따라 선택적으로 채워진다. 민원형 요청이면 `agency_candidates`와 `draft`가 채워지고, 공공서비스 탐색 요청이면 `services`와 신청 조건 요약이 채워지며 `draft`는 `null`일 수 있다.

### 정상 흐름 예시

사용자 요청:

> “회사에서 퇴직금을 안 줬는데 어디에 신고해야 하나요?”

Trace 예시:

```text
Step 1
Thought: 사용자의 요청은 임금·퇴직금 관련 민원으로 보인다. 먼저 요청 유형과 필요한 정보를 분류한다.
Action: RequestClassifierTool
Input: {"message": "회사에서 퇴직금을 안 줬는데 어디에 신고해야 하나요?"}
Observation: {"category": "labor", "intent": "complaint", "urgency": "normal", "keywords": ["퇴직금", "미지급", "신고"], "confidence": 0.91}

Step 2
Thought: 노동 민원으로 분류되었으므로 소관 기관 후보를 찾는다.
Action: AgencyRoutingTool
Input: {"category": "labor", "intent": "complaint", "keywords": ["퇴직금", "미지급"]}
Observation: {"candidates": [{"agency": "고용노동부", "unit": "지방고용노동관서", "confidence": 0.88}], "channels": ["국민신문고", "고용노동부 민원"]}

Step 3
Thought: 사용자가 실제 민원을 넣으려면 필요한 정보와 증빙자료를 알아야 한다.
Action: RequirementAndDraftTool
Input: {"category": "labor", "agency": "고용노동부", "facts": {"issue": "퇴직금 미지급"}}
Observation: {"required_info": ["사업장명", "근무기간", "퇴사일", "미지급 금액", "근로계약서", "급여명세서"], "missing_info": ["사업장명", "근무기간", "퇴사일", "미지급 금액"], "draft_available": true}

Step 4
Thought: 필수 정보가 부족하지만, 일반 안내와 초안 형태는 제공할 수 있다. 추가로 필요한 정보를 질문한다.
Action: FinalAnswer
Observation: 최종 응답 생성
```

최종 응답 예시:

```text
퇴직금 미지급은 노동 민원으로 볼 가능성이 높습니다. 우선 고용노동부 또는 관할 지방고용노동관서에 문의·진정 접수를 검토하시면 됩니다.

접수 전 준비하면 좋은 정보는 사업장명, 근무기간, 퇴사일, 미지급 금액, 근로계약서, 급여명세서, 입금내역입니다.

민원 문안 초안:
안녕하세요. 저는 ○○회사에서 ○년 ○월부터 ○년 ○월까지 근무한 뒤 퇴사하였으나, 퇴직금을 지급받지 못했습니다. 이에 퇴직금 미지급 여부 확인 및 필요한 조치를 요청드립니다.
```

### 예외 흐름

#### Tool 실패

Tool은 예외를 직접 던지지 않고 다음 형식으로 실패 결과를 반환한다.

```json
{
  "error": "API_TIMEOUT",
  "detail": "PublicServiceSearchTool request timed out",
  "fallback_available": true
}
```

Agent는 실패 시 다음 순서로 대응한다.

```text
1. 같은 Tool을 동일 파라미터로 1회만 재시도
2. 재시도 실패 시 Mock 데이터 또는 일반 안내로 fallback
3. 최종 응답에 "정확한 최신 정보 확인 필요" 표시
```

#### 소관 기관이 불명확한 경우

후보를 2~3개 제시하고, 추가 질문을 한다.

```text
이 요청은 주거 민원과 소비자 분쟁이 모두 가능해 보입니다.
정확히 안내하려면 다음 중 어떤 상황인지 알려주세요.

1. 임대차계약, 보증금, 월세 관련 문제인가요?
2. 업체와의 계약·환불·서비스 불만 문제인가요?
3. 범죄 피해나 사기 신고에 가까운 문제인가요?
```

#### 긴급 안전 상황

도로 붕괴, 화재, 인명 피해, 범죄 등 즉시 위험이 있는 경우 일반 민원 안내보다 긴급 신고 안내를 우선한다.

```text
인명 피해나 즉시 위험이 있다면 민원 접수보다 119 또는 112 신고가 우선입니다.
그 후 위치, 사진, 발견 시각을 정리해 지자체 또는 안전 관련 민원 채널에 신고할 수 있습니다.
```

#### 권한 부족 또는 개인정보 문제

Agent는 사용자를 대신해 실제 민원을 접수하지 않는다. 인증, 개인정보 입력, 증빙자료 첨부, 법적 제출 행위는 사용자가 직접 수행해야 한다.

### 종료 조건

```text
max_steps = 6

종료 조건:
1. `category`와 `channels`를 확보했고, 민원형이면 `agency_candidates` 또는 `required_info`, 공공서비스형이면 `services` 또는 `required_info`를 추가 확보
2. 사용자의 요청이 단순 문의이고 소관기관 후보가 충분히 명확함
3. 필수 정보가 부족해 추가 질문이 필요한 상태
4. 동일 Tool을 같은 파라미터로 2회 이상 호출하려는 경우
5. 긴급 상황으로 판단되어 즉시 신고 안내를 완료한 경우
```

---

## 6. Tool 명세

| Tool 이름 | 목적 | 입력 스키마 | 출력 스키마 | 실패 시 반환 | 사용 조건 |
|---|---|---|---|---|---|
| `RequestClassifierTool` | 사용자 요청을 민원 유형, 의도, 긴급성으로 분류한다 | `{ "message": string }` | `{ "category": string, "intent": string, "urgency": string, "keywords": string[], "needs_region": boolean, "confidence": number }` | `{ "error": "CLASSIFICATION_FAILED", "detail": string }` | 모든 요청의 첫 단계에서 사용 |
| `RegionNormalizeTool` | 사용자가 입력한 지역명을 표준 행정구역명 또는 법정동 코드로 변환한다 | `{ "region_text": string }` | `{ "sido": string, "sigungu": string, "dong": string, "legal_dong_code": string, "confidence": number }` | `{ "error": "REGION_NOT_FOUND", "detail": string }` | 지역별 소관 부서, 지자체 민원, 도로·안전 민원일 때 사용 |
| `AgencyRoutingTool` | 민원 유형과 키워드를 바탕으로 소관 기관 후보를 찾는다 | `{ "category": string, "intent": string, "keywords": string[], "region": object }` | `{ "candidates": object[], "channels": string[], "routing_reason": string }` | `{ "error": "NO_ROUTE_FOUND", "detail": string, "fallback": object[] }` | 신고·문의·민원 접수 요청일 때 사용 |
| `PublicServiceSearchTool` | 출산지원, 청년지원, 복지혜택 등 공공서비스 정보를 검색한다 | `{ "keywords": string[], "region": object, "profile": object }` | `{ "services": object[], "matched_conditions": string[], "need_more_info": string[] }` | `{ "error": "SERVICE_API_FAILED", "detail": string }` | 사용자가 지원금, 혜택, 신청 가능 서비스를 물을 때 사용 |
| `RequirementAndDraftTool` | 민원 제출 또는 서비스 신청에 필요한 정보와 안내 초안을 생성한다 | `{ "category": string, "agency": object, "facts": object, "want_draft": boolean }` | `{ "required_info": string[], "missing_info": string[], "draft": string \| null, "application_guide": string \| null, "cautions": string[] }` | `{ "error": "DRAFT_FAILED", "detail": string }` | 소관 기관 후보 또는 서비스 후보가 정해진 뒤 사용 |

### Tool별 데이터 후보

- `RegionNormalizeTool`: 행정안전부_행정표준코드_법정동코드 API
- `PublicServiceSearchTool`: 행정안전부_대한민국 공공서비스(혜택) 정보 API
- `AgencyRoutingTool`: 국민권익위원회_민원빅데이터_분석정보_API_2025 + Mock 라우팅 데이터
- `RequirementAndDraftTool`: Mock 필요 정보 데이터 + LLM 문안 생성

---

## 7. 데이터셋

### 데이터 출처

| 데이터 | 사용 Tool | 용도 | 실제 사용 여부 |
|---|---|---|---|
| 행정안전부_행정표준코드_법정동코드 | `RegionNormalizeTool` | 지역명 정규화, 법정동 코드 조회 | 실제 API 후보 |
| 행정안전부_대한민국 공공서비스(혜택) 정보 | `PublicServiceSearchTool` | 공공서비스·정부 혜택 검색 | 실제 API 후보 |
| 국민권익위원회_민원빅데이터_분석정보_API_2025 | `AgencyRoutingTool` 보조 | 키워드 기반 기관별 민원 경향 참고 | 실제 API 후보 |
| `examples/agency_routing_mock.json` | `AgencyRoutingTool` | 민원 유형별 소관 기관 매핑 | Mock |
| `examples/requirements_mock.json` | `RequirementAndDraftTool` | 민원 유형별 필요 정보·증빙자료 | Mock |

### 참고 API 설명

- 행정안전부_대한민국 공공서비스(혜택) 정보 API는 정부 부처, 지방자치단체, 공공기관, 교육청 등이 제공하는 공공서비스와 정부 혜택 정보의 목록 및 상세 내용을 제공한다. 데이터포맷은 JSON+XML이며 REST API다.
- 행정안전부_행정표준코드_법정동코드 API는 법정동코드, 시도코드, 읍면동코드, 지역주소명 등을 조회할 수 있다. 데이터포맷은 JSON+XML이며 REST API다.
- 국민권익위원회_민원빅데이터_분석정보_API_2025는 키워드 기반 기관별 민원 건수 정보와 키워드 기반 법령 정보를 조회할 수 있다. 데이터포맷은 JSON이다.
- `public-apis-4Kr` 목록은 한국에서 활용 가능한 공개 API를 정리한 색인으로 사용한다.

### Mock 데이터 1: `examples/agency_routing_mock.json`

```json
[
  {
    "category": "labor",
    "keywords": ["임금체불", "퇴직금", "부당해고", "근로계약서"],
    "primary_agency": "고용노동부",
    "local_unit": "지방고용노동관서",
    "channels": ["국민신문고", "고용노동부 민원", "방문 상담"],
    "required_info": ["사업장명", "근무기간", "퇴사일", "미지급 금액", "근로계약서", "급여명세서"]
  },
  {
    "category": "road_safety",
    "keywords": ["도로 파임", "포트홀", "싱크홀", "보도블럭 파손", "가로등 고장"],
    "primary_agency": "지방자치단체",
    "local_unit": "시군구청 도로관리부서",
    "channels": ["안전신문고", "지자체 민원", "긴급 상황 시 119"],
    "required_info": ["정확한 위치", "사진", "발견 시각", "위험 정도"]
  },
  {
    "category": "food_safety",
    "keywords": ["식당 위생", "이물질", "식중독", "불량식품"],
    "primary_agency": "식품의약품안전처 또는 지방자치단체",
    "local_unit": "시군구청 위생과",
    "channels": ["국민신문고", "지자체 민원"],
    "required_info": ["상호명", "주소", "방문일시", "사진", "상황 설명"]
  },
  {
    "category": "housing",
    "keywords": ["전세사기", "보증금", "월세", "임대차", "집주인"],
    "primary_agency": "국토교통부 또는 지방자치단체",
    "local_unit": "주거복지센터 또는 구청 주택과",
    "channels": ["정부24", "국민신문고", "지자체 민원"],
    "required_info": ["임대차계약서", "주소", "보증금 금액", "계약기간", "피해 내용"]
  },
  {
    "category": "birth_support",
    "keywords": ["출산지원", "아동수당", "부모급여", "출생신고", "육아지원"],
    "primary_agency": "보건복지부 또는 지방자치단체",
    "local_unit": "읍면동 주민센터",
    "channels": ["정부24", "복지로", "주민센터"],
    "required_info": ["자녀 출생일", "보호자 정보", "거주지", "가구 상황"]
  }
]
```

### Mock 데이터 2: `examples/requirements_mock.json`

```json
{
  "labor": {
    "required_info": ["사업장명", "근무기간", "퇴사일", "미지급 금액", "근로계약서", "급여명세서", "입금내역"],
    "draft_template": "안녕하세요. 저는 {company}에서 {start_date}부터 {end_date}까지 근무하였으나 퇴사 후 퇴직금을 지급받지 못했습니다. 관련 사실 확인 및 필요한 조치를 요청드립니다."
  },
  "road_safety": {
    "required_info": ["정확한 위치", "사진", "발견 시각", "차량 또는 보행자 위험 여부"],
    "draft_template": "{location} 부근 도로에 큰 파임 또는 구멍이 있어 차량 및 보행자 안전사고 위험이 있습니다. 현장 확인 및 보수를 요청드립니다."
  },
  "food_safety": {
    "required_info": ["상호명", "주소", "방문일시", "문제 상황", "사진 또는 영수증"],
    "draft_template": "{store_name}에서 위생상 문제가 의심되는 상황을 경험했습니다. 방문일시는 {visited_at}이며, 관련 사진을 첨부합니다. 확인을 요청드립니다."
  },
  "birth_support": {
    "required_info": ["자녀 출생일", "거주지", "보호자 정보", "가구 상황", "소득 또는 지원 자격 확인 필요 여부"],
    "application_guide": "거주지 기준으로 정부24, 복지로, 주민센터에서 부모급여, 아동수당, 출산지원금 대상 여부를 확인하고 신청한다."
  }
}
```

### 데이터 특성 요약

공공데이터포털 기반 API는 대체로 인증키가 필요하다. 따라서 MVP에서는 실제 API 호출이 실패하거나 인증키가 없을 때를 대비해 Mock 데이터를 함께 사용한다. 실제 제출물에는 `examples/agency_routing_mock.json`, `examples/requirements_mock.json` 두 개 파일을 선택 제출물로 포함할 수 있다.

---

## 8. 성공 판정 기준

| 번호 | 성공 기준 | 판정 방식 |
|---:|---|---|
| 1 | “퇴직금을 못 받았다” 유형의 요청이 들어오면 `RequestClassifierTool → AgencyRoutingTool → RequirementAndDraftTool` 순서로 호출한다 | 예/아니오 |
| 2 | 도로 파임, 싱크홀, 가로등 고장처럼 위치가 필요한 요청이면 `RegionNormalizeTool`을 호출하거나 위치 추가 질문을 생성한다 | 예/아니오 |
| 3 | 출산지원, 청년지원, 복지혜택처럼 공공서비스 탐색 요청이면 `PublicServiceSearchTool`을 호출하고 최종 응답에 서비스명, 제공기관, 신청 조건 요약을 포함한다 | 예/아니오 |
| 4 | 소관 기관 확신도가 낮거나 후보가 2개 이상이면 하나로 단정하지 않고 후보와 추가 질문을 함께 제시한다 | 예/아니오 |
| 5 | 모든 정상 시나리오는 6 step 이내에 종료하고, Tool 실패 시 재시도 또는 Mock fallback 결과를 표시한다 | 예/아니오 |

---

## 9. 제약·확장

### 현재 설계의 한계

이 Agent는 사용자를 대신해 실제 민원을 접수하지 않는다. 실제 접수에는 본인 인증, 개인정보 입력, 증빙자료 첨부, 법적 책임 문제가 있기 때문이다. 따라서 이 Agent의 역할은 “접수 전 안내”로 제한한다.

또한 소관 기관 판단은 최종 법적 판단이 아니다. 같은 문제라도 지역, 사건의 구체적 사실, 관련 법령, 기관 내부 업무분장에 따라 실제 담당 기관이 달라질 수 있다. 그래서 최종 응답에는 “소관 가능성이 높은 기관 후보”와 “확인 필요 사항”을 함께 제시해야 한다.

공개 API만으로 모든 민원을 정확히 라우팅하기 어렵다. 정부24 공공서비스 API는 혜택과 서비스 탐색에는 적합하지만, 모든 민원 유형을 소관 부서까지 완벽히 매핑하는 용도는 아니다. 따라서 MVP에서는 소관 기관 매핑과 필요 서류 판단에 Mock 데이터를 함께 사용한다.

### Multi-Agent로 확장한다면

| Agent | 역할 |
|---|---|
| `OrchestratorAgent` | 사용자 요청을 받아 전체 계획을 세우고 Worker를 호출 |
| `ComplaintClassifierAgent` | 요청을 노동, 주거, 안전, 복지, 식품위생 등으로 분류 |
| `RoutingAgent` | 소관 기관 후보와 접수 채널 탐색 |
| `PublicServiceAgent` | 정부 혜택·지원금·공공서비스 검색 |
| `DraftingAgent` | 민원 문안 초안과 필요 서류 체크리스트 작성 |
| `SafetyGuardAgent` | 긴급 상황, 개인정보, 법률 단정 표현을 점검 |

현재 과제에서는 단일 Agent 설계가 목적이므로 Multi-Agent는 확장 지점으로만 둔다.

### 장기 상태·메모리가 필요한 시나리오

장기 메모리는 다음 상황에서 유용하다.

1. 사용자가 자주 사용하는 지역
   - 예: “저번처럼 강남구 기준으로 봐줘.”

2. 이전에 상담한 민원 유형
   - 예: “지난번 퇴직금 건 이어서 문안 다시 써줘.”

3. 선호 접수 채널
   - 예: “나는 방문 말고 온라인 접수 위주로 알려줘.”

단, 주민등록번호, 상세 주소, 회사명, 계좌정보, 민감한 피해 사실 등은 장기 저장하지 않는다. 저장이 필요하면 사용자 동의를 받은 뒤 최소 정보만 저장한다.

---

## 참고 링크

- 과제 템플릿: https://github.com/bjc1102/ai-agent-repo/blob/main/week-6/TASK.md
- Agent 패턴 참고자료: https://github.com/bjc1102/ai-agent-repo/blob/main/week-6/README.md
- 한국 공개 API 목록: https://github.com/yybmion/public-apis-4Kr
- 행정안전부_대한민국 공공서비스(혜택) 정보: https://www.data.go.kr/data/15113968/openapi.do
- 행정안전부_행정표준코드_법정동코드: https://www.data.go.kr/data/15077871/openapi.do
- 국민권익위원회_민원빅데이터_분석정보_API_2025: https://www.data.go.kr/data/15143948/openapi.do
