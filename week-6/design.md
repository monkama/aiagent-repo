# 행정 민원 내비게이터 Agent 설계서

## 1. 개요·목적

### 해결하려는 문제

대한민국의 행정 업무는 중앙부처, 지방자치단체, 공공기관, 교육청 등으로 나뉘어 있어 일반 사용자가 자신의 문의, 신고, 신청을 어느 기관에 해야 하는지 판단하기 어렵다. 또한 사용자의 요청은 모두 같은 성격이 아니다. 어떤 요청은 정부 지원금이나 복지 혜택을 찾는 문제이고, 어떤 요청은 피해·불편·위험을 해결하기 위한 신고 문제이며, 어떤 요청은 전입신고나 증명서 발급처럼 사용자가 직접 처리해야 하는 행정 절차 문제이다.

이 Agent는 사용자의 자연어 요청을 분석해 먼저 요청의 성격을 파악한다. 이후 요청이 속한 행정 분야와 긴급도를 판단하고, 필요한 Tool을 동적으로 호출해 소관 가능성이 높은 기관, 접수 채널, 필요한 정보, 민원 문안 초안을 안내한다.

예를 들어 사용자가 “퇴직금을 못 받았는데 어디에 신고해야 하나요?”라고 물으면 임금·퇴직금 관련 문제로 보고, 고용노동부 또는 관할 지방고용노동관서에 문의할 가능성이 높다고 안내한다. 반대로 “아이가 태어났는데 받을 수 있는 지원이 있나요?”라고 묻는 경우에는 신고 민원이 아니라 정부 지원이나 복지 혜택을 찾는 요청으로 보고, 출산지원·아동수당·부모급여 같은 공공서비스 정보를 우선 확인한다.

또한 “도로에 큰 구멍이 났어요”처럼 안전과 관련된 요청은 지역 정보와 긴급성을 함께 확인해야 한다. 일반적인 도로 파손 신고라면 지자체 도로관리부서나 안전신문고 안내가 적절할 수 있지만, 즉시 사고 위험이 있는 경우에는 일반 민원 접수보다 112, 119, 지자체 당직실 등 긴급 대응 채널 안내가 우선된다.

### 타깃 사용자

행정기관 구조, 민원 접수 채널, 공공서비스 신청 절차를 잘 모르는 일반 시민을 대상으로 한다.

대표 사용자는 다음과 같다.

- 어느 기관에 문의하거나 신고해야 하는지 모르는 사람
- 정부 지원금, 복지 혜택, 공공서비스를 받을 수 있는지 알고 싶은 사람
- 전입신고, 출생신고, 발급, 등록, 변경 같은 행정 절차를 어디서 어떻게 처리해야 하는지 알고 싶은 사람
- 문제가 발생했지만 중앙부처, 지자체, 공공기관 중 어디에 연락해야 할지 모르는 사람

### 왜 Agent여야 하는가

이 문제는 단일 LLM 호출이나 단순 RAG만으로 해결하기 어렵다. 사용자의 요청마다 필요한 판단 순서와 Tool 조합이 달라지기 때문이다.

예를 들어 “회사에서 퇴직금을 안 줬어요”는 먼저 노동 관련 문제인지 판단해야 한다. 이후 소관 기관 후보를 찾고, 사업장명·근무기간·퇴사일·미지급 금액 같은 필요 정보를 확인한 다음 민원 문안 초안을 작성해야 한다. 이 경우에는 요청 분류, 기관 탐색, 준비 정보 정리 Tool이 순서대로 필요하다.

반면 “아이가 태어났는데 받을 수 있는 지원이 있나요?”는 신고나 분쟁 해결이 아니라 공공서비스·정부 혜택 탐색이 핵심이다. 사용자의 거주 지역, 자녀 출생일, 가구 상황에 따라 받을 수 있는 서비스가 달라질 수 있으므로 지역 정규화, 공공서비스 검색, 추가 정보 확인 Tool이 필요하다.

또 “이사했는데 전입신고는 어디서 어떻게 하나요?”는 피해 해결이 아니라 행정 절차 문의이다. 이 경우에는 전입신고를 처리할 수 있는 기관, 접수 채널, 준비 정보, 절차 안내가 필요하다.

마지막으로 “도로가 무너져서 차들이 위험해요”처럼 긴급성이 높은 요청은 일반적인 민원 안내보다 즉시 신고 안내가 우선되어야 한다. 이 경우 Agent는 긴급 상황으로 판단하고, Tool 호출 결과와 관계없이 112, 119, 지자체 당직실 등 즉시 대응 가능한 채널을 먼저 안내해야 한다.

따라서 모든 요청을 고정된 `분류 → 검색 → 답변` 순서로 처리하면 부족하다. 이 Agent는 사용자의 요청을 보고 요청의 성격, 행정 분야, 긴급도, 지역 정보 필요 여부, 판단 신뢰도를 고려한 뒤, 어떤 Tool이 필요한지, 어떤 순서로 실행할지, 추가 질문이 필요한지, 즉시 답변해야 하는지를 동적으로 결정해야 한다.


## 2. 사용자 시나리오

### Persona

- 이름: 김민지
- 역할: 29세 직장인
- 목적: 행정기관에 문의하거나 민원을 넣어야 하는 상황에서 어디에, 어떤 방식으로, 어떤 자료를 준비해 요청해야 하는지 알고 싶다.
- 특징: 법률 용어와 부처명을 잘 모르고, 국민신문고·정부24·지자체 민원 창구의 차이를 명확히 알지 못한다.

### 대표 요청 1: 문제 발생 문의

> “회사에서 퇴직금을 안 줬는데 어디에 신고해야 하나요?”

분류 결과 예시:

```json
{
  "response_type": "issue_resolution",
  "category": "labor",
  "urgency": "normal"
}
```

단일 Tool 한 번으로 끝나지 않는 이유:  
이 요청은 사용자가 실제 문제를 겪고 있고, 어느 기관에 신고해야 하는지 모르는 상황이다. 따라서 노동 민원인지 분류하고, 소관 기관을 찾은 뒤, 사용자가 준비해야 할 증빙자료와 민원 문안까지 정리해야 한다.

예상 흐름:

```text
RequestClassifierTool → AgencyRoutingTool → RequirementAndDraftTool
```

---

### 대표 요청 2: 행정 절차 문의

> “이사했는데 전입신고는 어디서 어떻게 하나요?”

분류 결과 예시:

```json
{
  "response_type": "administrative_procedure",
  "category": "residence",
  "urgency": "normal"
}
```

단일 Tool 한 번으로 끝나지 않는 이유:  
이 요청은 피해나 신고가 아니라 사용자가 직접 처리해야 하는 행정 절차를 묻는 상황이다. 전입신고는 거주지 기준으로 처리되므로 지역 정보를 확인해야 하고, 접수 가능한 기관과 채널, 준비할 정보, 처리 절차를 함께 안내해야 한다.

예상 흐름:

```text
RequestClassifierTool → RegionNormalizeTool → AgencyRoutingTool → RequirementAndDraftTool
```

---

### 대표 요청 3: 공공서비스 문의

> “아이가 태어났는데 받을 수 있는 정부 지원이 있나요?”

분류 결과 예시:

```json
{
  "response_type": "public_service_inquiry",
  "category": "birth_support",
  "urgency": "normal"
}
```

단일 Tool 한 번으로 끝나지 않는 이유:  
이 요청은 신고 민원이 아니라 공공서비스 탐색 요청이다. 사용자의 지역, 가족 상태, 자녀 출생 여부에 따라 받을 수 있는 서비스와 신청 조건이 달라지므로 공공서비스 검색과 추가 정보 확인이 필요하다.

예상 흐름:

```text
RequestClassifierTool → RegionNormalizeTool → PublicServiceSearchTool → RequirementAndDraftTool
```

---

### 추가 예시: 긴급성이 있는 문제 발생 문의

> “우리 동네 도로에 큰 구멍이 생겨서 차들이 위험해요.”

분류 결과 예시:

```json
{
  "response_type": "issue_resolution",
  "category": "road_safety",
  "urgency": "high"
}
```

단일 Tool 한 번으로 끝나지 않는 이유:  
이 요청은 도로 안전 문제에 대한 신고 문의이면서, 사고 위험이 있을 수 있는 상황이다. 따라서 도로 안전 민원인지 분류하고, 위치 정보를 표준화한 뒤, 소관 기관과 접수 채널을 찾고, 긴급 안내가 필요한지 함께 판단해야 한다.

예상 흐름:

```text
RequestClassifierTool → RegionNormalizeTool → AgencyRoutingTool → RequirementAndDraftTool
```

긴급도가 `emergency`로 판단되는 경우에는 일반 민원 접수 안내보다 112, 119, 지자체 당직실 등 즉시 대응 가능한 신고 채널 안내를 우선한다.

---

## 3. 기능 요구사항

### Must-have

1. 사용자의 자연어 요청을 입력받아 요청의 처리 유형을 분류한다.
   - 입력: “퇴직금을 못 받았어요”, “전입신고는 어디서 하나요?”, “출산 지원금이 있나요?”
   - 출력: `public_service_inquiry`, `issue_resolution`, `administrative_procedure`
   - 설명:
     - `public_service_inquiry`: 공공서비스, 지원금, 복지 혜택, 보조금 등을 찾는 요청
     - `issue_resolution`: 피해, 불편, 위반, 고장, 위험 등 문제가 발생해 해결 기관을 찾는 요청
     - `administrative_procedure`: 신고, 등록, 변경, 발급, 납부 등 행정 절차를 알고 싶은 요청

2. 사용자의 요청이 속하는 세부 행정 분야를 분류한다.
   - 입력: “퇴직금을 못 받았어요”, “도로가 파였어요”, “식당 위생이 불량해요”
   - 출력: `labor`, `housing`, `road_safety`, `food_safety`, `birth_support`, `welfare`, `tax`, `environment`, `residence`, `unknown`
   - 설명: `response_type`이 사용자의 요청 목적을 나타낸다면, `category`는 해당 요청이 속한 행정 분야를 나타낸다.

3. 요청의 긴급도를 판단한다.
   - 입력: “가로등이 고장났어요”, “도로가 무너져서 차들이 위험해요”
   - 출력: `normal`, `high`, `emergency`
   - 설명: 긴급도가 높으면 일반 민원 접수 안내보다 112, 119, 지자체 당직실 등 즉시 대응 가능한 채널 안내를 우선한다.

4. 요청 유형, 세부 분야, 긴급도, 지역 필요 여부에 따라 필요한 Tool을 동적으로 선택한다.
   - `public_service_inquiry`이면 `PublicServiceSearchTool`을 우선 사용한다.
   - `issue_resolution`이면 `AgencyRoutingTool`을 우선 사용한다.
   - `administrative_procedure`이면 `AgencyRoutingTool`과 `RequirementAndDraftTool`을 사용한다.
   - 지역 정보가 필요하면 `RegionNormalizeTool`을 사용한다.
   - 필수 정보가 부족하면 Tool을 무리하게 호출하지 않고 추가 질문을 생성한다.

5. 지역 정보가 필요한 요청이면 사용자가 입력한 지역명을 표준 행정구역 정보로 변환한다.
   - 입력: “강남구 역삼동”, “성남 분당”, “부산 해운대”
   - 출력: 시도, 시군구, 읍면동, 법정동 코드 후보
   - 사용 예: 도로 파손 신고, 지자체 민원, 지역별 복지 혜택, 전입신고 등

6. 요청 유형과 키워드를 바탕으로 소관 가능성이 높은 기관 후보를 제시한다.
   - 입력: `response_type`, `category`, 핵심 키워드, 지역 정보
   - 출력: 기관명, 담당 단위, 추천 이유, 접수 채널, 확신도
   - 예: 퇴직금 미지급 → 고용노동부 또는 지방고용노동관서

7. 공공서비스 문의인 경우 관련 공공서비스 또는 정부 혜택 정보를 검색한다.
   - 입력: 핵심 키워드, 지역 정보, 사용자 조건
   - 출력: 서비스명, 제공 기관, 신청 대상, 신청 채널, 추가로 필요한 정보
   - 예: 출산지원, 아동수당, 부모급여, 청년월세지원 등

8. 사용자가 민원 제출이나 행정 절차 진행에 필요한 정보를 빠뜨렸으면 바로 단정하지 않고 추가 질문을 생성한다.
   - 예: 퇴직금 민원에는 사업장명, 근무기간, 퇴사일, 미지급 금액, 급여명세서 여부를 확인한다.
   - 예: 도로 파손 민원에는 정확한 위치, 사진, 발견 시각, 위험 정도를 확인한다.
   - 예: 출산지원 문의에는 거주지, 자녀 출생일, 가구 상황 등을 확인한다.

9. 최종 응답은 자연어 답변과 구조화 정보를 함께 제공한다.
   - 포함 필드:
     - `query_summary`
     - `response_type`
     - `category`
     - `urgency`
     - `agency_candidates`
     - `channels`
     - `answer`
   - 최종 답변에는 소관 기관 후보, 접수 채널, 준비할 정보, 주의사항, 필요한 경우 추가 질문을 포함한다.

### Nice-to-have

1. 관련 법령, 행정규칙, 공공서비스 근거를 참고 정보로 보여준다.
   - 예: 퇴직금, 전입신고, 식품위생, 도로안전 등과 관련된 법령 또는 민원안내정보를 함께 제공한다.

2. 사용자의 이전 상담 이력을 바탕으로 같은 지역이나 같은 기관에 대한 후속 질문을 더 짧게 처리한다.
   - 예: 사용자가 이전에 “서울 강남구” 기준으로 문의했다면, 후속 질문에서도 같은 지역을 기본값으로 사용할 수 있다.

3. 민원 문안 초안을 사용자의 상황에 맞게 공손한 공식 문체로 변환한다.
   - 예: “돈을 안 줘요” → “퇴직금 미지급과 관련하여 사실 확인 및 필요한 조치를 요청드립니다.”

4. 소관 기관 후보가 여러 개인 경우, 기관별 차이를 비교해 안내한다.
   - 예: 국토교통부, 지자체, 경찰청 등 여러 기관이 관련될 수 있는 경우 각 기관에 문의해야 하는 상황을 구분해 설명한다.
---

## 4. Agent 패턴 선택과 근거

### 선택한 패턴

**Plan-and-Execute + ReAct 혼합**

### 선택 근거

이 Agent는 먼저 사용자의 요청을 보고 전체 처리 방향을 정해야 한다.  
사용자의 요청이 공공서비스 문의인지, 문제 발생 문의인지, 행정 절차 문의인지 먼저 구분하고, 그다음 세부 행정 분야와 긴급도를 판단해야 한다. 이처럼 전체 처리 계획을 먼저 세우는 부분은 Plan-and-Execute에 가깝다.

예를 들어 사용자의 요청은 다음과 같이 먼저 분류된다.

```json
{
  "response_type": "issue_resolution",
  "category": "labor",
  "urgency": "normal"
}
```

분류 이후에는 Tool 결과를 보고 다음 행동을 바꿔야 한다.  
예를 들어 지역 정보가 필요하지만 사용자가 지역을 말하지 않았다면 `RegionNormalizeTool`을 호출하지 않고 지역을 추가 질문해야 한다. 소관 기관 후보가 여러 개라면 하나로 단정하지 않고 후보를 비교하거나 추가 질문해야 한다. 긴급도가 `emergency`이면 일반 민원 접수 안내보다 즉시 신고 안내를 우선해야 한다.

이처럼 각 단계에서 Tool을 호출하고, 결과를 확인한 뒤, 다음 행동을 다시 결정하는 부분은 ReAct의 `Thought → Action → Observation` 흐름에 가깝다.

### 동적 Tool 호출 기준

이 Agent는 모든 요청에 대해 같은 Tool 순서를 고정하지 않는다.  
`RequestClassifierTool`로 먼저 요청을 분류한 뒤, `response_type`, `category`, `urgency`, `needs_region`, `confidence` 값을 기준으로 다음 Tool을 선택한다.

```text
response_type이 public_service_inquiry이면 PublicServiceSearchTool을 우선 호출한다.
response_type이 issue_resolution이면 AgencyRoutingTool을 우선 호출한다.
response_type이 administrative_procedure이면 AgencyRoutingTool과 RequirementAndDraftTool을 사용한다.

needs_region이 true이고 지역 정보가 있으면 RegionNormalizeTool을 호출한다.
needs_region이 true인데 지역 정보가 없으면 Tool 호출 대신 지역 확인 질문을 생성한다.

urgency가 emergency이면 일반 민원 안내보다 112, 119, 지자체 당직실 등 즉시 대응 가능한 채널 안내를 우선한다.

confidence가 낮으면 기관이나 절차를 단정하지 않고 추가 질문을 생성한다.
```

### 루프 구조

```text
1. 사용자 요청 수신

2. RequestClassifierTool 호출
   - response_type 추출
   - category 추출
   - urgency 추출
   - keywords 추출
   - needs_region 판단
   - confidence 확인

3. Agent가 처리 계획 수립
   - 공공서비스 문의인가?
   - 문제 발생 문의인가?
   - 행정 절차 문의인가?
   - 지역 정보가 필요한가?
   - 긴급 안내가 필요한가?
   - 소관 기관 탐색이 필요한가?
   - 민원 문안 또는 준비 정보가 필요한가?

4. urgency가 emergency이면 긴급 안내를 우선 생성
   - 필요하면 이후 일반 민원 접수 채널도 함께 안내

5. needs_region이 true인 경우
   - 사용자 입력에 지역이 있으면 RegionNormalizeTool 호출
   - 지역이 없으면 지역 확인 질문을 생성하고 종료

6. response_type에 따라 다음 Tool 선택
   - public_service_inquiry → PublicServiceSearchTool
   - issue_resolution → AgencyRoutingTool
   - administrative_procedure → AgencyRoutingTool

7. 소관 기관 후보 또는 공공서비스 후보 확인
   - 후보가 명확하면 다음 단계 진행
   - 후보가 여러 개이거나 confidence가 낮으면 추가 질문 생성

8. 필요하면 RequirementAndDraftTool 호출
   - 준비할 정보 정리
   - 부족한 정보 확인
   - 민원 문안 초안 생성
   - 주의사항 생성

9. 최종 응답 생성
   - query_summary
   - response_type
   - category
   - urgency
   - agency_candidates
   - channels
   - answer 포함

10. max_steps 또는 stop 조건에 도달하면 종료
```

### 예시 흐름

```text
예시 1. 퇴직금 미지급 문의

사용자 요청:
"회사에서 퇴직금을 안 줬는데 어디에 신고해야 하나요?"

분류 결과:
response_type = issue_resolution
category = labor
urgency = normal

Tool 흐름:
RequestClassifierTool → AgencyRoutingTool → RequirementAndDraftTool
```

```text
예시 2. 전입신고 절차 문의

사용자 요청:
"이사했는데 전입신고는 어디서 어떻게 하나요?"

분류 결과:
response_type = administrative_procedure
category = residence
urgency = normal

Tool 흐름:
RequestClassifierTool → RegionNormalizeTool → AgencyRoutingTool → RequirementAndDraftTool
```

```text
예시 3. 출산지원 문의

사용자 요청:
"아이가 태어났는데 받을 수 있는 정부 지원이 있나요?"

분류 결과:
response_type = public_service_inquiry
category = birth_support
urgency = normal

Tool 흐름:
RequestClassifierTool → RegionNormalizeTool → PublicServiceSearchTool → RequirementAndDraftTool
```

```text
예시 4. 도로 붕괴 위험 문의

사용자 요청:
"도로가 무너져서 차들이 위험해요."

분류 결과:
response_type = issue_resolution
category = road_safety
urgency = emergency

Tool 흐름:
RequestClassifierTool → RegionNormalizeTool → AgencyRoutingTool → FinalAnswer

주의:
urgency가 emergency이므로 일반 민원 접수 안내보다 즉시 신고 채널 안내를 우선한다.
```

### 종료 조건

```text
max_steps = 6

종료 조건:
1. 최종 응답에 필요한 핵심 정보가 확보된 경우
   - query_summary
   - response_type
   - category
   - urgency
   - agency_candidates
   - channels
   - answer

2. 지역 정보, 사용자 조건, 증빙 정보 등 필수 정보가 부족해 추가 질문이 필요한 경우

3. 긴급 상황으로 판단되어 즉시 신고 안내를 완료한 경우

4. 소관 기관 후보가 여러 개이고 추가 정보 없이는 더 좁힐 수 없는 경우

5. 같은 Tool을 같은 입력으로 반복 호출하려는 경우

6. Tool 실패 후 fallback 안내까지 완료한 경우
```

---

## 5. 동작 명세

### 입력 스키마

입력은 자연어 문자열을 기본으로 받는다. 사용자 질의를 받고, 그대로 RequestClassifierTool 을 호출한다. 즉, 일단 사용자는 질의만 하는 형태로 최초 입력을 한다.


### 출력 스키마

최종 응답은 자연어 설명과 구조화 정보를 함께 제공한다.

```json
{
  "query_summary": "string",
  "response_type": "string",
  "category": "string",
  "urgency": "string",
  "agency_candidates": [
    {
      "name": "string",
      "local_unit": "string",
      "reason": "string",
      "confidence": 0.0
    }
  ],
  "channels": ["string"],
  "answer": "string"
}
```


```text
query_summary: 사용자의 요청을 한 문장으로 요약한 값입니다. Agent가 사용자의 질문을 어떻게 이해했는지 보여줍니다.

response_type: 최종 응답의 처리 방식을 나타냅니다. 공공서비스 문의인지, 문제 발생 문의인지, 행정 절차 문의인지 구분합니다. 값은 public_service_inquiry, issue_resolution, administrative_procedure 중 하나로 설정합니다.

category: 사용자의 요청이 속하는 행정 민원 또는 공공서비스 분야입니다. 예를 들어 labor, housing, road_safety, food_safety, birth_support, welfare, tax, environment 등이 들어갈 수 있습니다.

urgency: 요청의 긴급도를 나타냅니다. 일반 민원인지, 빠른 처리가 필요한 민원인지, 즉시 대응이 필요한 상황인지 구분합니다. 값은 normal, high, emergency 중 하나로 설정합니다.

agency_candidates: 사용자의 요청을 처리할 가능성이 있는 기관 후보 목록입니다. 소관 기관이 하나로 확정되지 않을 수 있으므로 배열 형태로 제공합니다.

agency_candidates.name: 소관 가능성이 있는 기관명입니다. 예를 들어 고용노동부, 국토교통부, 지방자치단체, 식품의약품안전처 등이 들어갈 수 있습니다.

agency_candidates.local_unit: 실제 문의나 접수를 담당할 가능성이 있는 하위 기관 또는 지역 단위 부서입니다. 예를 들어 지방고용노동관서, 시군구청 도로관리부서, 시군구청 위생과 등이 들어갈 수 있습니다.

agency_candidates.reason: 해당 기관을 후보로 제시한 이유입니다. 사용자의 요청 내용과 기관의 담당 업무가 어떻게 연결되는지 설명합니다.

agency_candidates.confidence: 해당 기관이 적절한 후보라고 판단한 신뢰도입니다. 0에서 1 사이의 숫자로 표현합니다. 값이 높을수록 해당 기관이 적절할 가능성이 높다는 의미입니다.

channels: 사용자가 실제로 문의하거나 접수할 수 있는 채널 목록입니다. 예를 들어 국민신문고, 정부24, 고용노동부 민원, 지자체 민원, 안전신문고 등이 들어갈 수 있습니다.

answer: 사용자에게 보여줄 최종 자연어 답변입니다. 소관 기관 후보, 접수 방법, 준비할 정보, 주의사항, 필요한 경우 추가 질문을 포함합니다.
```

```text
위와 같이 1차 최종 출력 스키마를 출력한 뒤, 해당 스키마에 맞게 자연어 응답을 한다.
```


### 정상 흐름 예시

사용자 요청:

> “회사에서 퇴직금을 안 줬는데 어디에 신고해야 하나요?”

이 요청은 사용자가 실제 문제를 겪고 있으며, 어느 기관에 신고하거나 문의해야 하는지 모르는 상황이다.  
따라서 Agent는 먼저 요청의 처리 유형과 행정 분야를 분류한 뒤, 소관 기관을 찾고, 민원 접수에 필요한 정보와 문안 초안을 생성한다.

#### Trace 예시

```text
Step 1
Agent 판단: 사용자의 요청은 퇴직금 미지급 문제로 보인다. 먼저 요청의 처리 유형, 행정 분야, 긴급도, 핵심 키워드를 분류한다.

Action: RequestClassifierTool

Input:
{
  "message": "회사에서 퇴직금을 안 줬는데 어디에 신고해야 하나요?"
}

Observation:
{
  "response_type": "issue_resolution",
  "category": "labor",
  "urgency": "normal",
  "keywords": ["퇴직금", "미지급", "신고"],
  "needs_region": false,
  "confidence": 0.91
}
```

```text
Step 2
Agent 판단: response_type이 issue_resolution이고 category가 labor이므로, 공공서비스 검색보다 소관 기관 탐색이 우선이다. 지역 정보는 필수로 판단되지 않으므로 RegionNormalizeTool은 호출하지 않는다.

Action: AgencyRoutingTool

Input:
{
  "response_type": "issue_resolution",
  "category": "labor",
  "keywords": ["퇴직금", "미지급", "신고"],
  "region": null
}

Observation:
{
  "candidates": [
    {
      "name": "고용노동부",
      "local_unit": "지방고용노동관서",
      "reason": "퇴직금 미지급은 임금체불 관련 노동 민원에 해당할 가능성이 높기 때문입니다.",
      "confidence": 0.88
    }
  ],
  "channels": ["국민신문고", "고용노동부 민원", "지방고용노동관서 상담"],
  "routing_reason": "퇴직금 미지급은 근로관계에서 발생한 금품 미지급 문제이므로 노동 분야 소관 기관으로 라우팅합니다."
}
```

```text
Step 3
Agent 판단: 소관 기관 후보는 확인되었지만, 사용자가 실제 민원을 접수하려면 준비할 정보와 증빙자료가 필요하다. 민원 문안 초안도 함께 생성한다.

Action: RequirementAndDraftTool

Input:
{
  "category": "labor",
  "agency": {
    "name": "고용노동부",
    "local_unit": "지방고용노동관서",
    "reason": "퇴직금 미지급은 임금체불 관련 노동 민원에 해당할 가능성이 높기 때문입니다.",
    "confidence": 0.88
  },
  "facts": {
    "issue": "퇴직금 미지급"
  },
  "want_draft": true
}

Observation:
{
  "required_info": [
    "사업장명",
    "근무기간",
    "퇴사일",
    "미지급 금액",
    "근로계약서",
    "급여명세서",
    "입금내역"
  ],
  "missing_info": [
    "사업장명",
    "근무기간",
    "퇴사일",
    "미지급 금액"
  ],
  "draft": "안녕하세요. 저는 ○○회사에서 ○년 ○월부터 ○년 ○월까지 근무한 뒤 퇴사하였으나, 퇴직금을 지급받지 못했습니다. 퇴직금 미지급 사실 확인 및 필요한 조치를 요청드립니다.",
  "cautions": [
    "정확한 사업장명과 근무기간을 정리해야 합니다.",
    "근로계약서, 급여명세서, 입금내역 등 증빙자료를 준비하는 것이 좋습니다.",
    "Agent는 실제 민원 접수를 대행하지 않으며, 접수는 사용자가 직접 진행해야 합니다."
  ]
}
```

```text
Step 4
Agent 판단: 소관 기관, 접수 채널, 준비 정보, 문안 초안이 확보되었다. 일부 필수 정보가 부족하므로 최종 답변에는 추가로 확인해야 할 정보를 함께 포함한다.

Action: FinalAnswer

Observation:
최종 응답 생성
```

#### 최종 응답 예시

```json
{
  "query_summary": "퇴직금 미지급 신고 기관 문의",
  "response_type": "issue_resolution",
  "category": "labor",
  "urgency": "normal",
  "agency_candidates": [
    {
      "name": "고용노동부",
      "local_unit": "지방고용노동관서",
      "reason": "퇴직금 미지급은 임금체불 관련 노동 민원에 해당할 가능성이 높기 때문입니다.",
      "confidence": 0.88
    }
  ],
  "channels": [
    "국민신문고",
    "고용노동부 민원",
    "지방고용노동관서 상담"
  ],
  "answer": "퇴직금 미지급은 노동 분야의 문제 발생 문의로 볼 수 있습니다. 우선 고용노동부 또는 관할 지방고용노동관서에 문의하거나 진정 접수를 검토할 수 있습니다. 접수 전에는 사업장명, 근무기간, 퇴사일, 미지급 금액, 근로계약서, 급여명세서, 입금내역을 준비하는 것이 좋습니다. 현재 요청만으로는 사업장명, 근무기간, 퇴사일, 미지급 금액이 부족하므로 실제 접수 전 이 정보를 정리해야 합니다.\n\n민원 문안 초안:\n안녕하세요. 저는 ○○회사에서 ○년 ○월부터 ○년 ○월까지 근무한 뒤 퇴사하였으나, 퇴직금을 지급받지 못했습니다. 퇴직금 미지급 사실 확인 및 필요한 조치를 요청드립니다."
}
```

#### 자연어 응답 예시

```text
퇴직금 미지급은 노동 분야의 문제 발생 문의로 볼 수 있습니다.
우선 고용노동부 또는 관할 지방고용노동관서에 문의하거나 진정 접수를 검토할 수 있습니다.

접수 가능한 채널:
- 국민신문고
- 고용노동부 민원
- 지방고용노동관서 상담

접수 전 준비하면 좋은 정보:
- 사업장명
- 근무기간
- 퇴사일
- 미지급 금액
- 근로계약서
- 급여명세서
- 입금내역

추가로 확인이 필요한 정보:
- 정확한 사업장명
- 근무기간
- 퇴사일
- 미지급 금액

민원 문안 초안:
안녕하세요. 저는 ○○회사에서 ○년 ○월부터 ○년 ○월까지 근무한 뒤 퇴사하였으나, 퇴직금을 지급받지 못했습니다. 퇴직금 미지급 사실 확인 및 필요한 조치를 요청드립니다.

주의사항:
이 Agent는 실제 민원 접수를 대신하지 않습니다. 본인 인증, 개인정보 입력, 증빙자료 첨부, 최종 제출은 사용자가 직접 진행해야 합니다.
```

### 예외 흐름

#### Tool 실패

Tool은 예외를 직접 던지지 않고 실패 결과를 구조화된 형태로 반환한다.  
Agent는 이 값을 보고 재시도할지, 다른 Tool을 사용할지, 일반 안내로 전환할지 결정한다.

#### Tool 실패 응답 스키마

Tool 실행에 실패한 경우에는 예외를 직접 던지지 않고, 아래와 같은 공통 실패 응답 형식으로 반환한다.

```json
{
  "error": "string",
  "detail": "string",
  "retryable": true,
  "fallback_strategy": "string"
}
```

```text
error: Tool 실패 유형을 나타내는 에러 코드입니다. Agent는 이 값을 보고 재시도할지, 추가 질문을 할지, 다른 안내로 전환할지 판단합니다.

error 값 예시:
- API_TIMEOUT: 외부 API 응답 시간이 초과된 경우입니다. 같은 요청을 1회 재시도할 수 있습니다.
- API_UNAVAILABLE: 외부 API 서버가 응답하지 않거나 일시적으로 사용할 수 없는 경우입니다.
- INVALID_INPUT: Tool 호출에 필요한 입력값이 잘못되었거나 누락된 경우입니다.
- CLASSIFICATION_FAILED: 사용자 요청을 분류하지 못한 경우입니다. RequestClassifierTool에서 발생할 수 있습니다.
- REGION_NOT_FOUND: 입력된 지역명을 표준 행정구역 정보에서 찾지 못한 경우입니다. RegionNormalizeTool에서 발생할 수 있습니다.
- AMBIGUOUS_REGION: 같은 이름의 지역 후보가 여러 개 있어 하나로 확정하기 어려운 경우입니다.
- NO_ROUTE_FOUND: 요청에 맞는 소관 기관 후보를 찾지 못한 경우입니다. AgencyRoutingTool에서 발생할 수 있습니다.
- SERVICE_API_FAILED: 공공서비스 검색 API 호출에 실패한 경우입니다. PublicServiceSearchTool에서 발생할 수 있습니다.
- DRAFT_FAILED: 필요 정보 정리나 민원 문안 초안 생성에 실패한 경우입니다. RequirementAndDraftTool에서 발생할 수 있습니다.

detail: 실패 원인에 대한 구체적인 설명입니다. 사용자가 직접 보는 메시지라기보다는 Agent가 다음 행동을 결정하기 위해 참고하는 내부 설명입니다.

detail 값 예시:
- "PublicServiceSearchTool request timed out"
- "입력된 지역명을 표준 행정구역에서 찾을 수 없음"
- "소관 기관 후보의 confidence가 모두 낮음"
- "category 값이 누락되어 문안 생성 불가"

retryable: 같은 Tool을 같은 입력으로 다시 호출할 수 있는지 여부입니다.

retryable 값:
- true: 일시적인 API 오류나 네트워크 오류처럼 1회 재시도해볼 수 있는 경우입니다.
- false: 입력값 누락, 지역명 불명확, 분류 실패처럼 재시도해도 같은 결과가 나올 가능성이 높은 경우입니다.

fallback_strategy: Tool 실패 후 Agent가 선택할 대체 처리 방식입니다.

fallback_strategy 값 예시:
- retry_once: 같은 Tool을 동일 파라미터로 1회만 재시도합니다.
- ask_user: 필요한 정보가 부족하거나 입력이 모호하므로 사용자에게 추가 질문을 합니다.
- use_alternative_source: 같은 목적을 달성할 수 있는 다른 API나 데이터 출처를 사용합니다.
- general_guidance: API 조회 없이 제공할 수 있는 기본 안내로 전환합니다.
- stop_with_notice: 더 이상 안전하게 진행할 수 없으므로 실패 사유와 재확인 필요성을 안내하고 종료합니다.
```

#### Tool별 실패 응답 예시

```json
{
  "error": "API_TIMEOUT",
  "detail": "PublicServiceSearchTool request timed out",
  "retryable": true,
  "fallback_strategy": "retry_once"
}
```

```json
{
  "error": "REGION_NOT_FOUND",
  "detail": "입력된 지역명을 표준 행정구역에서 찾을 수 없음",
  "retryable": false,
  "fallback_strategy": "ask_user"
}
```

```json
{
  "error": "NO_ROUTE_FOUND",
  "detail": "해당 category와 keyword에 맞는 소관 기관 후보를 찾지 못함",
  "retryable": false,
  "fallback_strategy": "general_guidance"
}
```

```json
{
  "error": "SERVICE_API_FAILED",
  "detail": "공공서비스 API 호출에 실패함",
  "retryable": true,
  "fallback_strategy": "use_alternative_source"
}
```

```json
{
  "error": "DRAFT_FAILED",
  "detail": "기관 정보 또는 사용자 사실 정보가 부족해 민원 문안 생성 불가",
  "retryable": false,
  "fallback_strategy": "ask_user"
}
```

#### fallback_strategy 처리 기준

```text
retry_once: API_TIMEOUT, API_UNAVAILABLE처럼 일시적인 오류일 때 사용합니다. 단, 같은 Tool은 같은 입력으로 최대 1회만 재시도합니다.

ask_user: 지역명, 사용자 조건, 사건 정보, 증빙 정보 등 필수 입력이 부족하거나 모호할 때 사용합니다.

use_alternative_source: 같은 목적을 수행할 수 있는 다른 API나 공식 데이터 출처가 있을 때 사용합니다. 예를 들어 공공서비스 검색 API가 실패하면 정부24 또는 복지로 확인 안내로 전환할 수 있습니다.

general_guidance: 정확한 API 조회 결과 없이도 제공할 수 있는 기본 안내로 전환할 때 사용합니다. 이 경우 특정 기관이나 서비스명을 확정하지 않고, 일반 접수 채널, 준비할 정보, 공식 사이트 재확인 필요성을 안내합니다.

stop_with_notice: 개인정보, 법률 판단, 실제 민원 접수 대행처럼 Agent가 처리할 수 없는 범위이거나 더 이상 안전하게 진행할 수 없을 때 사용합니다.
```

Agent는 Tool 실패 시 다음 순서로 대응한다.

```text
1. retryable이 true이면 같은 Tool을 동일 파라미터로 1회만 재시도한다.

2. 재시도에 실패하면 같은 기능을 수행할 수 있는 다른 데이터 출처나 Tool이 있는지 확인한다.
   - 예: 공공서비스 검색 실패 시 정부24 또는 일반 공공서비스 안내 채널을 안내한다.
   - 예: 지역 정규화 실패 시 사용자가 입력한 지역명을 그대로 사용하되, 정확한 시·군·구 확인 질문을 생성한다.

3. 대체 Tool이나 대체 데이터 출처가 없으면 기본 안내로 전환한다.

기본 안내란, API 조회 결과 없이도 제공할 수 있는 최소한의 안전한 안내를 의미한다.
이 경우 Agent는 특정 기관이나 서비스명을 확정하지 않고, 사용자가 확인해야 할 일반 채널, 준비 정보, 추가 확인 질문을 제공한다.

예를 들어 공공서비스 검색 API가 실패하면 특정 지원금 신청 가능 여부를 단정하지 않고, 정부24, 복지로, 주민센터 등에서 확인할 수 있음을 안내한다.
소관 기관 탐색에 실패하면 특정 부처를 확정하지 않고, 국민신문고 또는 지자체 종합민원실을 통해 소관 기관 확인이 필요하다고 안내한다.
지역 정규화에 실패하면 사용자가 입력한 지역명을 그대로 사용하지 않고, 시·군·구 단위 지역명을 다시 요청한다.

4. 최종 응답에는 "API 조회에 실패했으므로 최신 정보는 공식 사이트에서 재확인해야 한다"는 주의사항을 포함한다.
```

---

#### 요청 분류가 불명확한 경우

`RequestClassifierTool`의 `confidence`가 낮거나, `response_type`을 하나로 정하기 어려운 경우에는 기관이나 절차를 단정하지 않는다.  
이 경우 Agent는 가능한 후보를 제시하고 추가 질문을 생성한다.

```text
이 요청은 공공서비스 문의와 행정 절차 문의가 모두 가능해 보입니다.
정확히 안내하려면 다음 중 어떤 상황인지 알려주세요.

1. 받을 수 있는 지원금이나 혜택을 찾고 싶으신가요?
2. 이미 신청할 서비스가 정해져 있고 신청 절차를 알고 싶으신가요?
3. 특정 기관에 문의하거나 민원을 접수하려는 상황인가요?
```

---

#### 소관 기관이 불명확한 경우

`AgencyRoutingTool`이 여러 기관 후보를 반환하거나, 후보들의 `confidence`가 비슷한 경우에는 하나의 기관으로 단정하지 않는다.  
Agent는 후보를 2~3개 제시하고, 어느 기관이 더 적절한지 판단하기 위한 추가 질문을 생성한다.

```text
이 요청은 주거 민원과 소비자 분쟁이 모두 가능해 보입니다.
정확히 안내하려면 다음 중 어떤 상황인지 알려주세요.

1. 임대차계약, 보증금, 월세 관련 문제인가요?
2. 업체와의 계약·환불·서비스 불만 문제인가요?
3. 범죄 피해나 사기 신고에 가까운 문제인가요?
```

---

#### 지역 정보가 필요한데 누락된 경우

`needs_region`이 true인데 사용자 입력에 지역 정보가 없으면 `RegionNormalizeTool`을 호출하지 않는다.  
이 경우 Agent는 지역 확인 질문을 생성하고 종료한다.

```text
이 민원은 지역별 담당 기관이나 지자체 부서가 달라질 수 있습니다.
정확히 안내하려면 시·군·구 단위의 지역을 알려주세요.

예: 서울 강남구, 경기 성남시 분당구, 부산 해운대구
```

---

#### 공공서비스 검색에 사용자 조건이 부족한 경우

`PublicServiceSearchTool`에서 `need_more_info`가 반환되면 신청 가능 여부를 단정하지 않는다.  
Agent는 현재 확인 가능한 서비스 후보를 제시하되, 부족한 조건을 추가 질문한다.

```text
출산지원 관련 공공서비스를 확인하려면 추가 정보가 필요합니다.

1. 거주 지역은 어디인가요?
2. 자녀 출생일은 언제인가요?
3. 보호자의 가구 형태나 소득 조건을 확인할 수 있나요?

이 정보에 따라 중앙정부 지원과 지자체 지원이 달라질 수 있습니다.
```

---

#### 긴급 안전 상황

`urgency`가 `emergency`로 판단되면 일반 민원 접수 안내보다 즉시 대응 가능한 신고 채널 안내를 우선한다.  
도로 붕괴, 화재, 인명 피해, 범죄, 가스 누출, 즉시 사고 위험이 있는 경우가 여기에 해당한다.

```text
인명 피해나 즉시 위험이 있다면 일반 민원 접수보다 119 또는 112 신고가 우선입니다.
위치, 사진, 발견 시각을 정리해두면 이후 지자체 또는 안전 관련 민원 채널에 신고할 때 도움이 됩니다.
```

긴급 상황에서도 최종 응답에는 가능한 범위에서 소관 기관 후보와 후속 접수 채널을 함께 안내한다.

---

#### 권한 부족 또는 개인정보 문제

Agent는 사용자를 대신해 실제 민원을 접수하지 않는다.  
본인 인증, 개인정보 입력, 증빙자료 첨부, 법적 제출 행위는 사용자가 직접 수행해야 한다.

Agent는 다음 행위를 수행하지 않는다.

```text
1. 사용자를 대신한 민원 접수
2. 주민등록번호, 계좌번호 등 민감정보 저장
3. 법적 판단의 확정
4. 특정 기관이 반드시 처리해야 한다는 단정
5. 증빙자료의 진위 판단
```

Agent는 대신 다음 범위까지만 지원한다.

```text
1. 소관 가능성이 높은 기관 후보 안내
2. 접수 가능한 채널 안내
3. 준비할 정보와 증빙자료 정리
4. 민원 문안 초안 작성
5. 추가 확인이 필요한 질문 생성
```

---

## 6. Tool 명세

# RequestClassifierTool
```text
목적: RequestClassifierTool.목적: 사용자 요청을 분석해 행정 분야, 요청 목적, 긴급도, 핵심 키워드, 지역 정보 필요 여부를 분류합니다.


입력 스키마 필드

RequestClassifierTool.message: 사용자가 입력한 원문 요청입니다. Agent는 이 값을 기준으로 요청의 분야, 의도, 긴급성을 판단합니다.


출력 스키마 필드

RequestClassifierTool.category: 사용자의 요청이 속하는 행정 분야입니다. 예를 들어 labor, housing, road_safety, food_safety, birth_support, welfare, tax, environment 등이 들어갈 수 있습니다.

RequestClassifierTool.response_type: 사용자의 요청 목적입니다. 공공서비스 문의, 문제 발생 문의, 행정 절차 문의 중 어떤 성격인지 판단한 값입니다. 예를 들어 public_service_inquiry, issue_resolution, administrative_procedure 등이 들어갈 수 있습니다.

RequestClassifierTool.urgency: 요청의 긴급도입니다. 일반 민원인지, 빠른 처리가 필요한 민원인지, 즉시 대응이 필요한 상황인지 구분합니다. 예를 들어 normal, high, emergency 등이 들어갈 수 있습니다.

RequestClassifierTool.keywords: 사용자 요청에서 추출한 핵심 키워드 목록입니다. 예를 들어 “퇴직금을 못 받았다”는 요청에서는 ["퇴직금", "미지급", "신고"] 같은 값이 들어갈 수 있습니다.

RequestClassifierTool.needs_region: 해당 요청을 처리하는 데 지역 정보가 필요한지 여부입니다. 예를 들어 도로 파손, 지자체 민원, 지역별 복지 혜택 문의는 true가 될 수 있습니다.

RequestClassifierTool.confidence: 분류 결과에 대한 신뢰도입니다. 0에서 1 사이의 숫자로 표현하며, 값이 높을수록 분류 결과가 확실하다는 의미입니다.


실패 시 반환 필드

RequestClassifierTool.error: 분류에 실패했을 때 반환되는 에러 코드입니다. 예를 들어 CLASSIFICATION_FAILED가 들어갈 수 있습니다.

RequestClassifierTool.detail: 실패 원인에 대한 설명입니다. 예를 들어 “요청 내용이 너무 짧아 분류할 수 없음” 같은 값이 들어갈 수 있습니다.


사용 조건: 모든 사용자 요청의 첫 단계에서 사용합니다. 이후 어떤 Tool을 호출할지 결정하기 위한 기준 정보를 만들 때 사용합니다.
```

# RegionNormalizeTool
'''text
목적: 사용자가 입력한 지역명을 표준 행정구역명 또는 법정동 코드로 변환합니다.


입력 스키마 필드

RegionNormalizeTool.region_text: 사용자가 입력한 지역명입니다. 예를 들어 “강남구”, “서울 역삼동”, “부산 해운대” 같은 값이 들어갑니다.


출력 스키마 필드

RegionNormalizeTool.sido: 표준화된 시·도 이름입니다. 예를 들어 서울특별시, 경기도, 부산광역시 등이 들어갑니다.

RegionNormalizeTool.sigungu: 표준화된 시·군·구 이름입니다. 예를 들어 강남구, 성남시 분당구, 해운대구 등이 들어갑니다.

RegionNormalizeTool.dong: 표준화된 읍·면·동 이름입니다. 예를 들어 역삼동, 정자동, 우동 등이 들어갑니다.

RegionNormalizeTool.legal_dong_code: 법정동 코드입니다. 행정구역을 시스템에서 정확히 식별하기 위한 코드값입니다.

RegionNormalizeTool.confidence: 지역명 변환 결과에 대한 신뢰도입니다. 0에서 1 사이의 숫자로 표현합니다.


실패 시 반환 필드

RegionNormalizeTool.error: 지역명 변환에 실패했을 때 반환되는 에러 코드입니다. 예를 들어 REGION_NOT_FOUND가 들어갈 수 있습니다.

RegionNormalizeTool.detail: 실패 원인에 대한 설명입니다. 예를 들어 “입력된 지역명을 표준 행정구역에서 찾을 수 없음” 같은 값이 들어갈 수 있습니다.


사용 조건: 지역별 소관 부서, 지자체 민원, 도로·안전 민원, 지역별 공공서비스 검색처럼 지역 정보가 필요한 경우에 사용합니다.
'''

# AgencyRoutingTool
'''text
목적: 사용자의 민원 유형, 요청 목적, 핵심 키워드, 지역 정보를 바탕으로 소관 가능성이 높은 기관 후보와 접수 채널을 찾습니다.


입력 스키마 필드

AgencyRoutingTool.category: 사용자의 요청이 속하는 행정 분야입니다. RequestClassifierTool에서 분류한 category 값을 사용합니다.

AgencyRoutingTool.response_type: 사용자의 요청 목적입니다. 예를 들어 문제 발생 문의인지, 행정 절차 문의인지에 따라 소관 기관 탐색 방식이 달라질 수 있습니다.

AgencyRoutingTool.keywords: 소관 기관을 찾는 데 사용할 핵심 키워드 목록입니다. 예를 들어 ["퇴직금", "미지급", "임금체불"] 같은 값이 들어갈 수 있습니다.

AgencyRoutingTool.region: 지역 정보 객체입니다. RegionNormalizeTool의 결과를 넣을 수 있습니다. 지역별 담당 부서나 지자체 민원을 찾을 때 사용합니다.


출력 스키마 필드

AgencyRoutingTool.candidates: 사용자의 요청을 처리할 가능성이 있는 기관 후보 목록입니다. 소관 기관이 하나로 확정되지 않을 수 있으므로 배열 형태로 반환합니다.

AgencyRoutingTool.candidates.name: 소관 가능성이 있는 기관명입니다. 예를 들어 고용노동부, 국토교통부, 지방자치단체, 식품의약품안전처 등이 들어갈 수 있습니다.

AgencyRoutingTool.candidates.local_unit: 실제 문의나 접수를 담당할 가능성이 있는 하위 기관 또는 지역 단위 부서입니다. 예를 들어 지방고용노동관서, 시군구청 도로관리부서, 시군구청 위생과 등이 들어갈 수 있습니다.

AgencyRoutingTool.candidates.reason: 해당 기관을 후보로 제시한 이유입니다. 사용자의 요청 내용과 기관의 담당 업무가 어떻게 연결되는지 설명합니다.

AgencyRoutingTool.candidates.confidence: 해당 기관이 적절한 후보라고 판단한 신뢰도입니다. 0에서 1 사이의 숫자로 표현합니다.

AgencyRoutingTool.channels: 사용자가 실제로 문의하거나 접수할 수 있는 채널 목록입니다. 예를 들어 국민신문고, 정부24, 고용노동부 민원, 지자체 민원, 안전신문고 등이 들어갈 수 있습니다.

AgencyRoutingTool.routing_reason: 해당 기관 후보와 접수 채널을 선택한 전체 이유입니다. 최종 답변에서 사용자가 왜 이 기관에 문의해야 하는지 설명하는 데 사용됩니다.


실패 시 반환 필드

AgencyRoutingTool.error: 소관 기관 탐색에 실패했을 때 반환되는 에러 코드입니다. 예를 들어 NO_ROUTE_FOUND가 들어갈 수 있습니다.

AgencyRoutingTool.detail: 실패 원인에 대한 설명입니다. 예를 들어 “해당 category와 keyword에 맞는 기관을 찾지 못함” 같은 값이 들어갈 수 있습니다.

AgencyRoutingTool.fallback: 정확한 소관 기관을 찾지 못했을 때 대신 사용할 기본 후보 목록입니다. 예를 들어 국민신문고, 정부24, 지자체 종합민원실 같은 일반 채널이 들어갈 수 있습니다.


사용 조건: 사용자가 어디에 문의해야 하는지, 어디에 신고해야 하는지, 어느 기관에 민원을 접수해야 하는지 묻는 경우에 사용합니다.
'''

# PublicServiceSearchTool
'''text
목적: 출산지원, 청년지원, 복지혜택, 보조금 등 사용자가 신청하거나 받을 수 있는 공공서비스를 검색합니다.


입력 스키마 필드

PublicServiceSearchTool.keywords: 공공서비스를 검색할 때 사용할 핵심 키워드 목록입니다. 예를 들어 ["출산지원", "아동수당", "부모급여"] 같은 값이 들어갈 수 있습니다.

PublicServiceSearchTool.region: 사용자의 지역 정보 객체입니다. 지역별 지원금이나 지자체 서비스를 찾을 때 사용합니다.

PublicServiceSearchTool.profile: 사용자의 조건 정보입니다. 예를 들어 나이, 가구 형태, 자녀 여부, 취업 상태, 소득 구간 등이 들어갈 수 있습니다.


출력 스키마 필드

PublicServiceSearchTool.services: 검색된 공공서비스 목록입니다. 지원금, 복지 혜택, 출산지원, 청년정책 등이 배열 형태로 반환됩니다.

PublicServiceSearchTool.services.name: 공공서비스 또는 지원 제도의 이름입니다. 예를 들어 부모급여, 아동수당, 청년월세지원 등이 들어갈 수 있습니다.

PublicServiceSearchTool.services.provider: 서비스를 제공하는 기관입니다. 예를 들어 보건복지부, 국토교통부, 지방자치단체 등이 들어갈 수 있습니다.

PublicServiceSearchTool.services.summary: 서비스의 핵심 내용을 요약한 설명입니다.

PublicServiceSearchTool.services.eligibility: 신청 대상 또는 자격 조건입니다. 예를 들어 “만 0세 아동을 양육하는 가구” 같은 값이 들어갈 수 있습니다.

PublicServiceSearchTool.services.application_channel: 신청 가능한 채널입니다. 예를 들어 정부24, 복지로, 주민센터 등이 들어갈 수 있습니다.

PublicServiceSearchTool.matched_conditions: 사용자의 조건과 서비스 조건이 일치한 항목 목록입니다. 예를 들어 지역, 연령, 자녀 여부 등이 들어갈 수 있습니다.

PublicServiceSearchTool.need_more_info: 서비스 신청 가능성을 더 정확히 판단하기 위해 추가로 필요한 정보 목록입니다. 예를 들어 소득 구간, 거주지, 자녀 출생일 등이 들어갈 수 있습니다.


실패 시 반환 필드

PublicServiceSearchTool.error: 공공서비스 검색에 실패했을 때 반환되는 에러 코드입니다. 예를 들어 SERVICE_API_FAILED가 들어갈 수 있습니다.

PublicServiceSearchTool.detail: 실패 원인에 대한 설명입니다. 예를 들어 “공공서비스 API 응답 지연” 같은 값이 들어갈 수 있습니다.


사용 조건: 사용자가 지원금, 혜택, 복지서비스, 신청 가능한 제도, 받을 수 있는 공공서비스를 묻는 경우에 사용합니다.
'''

# RequirementAndDraftTool
'''text
목적: 민원 접수나 문의에 필요한 준비 정보, 부족한 정보, 주의사항, 민원 문안 초안을 생성합니다.


입력 스키마 필드

RequirementAndDraftTool.category: 사용자의 요청이 속하는 행정 분야입니다. 필요한 정보와 민원 문안 양식을 결정하는 데 사용합니다.

RequirementAndDraftTool.agency: 소관 기관 후보 정보입니다. AgencyRoutingTool에서 반환한 기관 후보 중 최종적으로 사용할 기관 정보를 넣습니다.

RequirementAndDraftTool.facts: 사용자의 요청에서 추출한 사실 정보입니다. 예를 들어 사업장명, 근무기간, 퇴사일, 미지급 금액, 위치, 방문일시 등이 들어갈 수 있습니다.

RequirementAndDraftTool.want_draft: 사용자가 민원 문안 초안 생성을 원하는지 여부입니다. true이면 draft를 생성하고, false이면 필요 정보와 주의사항 중심으로 반환합니다.


출력 스키마 필드

RequirementAndDraftTool.required_info: 민원 접수나 문의 전에 준비해야 할 정보 목록입니다. 예를 들어 사업장명, 주소, 사진, 계약서, 방문일시 등이 들어갈 수 있습니다.

RequirementAndDraftTool.missing_info: 현재 사용자 입력에서 부족한 정보 목록입니다. Agent는 이 값을 바탕으로 추가 질문을 생성할 수 있습니다.

RequirementAndDraftTool.draft: 사용자가 실제 민원이나 문의에 사용할 수 있는 문안 초안입니다. 공손하고 공식적인 문체로 작성됩니다.

RequirementAndDraftTool.cautions: 사용자가 접수 전에 주의해야 할 사항 목록입니다. 예를 들어 개인정보 입력 주의, 증빙자료 첨부 필요, 긴급 상황 시 신고 우선 등이 들어갈 수 있습니다.


실패 시 반환 필드

RequirementAndDraftTool.error: 필요 정보 정리나 문안 생성에 실패했을 때 반환되는 에러 코드입니다. 예를 들어 DRAFT_FAILED가 들어갈 수 있습니다.

RequirementAndDraftTool.detail: 실패 원인에 대한 설명입니다. 예를 들어 “기관 정보가 없어 문안 생성 불가” 같은 값이 들어갈 수 있습니다.


사용 조건: 소관 기관 후보가 정해진 뒤, 사용자가 실제로 문의하거나 민원을 접수하기 위해 준비해야 할 정보나 문안이 필요한 경우에 사용합니다.
'''


### Tool별 데이터 후보

- `RegionNormalizeTool`: 행정안전부_행정표준코드_법정동코드 API
- `PublicServiceSearchTool`: 행정안전부_대한민국 공공서비스(혜택) 정보 API
- `AgencyRoutingTool`: 국민권익위원회_민원빅데이터_분석정보_API_2025 + Mock 라우팅 데이터
- `RequirementAndDraftTool`: Mock 필요 정보 데이터 + LLM 문안 생성

---

## 7. 데이터셋

### 데이터 출처

### 데이터 출처

| 데이터 | 사용 Tool | 용도 | 실제 사용 여부 |
|---|---|---|---|
| 행정안전부_행정표준코드_법정동코드 | `RegionNormalizeTool` | 사용자가 입력한 지역명을 표준 행정구역명과 법정동 코드로 변환 | 실제 API 후보 |
| 행정안전부_행정표준코드_기관코드 | `AgencyRoutingTool` | 기관명을 표준 기관코드와 연결하고, 기관명 표기를 정규화 | 실제 API 후보 |
| 행정안전부_정부조직도 조직개편일자별 기구정보 | `AgencyRoutingTool` | 중앙행정기관 및 하위조직 구조를 확인해 소관 기관 후보 탐색 | 실제 데이터 후보 |
| 행정안전부_정부기능별분류체계 | `RequestClassifierTool`, `AgencyRoutingTool` | 사용자 요청을 정부 기능 분류와 매칭해 행정 분야 분류 보조 | 실제 데이터 후보 |
| 행정안전부_민원안내정보 | `AgencyRoutingTool`, `RequirementAndDraftTool` | 민원사무명, 신청유형, 신청자격, 접수방법, 수수료, 근거법령 확인 | 실제 데이터 후보 |
| 행정안전부_대한민국 공공서비스(혜택) 정보 | `PublicServiceSearchTool` | 정부 부처, 지자체, 공공기관, 교육청 등이 제공하는 공공서비스·정부혜택 검색 | 실제 API 후보 |
| 국민권익위원회_민원빅데이터_분석정보_API_2025 | `RequestClassifierTool`, `AgencyRoutingTool` | 키워드 기반 기관별 민원 건수와 법령 정보를 참고해 민원 분야·기관 후보 판단 보조 | 실제 API 후보 |
| 국민권익위원회_민원빅데이터_분석정보_API_2024 | `RequestClassifierTool`, `AgencyRoutingTool` | 민원 발생지 정보와 포트홀 민원 위치 정보 등을 활용해 도로·안전 민원 판단 보조 | 실제 API 후보 |
| 행정안전부_통계연보_민원사무 종류 | `RequestClassifierTool` | 제증명, 인허가, 등록, 신고, 검사 등 민원사무 유형 분류 참고 | 실제 API 후보 |
| 행정안전부_통계연보_분야별 안전신고 | `RequestClassifierTool`, `AgencyRoutingTool` | 안전신문고, 생활불편신고, 불법주정차 등 안전 민원 유형 판단 보조 | 실제 API 후보 |
| 법제처 국가법령정보 공유서비스 | `RequirementAndDraftTool` | 민원 문안 작성 시 관련 법령, 행정규칙, 자치법규, 별표서식 등 근거 정보 조회 | 실제 API 후보 |
| 법제처_법령정보지식베이스 지능형 법령검색 시스템 검색 | `RequirementAndDraftTool` | 사용자 키워드와 관련된 법령 조문, 별표·서식, 행정규칙 검색 | 실제 API 후보 |
| 국민권익위원회_결정문 본문조회 국가법령정보센터API | `AgencyRoutingTool`, `RequirementAndDraftTool` | 고충민원 의결서, 조정·합의 사례를 참고해 유사 민원 대응 방향 보조 | 실제 API 후보 |

### 참고 API 설명

#### 행정안전부_행정표준코드_법정동코드

사용자가 입력한 지역명을 표준 행정구역명과 법정동 코드로 변환하는 데 사용한다.  
본 Agent에서는 `RegionNormalizeTool`의 지역명 정규화 데이터로 활용한다.

#### 행정안전부_행정표준코드_기관코드

행정기관명을 표준 기관명 또는 기관코드로 변환하는 데 사용한다.  
본 Agent에서는 `AgencyRoutingTool`에서 소관 기관 후보를 정규화하는 보조 데이터로 활용한다.

#### 행정안전부_대한민국 공공서비스(혜택) 정보

정부 부처, 지자체, 공공기관 등이 제공하는 공공서비스와 정부 혜택 정보를 검색하는 데 사용한다.  
본 Agent에서는 `PublicServiceSearchTool`의 핵심 데이터로 활용한다.

#### 국민권익위원회_민원빅데이터_분석정보_API_2025

민원 키워드와 기관별 민원 경향을 참고하는 데 사용한다.  
본 Agent에서는 `RequestClassifierTool`과 `AgencyRoutingTool`의 분류·라우팅 보조 데이터로 활용한다.

#### 행정안전부_민원안내정보

민원사무명, 신청유형, 접수방법, 소관부처, 신청서 정보를 확인하는 데 사용한다.  
본 Agent에서는 `AgencyRoutingTool`과 `RequirementAndDraftTool`에서 접수 절차와 준비 정보를 확인하는 데 활용한다.

#### 행정안전부_정부기능별분류체계

정부 업무를 정책분야, 정책영역, 기능 단위로 분류한 데이터이다.  
본 Agent에서는 `RequestClassifierTool`에서 사용자 요청을 행정 분야로 분류할 때 참고한다.

#### 법제처 국가법령정보 공동활용 API

현행법령, 행정규칙, 자치법규, 별표·서식 등 법령 정보를 조회하는 데 사용한다.  
본 Agent에서는 `RequirementAndDraftTool`에서 민원 문안 작성 시 관련 근거를 참고하는 데 활용한다.

#### 법제처_법령정보지식베이스 지능형 법령검색 시스템 검색

사용자 키워드와 관련된 법령 조문, 행정규칙, 별표·서식을 검색하는 데 사용한다.  
본 Agent에서는 민원 키워드와 연결되는 법령·서식 후보를 찾는 보조 데이터로 활용한다.


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

- 한국 공개 API 목록: https://github.com/yybmion/public-apis-4Kr
- 행정안전부_대한민국 공공서비스(혜택) 정보: https://www.data.go.kr/data/15113968/openapi.do
- 행정안전부_행정표준코드_법정동코드: https://www.data.go.kr/data/15077871/openapi.do
- 행정안전부_행정표준코드_기관코드: https://www.data.go.kr/data/15077870/openapi.do
- 국민권익위원회_민원빅데이터_분석정보_API_2025: https://www.data.go.kr/data/15143948/openapi.do
- 행정안전부_민원안내정보: https://www.data.go.kr/data/15069243/fileData.do
- 행정안전부_정부기능별분류체계: https://www.data.go.kr/data/15062615/fileData.do
- 법제처 국가법령정보 공동활용 API: https://open.law.go.kr/LSO/openApi/guideList.do
- 법제처_법령정보지식베이스 지능형 법령검색 시스템 검색: https://www.data.go.kr/data/15157095/openapi.do