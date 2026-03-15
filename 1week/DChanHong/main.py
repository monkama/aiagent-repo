import os
from openai import OpenAI
from dotenv import load_dotenv

def check_gemini_connection():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    model_name = "gemini-3-flash-preview"
    
    if not api_key:
        print("❌ 에러: .env 파일에서 GEMINI_API_KEY를 찾을 수 없습니다.")
        return

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        print(f"🔄 OpenAI 라이브러리로 {model_name} 연결 시도 중...")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "당신은 유능한 조력자입니다."},
                {"role": "user", "content": "성공이라고 짧게 답변해 주세요."}
            ],
            max_tokens=50
        )
        
        # [중요] 전체 응답 구조를 출력해서 눈으로 확인합니다.
        # print("-" * 30)
        # print(response) 
        # print("-" * 30)

        # 안전하게 내용 가져오기
        message = response.choices[0].message
        if message.content:
            print(f"✅ 연결 성공! 응답: {message.content.strip()}")
        else:
            # 내용이 없을 때 원인(finish_reason) 파악
            reason = response.choices[0].finish_reason
            print(f"⚠️ 모델이 응답을 비웠습니다. 이유(finish_reason): {reason}")
            if reason == "safety":
                print("💡 구글 안전 필터에 의해 차단되었습니다.")
            
    except Exception as e:
        print(f"❌ 연결 실패: {str(e)}")

if __name__ == "__main__":
    check_gemini_connection()