#!/usr/bin/env python3
"""
Google Gemini 모델을 사용한 기본 LLM 예제
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    # .env 파일에서 환경 변수 로드
    load_dotenv(".env")
    
    # Google API 키 확인
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 GOOGLE_API_KEY=your_api_key_here를 추가하세요.")
        return
    
    try:
        # Gemini 모델 초기화
        llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.9)
        print("✅ Gemini 모델이 초기화되었습니다.")
        
        # 질문-답변 루프
        while True:
            # 사용자로부터 질문 입력받기
            print("\n🤖 질문을 입력하세요:")
            print("   (예: langchain_google_genai에서 무료로 사용할 수 있는 Gemini 모델을 알려주세요.)")
            print("   (종료하려면 'quit' 또는 'exit' 입력)")
            print("-" * 50)
            
            try:
                print("질문을 입력하세요 (여러 줄 가능, 입력 완료 후 'END' 입력):")
                print("   (종료하려면 'quit' 또는 'exit' 입력)")
                print("-" * 50)
                
                lines = []
                while True:
                    try:
                        line = input()
                        if line.strip().lower() in ['quit', 'exit', '종료', 'q']:
                            print("\n👋 프로그램을 종료합니다.")
                            return
                        elif line.strip().upper() == 'END':
                            break
                        lines.append(line)
                    except KeyboardInterrupt:
                        print("\n👋 프로그램을 종료합니다.")
                        return
                
                question = '\n'.join(lines).strip()
                if not question:
                    print("❌ 질문을 입력해주세요.")
                    continue
                
                print(f"\n🤖 입력된 질문:")
                print("-" * 50)
                print(question)
                print("-" * 50)
                
                # AI 응답 생성
                print("🧠 AI 응답 생성 중...")
                ai_message = llm.invoke(question)
                
                print("\n" + "=" * 50)
                print("✅ AI 응답:")
                print(ai_message.content)
                print("=" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 질문 처리 중 오류 발생: {e}")
                print("   다시 시도해주세요.")
                continue
        
    except Exception as e:
        print(f"❌ 모델 초기화 중 오류 발생: {e}")
        print("   API 키가 올바른지 확인하세요.")

if __name__ == "__main__":
    main()