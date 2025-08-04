#!/usr/bin/env python3
"""
Ollama Gemma3 모델을 사용한 기본 LLM 예제
"""

from langchain_ollama import ChatOllama

def main():
    try:
        # Ollama 모델 초기화
        llm = ChatOllama(model="gemma3")
        
        # 질문 실행
        question = 'ollama에서 무료로 사용할 수 있는 gemma3 모델에 대해서 알려주세요.'
        print(f"🤖 질문: {question}")
        print("-" * 50)
        
        ai_message = llm.invoke(question)
        print("✅ 응답:")
        print(ai_message.content)
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("   Ollama가 설치되어 있고 실행 중인지 확인하세요.")
        print("   설치: https://ollama.ai/")
        print("   실행: ollama serve")

if __name__ == "__main__":
    main()