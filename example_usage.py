#!/usr/bin/env python3
"""
Retrieval 패키지 사용 예제

이 파일은 retrieval 패키지의 다양한 사용 방법을 보여줍니다.
"""

# 방법 1: 전체 패키지에서 주요 클래스들 import
from retrieval import (
    Config, 
    EmbeddingManager, 
    DatabaseManager, 
    ConversationHistory,
    EnhancedRAGSystem,
    UserInterface,
    CommandProcessor,
    RAGApplication
)

# 방법 2: 특정 모듈에서만 import
from retrieval.config import Config
from retrieval.embedding_manager import EmbeddingManager

# 방법 3: 패키지 정보 확인
from retrieval import get_package_info

def main():
    """패키지 사용 예제"""
    
    # 패키지 정보 출력
    print("📦 Retrieval 패키지 정보:")
    package_info = get_package_info()
    for key, value in package_info.items():
        print(f"  {key}: {value}")
    print()
    
    # 설정 확인
    print("⚙️  설정 정보:")
    print(f"  청크 크기: {Config.CHUNK_SIZE}")
    print(f"  청크 오버랩: {Config.CHUNK_OVERLAP}")
    print(f"  최대 히스토리: {Config.MAX_HISTORY}")
    print(f"  키워드 수: {len(Config.KEYWORDS)}")
    print()
    
    # 대화 히스토리 사용 예제
    print("💬 대화 히스토리 사용 예제:")
    history = ConversationHistory(max_history=5)
    
    # 대화 추가
    history.add_exchange(
        question="소득세는 어떻게 계산하나요?",
        answer="소득세는 과세표준에 세율을 적용하여 계산합니다."
    )
    
    # 히스토리 상태 확인
    status = history.get_history_status()
    print(f"  활성화: {status['enabled']}")
    print(f"  저장된 대화: {status['count']}개")
    print()
    
    # 사용자 인터페이스 사용 예제
    print("🖥️  사용자 인터페이스 사용 예제:")
    options = ["옵션 1", "옵션 2", "옵션 3"]
    print("사용자 선택 옵션:", options)
    print("(실제로는 UserInterface.get_user_choice()를 사용)")
    print()
    
    print("✅ 패키지 사용 예제 완료!")
    print("\n실제 사용을 위해서는 다음 명령어를 실행하세요:")
    print("python 3.retrival_with_gemini_2.py")

if __name__ == "__main__":
    main() 