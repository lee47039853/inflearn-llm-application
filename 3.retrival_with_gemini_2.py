#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) 시스템 예제
문서 기반 질의응답 시스템 - 대화 히스토리 및 LLM 쿼리 개선 포함

리팩토링된 버전: 모듈화된 구조로 중복 제거 및 클래스 분리
패키지 구조: retrieval 패키지 사용
"""

from retrieval import RAGApplication


def main():
    """메인 함수"""
    try:
        app = RAGApplication()
        app.run()
    except FileNotFoundError:
        print("❌ tax.docx 파일을 찾을 수 없습니다.")
        print("   프로젝트 루트에 tax.docx 파일이 있는지 확인하세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("   API 키가 올바른지 확인하세요.")


if __name__ == "__main__":
    main()