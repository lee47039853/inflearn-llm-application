"""
사용자 인터페이스 관리 모듈
"""

from typing import List, Dict, Tuple


class UserInterface:
    """사용자 인터페이스 관리 클래스"""
    
    @staticmethod
    def get_user_choice(prompt: str, options: List[str]) -> str:
        """사용자로부터 선택받기"""
        print(f"\n{prompt}")
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        
        while True:
            try:
                choice = input(f"\n선택하세요 (1-{len(options)}): ").strip()
                if choice in [str(i) for i in range(1, len(options) + 1)]:
                    return choice
                else:
                    print(f"❌ 1-{len(options)} 중에서 선택하세요.")
            except KeyboardInterrupt:
                print("\n👋 프로그램을 종료합니다.")
                exit()
    
    @staticmethod
    def get_multiline_input() -> str:
        """여러 줄 입력 받기"""
        print("질문을 입력하세요 (여러 줄 가능, 'END'로 입력 완료, 'CANCEL'로 취소):")
        query_lines = []
        line_count = 0
        
        while True:
            line_count += 1
            line = input(f"[{line_count}]> ").strip()
            
            if line.lower() == 'end':
                break
            elif line.lower() in ['quit', 'exit', '종료', 'q']:
                print("\n👋 프로그램을 종료합니다.")
                exit()
            elif line.lower() in ['cancel', '취소', 'c']:
                print("❌ 질문 입력이 취소되었습니다.")
                return ""
            elif line.lower() == 'clear':
                query_lines = []
                line_count = 0
                print("🗑️  입력 내용이 지워졌습니다. 다시 입력하세요.")
                continue
            else:
                query_lines.append(line)
        
        return " ".join(query_lines).strip()
    
    @staticmethod
    def display_query_info(query: str, query_lines: List[str]):
        """질문 정보 표시"""
        print(f"\n📝 입력된 질문 ({len(query_lines)}줄):")
        print("-" * 40)
        for i, line in enumerate(query_lines, 1):
            print(f"  {i}. {line}")
        print("-" * 40)
        print(f"결합된 질문: {query}")
        print("-" * 40)
    
    @staticmethod
    def display_similarity_analysis(retrieved_docs_with_scores: List[Tuple]):
        """유사도 분석 결과 표시"""
        print("\n🔍 유사도 점수 분석:")
        print("=" * 60)
        for i, (doc, score) in enumerate(retrieved_docs_with_scores, 1):
            print(f"\n📋 문서 {i} (유사도 점수: {score:.4f}):")
            print(f"   페이지: {doc.metadata.get('page', 'N/A')}")
            print(f"   소스: {doc.metadata.get('source', 'N/A')}")
            print(f"   내용 길이: {len(doc.page_content)}자")
            print(f"   내용 미리보기: {doc.page_content[:200]}...")
            print("-" * 40)
        print("=" * 60)
        
        print("\n💡 유사도 점수 해석:")
        print("   - 점수가 낮을수록 더 유사함 (0에 가까울수록 유사)")
        print("   - 점수가 높을수록 덜 유사함")
        
        if retrieved_docs_with_scores:
            best_match = min(retrieved_docs_with_scores, key=lambda x: x[1])
            print(f"\n🏆 가장 유사한 문서: 문서 {retrieved_docs_with_scores.index(best_match) + 1} (점수: {best_match[1]:.4f})")
    
    @staticmethod
    def display_documents(retrieved_docs_with_scores: List[Tuple]):
        """전체 문서 내용 출력"""
        print("\n📄 전체 문서 내용:")
        print("=" * 60)
        for i, (doc, score) in enumerate(retrieved_docs_with_scores, 1):
            print(f"\n📋 문서 {i} (유사도 점수: {score:.4f}):")
            print(f"   내용: {doc.page_content}")
            print("-" * 40)
        print("=" * 60)
    
    @staticmethod
    def display_query_log(result: Dict):
        """최종 질의 로그 출력"""
        print("\n📝 최종 질의 로그:")
        print("=" * 60)
        print(f"원본 질문: {result['original_query']}")
        if result['optimization_used']:
            print(f"개선된 질문: {result['improved_query']}")
        else:
            print(f"개선된 질문: {result['improved_query']} (최적화 비활성화)")
        
        retrieved_docs = [doc for doc, _ in result['retrieved_docs']]
        print(f"검색된 문서 수: {len(retrieved_docs)}")
        print(f"컨텍스트 길이: {sum(len(doc.page_content) for doc in retrieved_docs)}자")
        print(f"평균 문서 길이: {sum(len(doc.page_content) for doc in retrieved_docs) // len(retrieved_docs) if retrieved_docs else 0}자")
        print("-" * 60)
        
        print("📚 검색된 문서 요약:")
        for i, (doc, score) in enumerate(result['retrieved_docs'], 1):
            print(f"  문서 {i} (유사도: {score:.4f}): {doc.page_content[:100]}...")
        print("=" * 60)
    
    @staticmethod
    def display_rag_info(query: str, improved_query: str, retrieved_docs: List, current_context: str):
        """RAG 체인에 전달되는 정보 표시"""
        print("🧠 RAG 체인 입력 정보:")
        print(f"  - 원본 질문: {query}")
        print(f"  - 개선된 질문: {improved_query}")
        if current_context:
            print(f"  - 이전 컨텍스트: {current_context[:100]}...")
        print(f"  - 컨텍스트 문서 수: {len(retrieved_docs)}")
        print(f"  - 총 컨텍스트 길이: {sum(len(doc.page_content) for doc in retrieved_docs)}자")
        print(f"  - 사용 모델: Gemini 2.0 Flash")
        print(f"  - Temperature: 0.9")
        print("=" * 60) 