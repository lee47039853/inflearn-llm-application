"""
RAG 애플리케이션 메인 모듈
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from .config import Config
from .embedding_manager import EmbeddingManager
from .database_manager import DatabaseManager
from .conversation_history import ConversationHistory
from .enhanced_rag import EnhancedRAGSystem
from .user_interface import UserInterface
from .command_processor import CommandProcessor


class RAGApplication:
    """RAG 애플리케이션 메인 클래스"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.conversation_history = ConversationHistory()
        self.enhanced_rag = None
        self.command_processor = None
    
    def initialize(self):
        """애플리케이션 초기화"""
        load_dotenv()
        
        # Google API 키 확인
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ GOOGLE_API_KEY가 설정되지 않았습니다.")
            print("   .env 파일에 GOOGLE_API_KEY=your_api_key_here를 추가하세요.")
            return False
        
        # 데이터베이스 처리
        has_existing_db = self.db_manager.check_existing_database()
        
        if has_existing_db:
            choice = UserInterface.get_user_choice(
                "📋 데이터베이스 처리 옵션:",
                [
                    "기존 데이터베이스 재사용 (빠름)",
                    "새로운 데이터베이스 생성 (처음 실행)",
                    "기존 데이터베이스 삭제 후 새로 생성"
                ]
            )
            
            if choice == '3':
                self.db_manager.clear_database()
                has_existing_db = False
        else:
            print("📄 새로운 데이터베이스를 생성합니다.")
        
        # 임베딩 모델 선택
        embedding_choice = UserInterface.get_user_choice(
            "📊 임베딩 모델 선택:",
            [
                "한국어 특화 모델 (ko-sroberta-multitask) - 추천",
                "Google Gemini 임베딩 (기존)"
            ]
        )
        
        # 임베딩 모델 초기화
        print("\n🔧 임베딩 모델 초기화 중...")
        embedding = EmbeddingManager.create_embedding(embedding_choice)
        
        # 데이터베이스 초기화
        database = self.db_manager.create_database(embedding)
        
        # 데이터베이스 처리
        if has_existing_db and choice == '1':
            print("📚 기존 벡터 데이터베이스 재사용 중...")
            try:
                collection_count = database._collection.count()
                print(f"✅ 기존 데이터베이스에서 {collection_count}개의 문서를 찾았습니다.")
            except Exception as e:
                print(f"⚠️  기존 데이터베이스 정보 확인 중 오류: {e}")
                print("   새로운 데이터베이스를 생성합니다.")
                has_existing_db = False
        
        if not has_existing_db or choice == '2':
            document_list = self.db_manager.load_documents()
            database.add_documents(document_list)
            print("✅ 벡터 데이터베이스에 문서 저장 완료")
        
        # LLM 모델 초기화
        llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.9)
        
        # 쿼리 최적화 기능 선택
        optimization_choice = UserInterface.get_user_choice(
            "🔧 쿼리 최적화 기능 설정:",
            [
                "쿼리 최적화 활성화 (LLM을 통한 질문 개선) - 추천",
                "쿼리 최적화 비활성화 (원본 질문 그대로 사용)"
            ]
        )
        
        use_optimization = (optimization_choice == '1')
        
        # 향상된 RAG 시스템 초기화
        print("\n🔧 향상된 RAG 시스템 초기화 중...")
        self.enhanced_rag = EnhancedRAGSystem(database, llm, use_query_optimization=use_optimization)
        print("✅ 향상된 RAG 시스템 초기화 완료")
        
        if use_optimization:
            print("✅ 쿼리 최적화 기능이 활성화되었습니다.")
        else:
            print("🚫 쿼리 최적화 기능이 비활성화되었습니다.")
        
        # 명령어 프로세서 초기화
        self.command_processor = CommandProcessor(self.conversation_history, self.enhanced_rag)
        
        return True
    
    def run(self):
        """애플리케이션 실행"""
        if not self.initialize():
            print("❌ 애플리케이션 초기화에 실패했습니다.")
            return
        
        print("\n🤖 질문을 입력하세요:")
        print("   (여러 줄 입력 가능: 각 줄 입력 후 Enter, 'END'로 완료, 'CANCEL'로 취소)")
        print("   (예: 제 55조의 종합소득 과제표준 기준으로 거주자의 연봉이 5천만원 인 경우")
        print("        해당 거주자의 소득세는 얼마인가요?)")
        print("   (입력 중 'CLEAR'로 내용 지우기, 'CANCEL'로 입력 취소)")
        print("   (입력 중 히스토리 제어: disable_history, enable_history, clear_history, reset_conversation, clear_and_disable)")
        print("   (쿼리 최적화 제어: toggle_optimization, optimization_status)")
        print("   (종료하려면 'quit' 또는 'exit' 입력)")
        print("   (대화 히스토리: show_history, clear_history, show_context)")
        print("   (히스토리 제어: disable_history, enable_history, remove_last, remove_history:번호, history_status)")
        print("   (완전 초기화: reset_conversation, clear_and_disable)")
        print("-" * 50)
        
        while True:
            try:
                # 여러 줄 질문 입력 받기
                query = UserInterface.get_multiline_input()
                
                if not query:
                    continue
                
                # 입력된 질문 확인
                query_lines = query.split()
                UserInterface.display_query_info(query, query_lines)
                
                # 히스토리 상태 표시
                status = self.conversation_history.get_history_status()
                if status['enabled']:
                    print(f"💾 히스토리 상태: 활성화 (저장됨: {status['count']}개)")
                else:
                    print(f"🚫 히스토리 상태: 비활성화 (저장되지 않음)")
                print("-" * 40)
                
                # 이전 대화 컨텍스트 확인 및 적용
                relevant_context = self.conversation_history.get_relevant_context(query)
                if relevant_context:
                    print(f"\n🔄 관련 이전 컨텍스트 발견:")
                    print(f"  {relevant_context}")
                    print("-" * 50)
                
                print(f"\n🤖 입력된 질문: {query}")
                print("-" * 50)
                
                # 쿼리 개선 및 RAG 처리
                if self.enhanced_rag is None:
                    print("❌ RAG 시스템이 초기화되지 않았습니다.")
                    continue
                result = self.enhanced_rag.process_query_with_improvement(query)
                
                # 결과 추출
                improved_query = result['improved_query']
                retrieved_docs_with_scores = result['retrieved_docs']
                retrieved_docs = [doc for doc, _ in retrieved_docs_with_scores]
                ai_response = result['response']
                
                # 유사도 점수 분석
                UserInterface.display_similarity_analysis(retrieved_docs_with_scores)
                
                # 전체 문서 내용 출력
                UserInterface.display_documents(retrieved_docs_with_scores)
                
                # 최종 질의 로그 출력
                UserInterface.display_query_log(result)
                
                # RAG 체인에 전달되는 정보
                UserInterface.display_rag_info(query, improved_query, retrieved_docs, self.conversation_history.current_context)
                
                # AI 응답 출력
                ai_message = {'result': ai_response}
                print("\n" + "=" * 50)
                print("✅ 최종 응답:")
                print(ai_message['result'])
                print("=" * 50)
                
                # 대화 히스토리에 교환 추가
                self.conversation_history.add_exchange(query, ai_message['result'], retrieved_docs_with_scores)
                
            except KeyboardInterrupt:
                print("\n👋 프로그램을 종료합니다.")
                return
            except Exception as e:
                print(f"❌ 질문 처리 중 오류 발생: {e}")
                print("   다시 시도해주세요.")
                continue 