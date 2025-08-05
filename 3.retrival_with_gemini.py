#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) 시스템 예제
문서 기반 질의응답 시스템 - 대화 히스토리 포함
"""

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.chains import RetrievalQA
from pathlib import Path
import shutil
import logging
import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime

class EnhancedRAGSystem:
    """향상된 RAG 시스템 클래스"""
    
    def __init__(self, database, llm):
        self.database = database
        self.llm = llm
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Tuple]:
        """문서 검색 수행"""
        try:
            results = self.database.similarity_search_with_score(query, k=top_k)
            return results
        except Exception as e:
            print(f"⚠️  검색 중 오류: {e}")
            return []

class ConversationHistory:
    """대화 히스토리 관리 클래스"""
    
    def __init__(self, max_history: int = 10):
        self.history = []
        self.max_history = max_history
        self.current_context = ""
        self.history_enabled = True  # 히스토리 활성화 상태
    
    def add_exchange(self, question: str, answer: str, retrieved_docs: List = None):
        """질문-답변 교환 추가"""
        if not self.history_enabled:
            return  # 히스토리가 비활성화된 경우 저장하지 않음
        
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs if retrieved_docs else [],
            'context_summary': self._extract_context(answer)
        }
        
        self.history.append(exchange)
        
        # 최대 히스토리 수 유지
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # 컨텍스트 업데이트
        self._update_context()
    
    def _extract_context(self, answer: str) -> str:
        """답변에서 핵심 컨텍스트 추출"""
        # 간단한 키워드 추출 (실제로는 더 정교한 방법 사용 가능)
        keywords = ['소득세', '세율', '공제', '과세표준', '연봉', '거주자']
        context_parts = []
        
        for keyword in keywords:
            if keyword in answer:
                # 키워드 주변 문장 추출
                sentences = answer.split('.')
                for sentence in sentences:
                    if keyword in sentence:
                        context_parts.append(sentence.strip())
        
        return '. '.join(context_parts[:3])  # 최대 3개 문장
    
    def _update_context(self):
        """전체 컨텍스트 업데이트"""
        if not self.history:
            self.current_context = ""
            return
        
        # 최근 3개 교환의 컨텍스트를 결합
        recent_contexts = []
        for exchange in self.history[-3:]:
            if exchange['context_summary']:
                recent_contexts.append(exchange['context_summary'])
        
        self.current_context = " ".join(recent_contexts)
    
    def get_context_for_query(self, new_question: str) -> str:
        """새 질문에 대한 컨텍스트 생성"""
        if not self.history_enabled or not self.history:
            return new_question
        
        # 이전 대화 컨텍스트와 새 질문 결합
        context_query = f"이전 대화: {self.current_context}\n\n새 질문: {new_question}"
        return context_query
    
    def show_history(self, limit: int = 5):
        """대화 히스토리 표시"""
        if not self.history_enabled:
            print("🚫 대화 히스토리가 비활성화되어 있습니다.")
            return
        
        if not self.history:
            print("📝 대화 히스토리가 없습니다.")
            return
        
        print(f"\n📝 최근 대화 히스토리 (최대 {limit}개):")
        print("=" * 60)
        
        for i, exchange in enumerate(self.history[-limit:], 1):
            print(f"\n💬 교환 {i}:")
            print(f"  질문: {exchange['question']}")
            print(f"  답변: {exchange['answer'][:100]}...")
            print(f"  컨텍스트: {exchange['context_summary']}")
            print("-" * 40)
        
        print("=" * 60)
    
    def clear_history(self):
        """대화 히스토리 초기화"""
        removed_count = len(self.history)
        self.history = []
        self.current_context = ""
        print(f"🗑️  대화 히스토리가 초기화되었습니다. (제거된 대화: {removed_count}개)")
    
    def reset_conversation(self):
        """대화 완전 초기화 (히스토리 + 컨텍스트)"""
        removed_count = len(self.history)
        self.history = []
        self.current_context = ""
        self.history_enabled = True  # 기본 상태로 복원
        print(f"🔄 대화가 완전히 초기화되었습니다.")
        print(f"  - 제거된 대화: {removed_count}개")
        print(f"  - 히스토리 상태: 활성화로 복원")
        print(f"  - 컨텍스트: 초기화됨")
    
    def clear_and_disable(self):
        """히스토리 초기화 후 비활성화"""
        removed_count = len(self.history)
        self.history = []
        self.current_context = ""
        self.history_enabled = False
        print(f"🚫 대화 히스토리가 초기화되고 비활성화되었습니다.")
        print(f"  - 제거된 대화: {removed_count}개")
        print(f"  - 향후 대화는 저장되지 않습니다.")
    
    def disable_history(self):
        """대화 히스토리 비활성화"""
        self.history_enabled = False
        print("🚫 대화 히스토리가 비활성화되었습니다.")
    
    def enable_history(self):
        """대화 히스토리 활성화"""
        self.history_enabled = True
        print("✅ 대화 히스토리가 활성화되었습니다.")
    
    def remove_last_exchange(self):
        """마지막 대화 교환 제거"""
        if self.history:
            removed = self.history.pop()
            self._update_context()
            print(f"🗑️  마지막 대화가 제거되었습니다:")
            print(f"  질문: {removed['question']}")
            print(f"  답변: {removed['answer'][:50]}...")
        else:
            print("❌ 제거할 대화가 없습니다.")
    
    def remove_exchange_by_index(self, index: int):
        """인덱스로 특정 대화 교환 제거"""
        if 0 <= index < len(self.history):
            removed = self.history.pop(index)
            self._update_context()
            print(f"🗑️  대화 {index + 1}이 제거되었습니다:")
            print(f"  질문: {removed['question']}")
            print(f"  답변: {removed['answer'][:50]}...")
        else:
            print(f"❌ 인덱스 {index + 1}의 대화가 존재하지 않습니다.")
    
    def get_history_status(self) -> Dict:
        """히스토리 상태 정보 반환"""
        return {
            'enabled': self.history_enabled,
            'count': len(self.history),
            'max_history': self.max_history,
            'has_context': bool(self.current_context)
        }
    
    def get_relevant_context(self, new_question: str) -> str:
        """새 질문과 관련된 이전 컨텍스트 찾기"""
        if not self.history_enabled or not self.history:
            return ""
        
        # 새 질문의 키워드 추출
        question_keywords = self._extract_keywords(new_question)
        
        relevant_contexts = []
        for exchange in self.history:
            # 이전 질문이나 답변에서 관련 키워드가 있는지 확인
            exchange_text = f"{exchange['question']} {exchange['answer']}"
            exchange_keywords = self._extract_keywords(exchange_text)
            
            # 키워드 겹침 확인
            common_keywords = set(question_keywords) & set(exchange_keywords)
            if common_keywords:
                relevant_contexts.append(exchange['context_summary'])
        
        return " ".join(relevant_contexts[-2:])  # 최근 2개 관련 컨텍스트
    
    def _extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 키워드 추출"""
        keywords = ['소득세', '세율', '공제', '과세표준', '연봉', '거주자', '종합소득', '법인세', '부가가치세']
        found_keywords = []
        
        for keyword in keywords:
            if keyword in text:
                found_keywords.append(keyword)
        
        return found_keywords

def check_existing_database():
    """기존 Chroma 데이터베이스 존재 여부 확인"""
    chroma_dir = Path("./chroma")
    if chroma_dir.exists() and any(chroma_dir.iterdir()):
        return True
    return False

def clear_database():
    """기존 데이터베이스 삭제"""
    chroma_dir = Path("./chroma")
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)
        print("🗑️  기존 데이터베이스가 삭제되었습니다.")

def get_user_choice():
    """사용자로부터 데이터베이스 처리 방향 선택받기"""
    print("\n📋 데이터베이스 처리 옵션:")
    print("1. 기존 데이터베이스 재사용 (빠름)")
    print("2. 새로운 데이터베이스 생성 (처음 실행)")
    print("3. 기존 데이터베이스 삭제 후 새로 생성")
    
    while True:
        try:
            choice = input("\n선택하세요 (1-3): ").strip()
            if choice in ['1', '2', '3']:
                return choice
            else:
                print("❌ 1, 2, 3 중에서 선택하세요.")
        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            exit()





def main():
    # LangChain 디버그 로깅 활성화 (선택사항)
    # logging.basicConfig(level=logging.DEBUG)
    
    # .env 파일에서 환경 변수 로드
    load_dotenv()
    
    # Google API 키 확인
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 GOOGLE_API_KEY=your_api_key_here를 추가하세요.")
        return
    
    try:
        # 기존 데이터베이스 확인
        has_existing_db = check_existing_database()
        
        if has_existing_db:
            choice = get_user_choice()
            
            if choice == '3':
                clear_database()
                has_existing_db = False
        else:
            print("📄 새로운 데이터베이스를 생성합니다.")
        
        # 임베딩 모델 초기화
        print("\n🔧 임베딩 모델 초기화 중...")
        
        # 사용자로부터 임베딩 모델 선택받기
        print("\n📊 임베딩 모델 선택:")
        print("1. 한국어 특화 모델 (ko-sroberta-multitask) - 추천")
        print("2. Google Gemini 임베딩 (기존)")
        
        while True:
            try:
                embedding_choice = input("\n선택하세요 (1-2): ").strip()
                if embedding_choice in ['1', '2']:
                    break
                else:
                    print("❌ 1, 2 중에서 선택하세요.")
            except KeyboardInterrupt:
                print("\n👋 프로그램을 종료합니다.")
                exit()
        
        if embedding_choice == '1':
            # 한국어 특화 임베딩 모델 사용
            print("🇰🇷 한국어 특화 임베딩 모델 로딩 중...")
            model_name = "jhgan/ko-sroberta-multitask"
            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            embedding = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            print("✅ 한국어 특화 임베딩 모델 로딩 완료")
        else:
            # Google Gemini 임베딩 사용 (기존)
            print("🌐 Google Gemini 임베딩 모델 로딩 중...")
            embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            print("✅ Google Gemini 임베딩 모델 로딩 완료") 

        # Chroma 벡터 데이터베이스 초기화
        database = Chroma(
            collection_name='chroma-tax', 
            persist_directory="./chroma", 
            embedding_function=embedding
        )

        # 데이터베이스 처리
        if has_existing_db and choice == '1':
            print("📚 기존 벡터 데이터베이스 재사용 중...")
            
            # 기존 컬렉션의 문서 수 확인
            try:
                collection_count = database._collection.count()
                print(f"✅ 기존 데이터베이스에서 {collection_count}개의 문서를 찾았습니다.")
            except Exception as e:
                print(f"⚠️  기존 데이터베이스 정보 확인 중 오류: {e}")
                print("   새로운 데이터베이스를 생성합니다.")
                has_existing_db = False
        
        if not has_existing_db or choice == '2':
            print("📄 문서 로딩 중...")
            
            # 텍스트 분할 설정
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
            )

            # Word 문서 로드 및 분할
            loader = Docx2txtLoader("./tax.docx")
            document_list = loader.load_and_split(text_splitter=text_splitter)
            
            print(f"✅ {len(document_list)}개의 문서 청크로 분할 완료")

            # 문서를 벡터 데이터베이스에 저장
            database.add_documents(document_list)
            print("✅ 벡터 데이터베이스에 문서 저장 완료")

        # LLM 모델 초기화
        llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.9)
        prompt = hub.pull("rlm/rag-prompt")

        # RAG 체인 생성
        qa_chain = RetrievalQA.from_chain_type(
            llm, 
            retriever=database.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )

        # 향상된 RAG 시스템 초기화
        print("\n🔧 향상된 RAG 시스템 초기화 중...")
        enhanced_rag = EnhancedRAGSystem(database, llm)
        print("✅ 향상된 RAG 시스템 초기화 완료")
        
        # 대화 히스토리 초기화
        conversation_history = ConversationHistory()

        # 질문-답변 루프
        while True:
            # 사용자로부터 질문 입력받기
            print("\n🤖 질문을 입력하세요:")
            print("   (여러 줄 입력 가능: 각 줄 입력 후 Enter, 'END'로 완료, 'CANCEL'로 취소)")
            print("   (예: 제 55조의 종합소득 과제표준 기준으로 거주자의 연봉이 5천만원 인 경우")
            print("        해당 거주자의 소득세는 얼마인가요?)")
            print("   (입력 중 'CLEAR'로 내용 지우기, 'CANCEL'로 입력 취소)")
            print("   (입력 중 히스토리 제어: disable_history, enable_history, clear_history, reset_conversation, clear_and_disable)")
            print("   (종료하려면 'quit' 또는 'exit' 입력)")
            print("   (대화 히스토리: show_history, clear_history, show_context)")
            print("   (히스토리 제어: disable_history, enable_history, remove_last, remove_history:번호, history_status)")
            print("   (완전 초기화: reset_conversation, clear_and_disable)")
            print("-" * 50)
            
            try:
                # 여러 줄 질문 입력 받기
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
                        return
                    elif line.lower() in ['cancel', '취소', 'c']:
                        print("❌ 질문 입력이 취소되었습니다.")
                        break
                    elif line.lower() == 'clear':
                        query_lines = []
                        line_count = 0
                        print("🗑️  입력 내용이 지워졌습니다. 다시 입력하세요.")
                        continue
                    # 히스토리 제어 명령어들
                    elif line.lower() == 'disable_history':
                        conversation_history.disable_history()
                        continue
                    elif line.lower() == 'enable_history':
                        conversation_history.enable_history()
                        continue
                    elif line.lower() == 'clear_history':
                        conversation_history.clear_history()
                        continue
                    elif line.lower() == 'reset_conversation':
                        conversation_history.reset_conversation()
                        continue
                    elif line.lower() == 'clear_and_disable':
                        conversation_history.clear_and_disable()
                        continue
                    elif line.lower() == 'remove_last':
                        conversation_history.remove_last_exchange()
                        continue
                    elif line.lower().startswith('remove_history:'):
                        try:
                            index = int(line.split(':', 1)[1].strip()) - 1
                            conversation_history.remove_exchange_by_index(index)
                        except (ValueError, IndexError):
                            print("❌ 형식: remove_history:번호 (예: remove_history:1)")
                        continue
                    elif line.lower() == 'history_status':
                        status = conversation_history.get_history_status()
                        print(f"\n📊 대화 히스토리 상태:")
                        print(f"  활성화: {'✅' if status['enabled'] else '❌'}")
                        print(f"  저장된 대화: {status['count']}개")
                        print(f"  최대 저장: {status['max_history']}개")
                        print(f"  컨텍스트: {'있음' if status['has_context'] else '없음'}")
                        continue
                    elif line.lower() == 'show_history':
                        conversation_history.show_history()
                        continue
                    elif line.lower() == 'show_context':
                        if conversation_history.current_context:
                            print(f"\n📝 현재 대화 컨텍스트:")
                            print(f"{conversation_history.current_context}")
                        else:
                            print("📝 현재 대화 컨텍스트가 없습니다.")
                        continue
                    else:
                        query_lines.append(line)
                
                # CANCEL로 취소된 경우 다음 루프로
                if not query_lines:
                    continue
                
                # 여러 줄을 하나의 질문으로 결합
                query = " ".join(query_lines).strip()
                
                if not query:
                    print("❌ 질문을 입력해주세요.")
                    continue
                
                # 입력된 질문 확인
                print(f"\n📝 입력된 질문 ({len(query_lines)}줄):")
                print("-" * 40)
                for i, line in enumerate(query_lines, 1):
                    print(f"  {i}. {line}")
                print("-" * 40)
                print(f"결합된 질문: {query}")
                print("-" * 40)
                
                # 히스토리 상태 표시
                status = conversation_history.get_history_status()
                if status['enabled']:
                    print(f"💾 히스토리 상태: 활성화 (저장됨: {status['count']}개)")
                else:
                    print(f"🚫 히스토리 상태: 비활성화 (저장되지 않음)")
                print("-" * 40)
                

                
                # 이전 대화 컨텍스트 확인 및 적용
                relevant_context = conversation_history.get_relevant_context(query)
                if relevant_context:
                    print(f"\n🔄 관련 이전 컨텍스트 발견:")
                    print(f"  {relevant_context}")
                    print("-" * 50)
                
                print(f"\n🤖 입력된 질문: {query}")
                print("-" * 50)
                
                # 문서 검색 수행
                print("🔍 문서 검색 수행 중...")
                retrieved_docs_with_scores = enhanced_rag.search_documents(query, top_k=5)
                retrieved_docs = [doc for doc, _ in retrieved_docs_with_scores]
                print(f"📚 검색된 관련 문서 수: {len(retrieved_docs)}")
                
                # 유사도 점수 분석
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
                
                # 유사도 점수 해석
                print("\n💡 유사도 점수 해석:")
                print("   - 점수가 낮을수록 더 유사함 (0에 가까울수록 유사)")
                print("   - 점수가 높을수록 덜 유사함")
                
                # 가장 유사한 문서 찾기
                best_match = min(retrieved_docs_with_scores, key=lambda x: x[1])
                print(f"\n🏆 가장 유사한 문서: 문서 {retrieved_docs_with_scores.index(best_match) + 1} (점수: {best_match[1]:.4f})")
                
                # 전체 문서 내용 출력
                print("\n📄 전체 문서 내용:")
                print("=" * 60)
                for i, (doc, score) in enumerate(retrieved_docs_with_scores, 1):
                    print(f"\n📋 문서 {i} (유사도 점수: {score:.4f}):")
                    print(f"   내용: {doc.page_content}")
                    print("-" * 40)
                print("=" * 60)

                # 실제 질의응답 실행
                print("🧠 AI 응답 생성 중...")
                
                # 최종 질의 로그 출력
                print("\n📝 최종 질의 로그:")
                print("=" * 60)
                print(f"원본 질문: {query}")
                print(f"검색된 문서 수: {len(retrieved_docs)}")
                print(f"컨텍스트 길이: {sum(len(doc.page_content) for doc in retrieved_docs)}자")
                print(f"평균 문서 길이: {sum(len(doc.page_content) for doc in retrieved_docs) // len(retrieved_docs) if retrieved_docs else 0}자")
                print("-" * 60)
                
                # 검색된 문서 요약
                print("📚 검색된 문서 요약:")
                for i, (doc, score) in enumerate(retrieved_docs_with_scores, 1):
                    print(f"  문서 {i} (유사도: {score:.4f}): {doc.page_content[:100]}...")
                print("=" * 60)
                
                # 컨텍스트가 포함된 질문 생성
                context_query = conversation_history.get_context_for_query(query)
                
                # RAG 체인에 전달되는 정보
                print("🧠 RAG 체인 입력 정보:")
                print(f"  - 원본 질문: {query}")
                if conversation_history.current_context:
                    print(f"  - 이전 컨텍스트: {conversation_history.current_context[:100]}...")
                print(f"  - 컨텍스트 문서 수: {len(retrieved_docs)}")
                print(f"  - 총 컨텍스트 길이: {sum(len(doc.page_content) for doc in retrieved_docs)}자")
                print(f"  - 사용 모델: Gemini 2.0 Flash")
                print(f"  - Temperature: 0.9")
                print("=" * 60)
                
                # RAG 체인 실행 (컨텍스트 포함)
                ai_message = qa_chain({"query": context_query})

                print("\n" + "=" * 50)
                print("✅ 최종 응답:")
                print(ai_message['result'])
                print("=" * 50)
                
                # 대화 히스토리에 교환 추가
                conversation_history.add_exchange(query, ai_message['result'], retrieved_docs_with_scores)

            except KeyboardInterrupt:
                print("\n👋 프로그램을 종료합니다.")
                return
            except Exception as e:
                print(f"❌ 질문 처리 중 오류 발생: {e}")
                print("   다시 시도해주세요.")
                continue
        
    except FileNotFoundError:
        print("❌ tax.docx 파일을 찾을 수 없습니다.")
        print("   프로젝트 루트에 tax.docx 파일이 있는지 확인하세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("   API 키가 올바른지 확인하세요.")

if __name__ == "__main__":
    main()