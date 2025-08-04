#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) 시스템 예제
문서 기반 질의응답 시스템
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

def get_debug_mode():
    """디버그 모드 선택"""
    print("\n🔍 디버그 옵션:")
    print("1. 기본 모드 (프롬프트 미표시)")
    print("2. 프롬프트 확인 모드 (LLM 전달 프롬프트 표시)")
    
    while True:
        try:
            choice = input("\n선택하세요 (1-2): ").strip()
            if choice in ['1', '2']:
                return choice == '2'
            else:
                print("❌ 1, 2 중에서 선택하세요.")
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
        
        # 디버그 모드 선택
        debug_mode = get_debug_mode()
        
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
        


        # 질문-답변 루프
        while True:
            # 사용자로부터 질문 입력받기
            print("\n🤖 질문을 입력하세요:")
            print("   (예: 제 55조의 종합소득 과제표준 기준으로 거주자의 연봉이 5천만원 인 경우 해당 거주자의 소득세는 얼마인가요?)")
            print("   (종료하려면 'quit' 또는 'exit' 입력)")
            print("-" * 50)
            
            try:
                query = input("질문: ").strip()
                if not query:
                    print("❌ 질문을 입력해주세요.")
                    continue
                
                if query.lower() in ['quit', 'exit', '종료', 'q']:
                    print("\n👋 프로그램을 종료합니다.")
                    return
                
                print(f"\n🤖 입력된 질문: {query}")
                print("-" * 50)
                
                # 유사도 검색으로 관련 문서 검색 (점수 포함)
                retrieved_docs_with_scores = database.similarity_search_with_score(query)
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

                # 디버그 모드에 따른 프롬프트 확인
                if debug_mode:
                    print("\n🔍 LLM에 전달되는 프롬프트 확인:")
                    print("=" * 60)
                    
                    # 검색된 문서들을 context로 결합
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    # 간단한 프롬프트 구조 표시
                    print("📋 프롬프트 구조:")
                    print("Context: [검색된 문서들]")
                    print("Question: [사용자 질문]")
                    print("Answer: [AI 응답]")
                    print("-" * 40)
                    
                    # 실제 전달되는 내용 미리보기
                    print("📝 실제 전달되는 내용 미리보기:")
                    print(f"질문: {query}")
                    print(f"검색된 문서 수: {len(retrieved_docs)}개")
                    print(f"Context 길이: {len(context)}자")
                    print(f"Context 미리보기: {context[:300]}...")
                    print("=" * 60)
                    
                    # 프롬프트 정보
                    print(f"📊 프롬프트 정보:")
                    print(f"   - 질문 길이: {len(query)}자")
                    print(f"   - 검색된 문서 수: {len(retrieved_docs)}개")
                    print(f"   - Context 길이: {len(context)}자")
                    print(f"   - 총 전달 데이터: {len(query) + len(context)}자")
                    
                    # 상세 데이터 분석
                    print(f"\n📈 상세 데이터 분석:")
                    print(f"   - 각 문서별 길이:")
                    for i, doc in enumerate(retrieved_docs, 1):
                        doc_length = len(doc.page_content)
                        print(f"     문서 {i}: {doc_length}자")
                    
                    # 토큰 수 추정 (대략적 계산)
                    estimated_tokens = (len(query) + len(context)) // 4  # 대략 1토큰 = 4자
                    print(f"   - 추정 토큰 수: 약 {estimated_tokens:,} 토큰")
                    
                    # 데이터 효율성 분석
                    avg_doc_length = len(context) / len(retrieved_docs) if retrieved_docs else 0
                    print(f"   - 평균 문서 길이: {avg_doc_length:.0f}자")
                    print(f"   - 데이터 효율성: {len(query)}자 질문 → {len(context)}자 컨텍스트")
                    
                    # 메모리 사용량 추정 (UTF-8 기준)
                    memory_usage = (len(query) + len(context)) * 4  # UTF-8은 최대 4바이트
                    print(f"   - 추정 메모리 사용량: 약 {memory_usage:,} 바이트 ({memory_usage/1024:.1f} KB)")
                    
                    # API 비용 추정 (Gemini 기준)
                    input_tokens = estimated_tokens
                    output_tokens_estimate = 100  # 예상 출력 토큰 수
                    # Gemini 2.0 Flash 요금 (예시 - 실제 요금은 다를 수 있음)
                    input_cost = (input_tokens / 1000) * 0.000075  # $0.000075 per 1K input tokens
                    output_cost = (output_tokens_estimate / 1000) * 0.0003   # $0.0003 per 1K output tokens
                    total_cost = input_cost + output_cost
                    print(f"   - 추정 API 비용: 약 ${total_cost:.6f} (입력: ${input_cost:.6f}, 출력: ${output_cost:.6f})")
                    
                    # 데이터 전송 시각화
                    print(f"\n📊 데이터 전송 시각화:")
                    total_chars = len(query) + len(context)
                    query_percent = (len(query) / total_chars) * 100
                    context_percent = (len(context) / total_chars) * 100
                    
                    print(f"   질문: {'█' * int(query_percent/2)}{' ' * (50 - int(query_percent/2))} {query_percent:.1f}%")
                    print(f"   컨텍스트: {'█' * int(context_percent/2)}{' ' * (50 - int(context_percent/2))} {context_percent:.1f}%")
                    print(f"   {'─' * 50}")
                    print(f"   총 {total_chars:,}자 ({estimated_tokens:,} 토큰)")
                    
                    # 성능 최적화 제안
                    print(f"\n💡 성능 최적화 제안:")
                    if len(context) > 8000:  # 8K 토큰 이상
                        print(f"   - ⚠️  컨텍스트가 큽니다. 검색 문서 수를 줄이거나 청크 크기를 조정하세요.")
                    if len(retrieved_docs) > 5:
                        print(f"   - 💡 검색된 문서가 많습니다. 더 구체적인 질문을 하거나 검색 파라미터를 조정하세요.")
                    if avg_doc_length > 1000:
                        print(f"   - 💡 문서가 길어서 처리 시간이 오래 걸릴 수 있습니다.")
                    
                    print("-" * 40)
                
                # 실제 질의응답 실행
                print("🧠 AI 응답 생성 중...")
                ai_message = qa_chain({"query": query})

                print("\n" + "=" * 50)
                print("✅ 최종 응답:")
                print(ai_message['result'])
                print("=" * 50)
                
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