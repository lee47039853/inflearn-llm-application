#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) 시스템 예제
문서 기반 질의응답 시스템 - 유사어 Dictionary 및 쿼리 최적화 파이프라인 포함
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
import json
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime

class SynonymDictionary:
    """유사어 Dictionary 관리 클래스"""
    
    def __init__(self, synonym_file: str = "synonyms.json"):
        self.synonym_file = synonym_file
        self.synonyms = self._load_synonyms()
        self.expansion_mode = "normalize"  # 기본값: 정규화 모드
    
    def _load_synonyms(self) -> Dict[str, List[str]]:
        """유사어 파일 로드"""
        default_synonyms = {
            "소득세": ["소득세", "소득세액", "소득세금", "소득세율", "소득세 계산"],
            "연봉": ["연봉", "연간소득", "연소득", "연간수입", "연수입", "연간급여"],
            "거주자": ["거주자", "직장인", "사람","거주세", "거주민", "거주자세", "거주자세율"],
            "과세표준": ["과세표준", "과세기준", "과세표준액", "과세기준액", "과세표준금액"],
            "종합소득": ["종합소득", "종합소득세", "종합소득세율", "종합소득세액"],
            "세율": ["세율", "세금율", "세율표", "세율기준", "세율계산"],
            "공제": ["공제", "세액공제", "소득공제", "공제액", "공제금액"],
            "신고": ["신고", "세금신고", "소득세신고", "신고서", "신고기간"],
            "납부": ["납부", "세금납부", "납부액", "납부기간", "납부방법"],
            "계산": ["계산", "세금계산", "소득세계산", "계산방법", "계산기"],
            "법인세": ["법인세", "법인세율", "법인세액", "법인세율표"],
            "부가가치세": ["부가가치세", "부가세", "부가가치세율", "부가세율"],
            "양도소득세": ["양도소득세", "양도세", "양도소득세율", "양도세율"],
            "상속세": ["상속세", "상속세율", "상속세액", "상속세율표"],
            "증여세": ["증여세", "증여세율", "증여세액", "증여세율표"]
        }
        
        try:
            if os.path.exists(self.synonym_file):
                with open(self.synonym_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # 기본 유사어 파일 생성
                with open(self.synonym_file, 'w', encoding='utf-8') as f:
                    json.dump(default_synonyms, f, ensure_ascii=False, indent=2)
                return default_synonyms
        except Exception as e:
            print(f"⚠️  유사어 파일 로드 중 오류: {e}")
            return default_synonyms
    
    def expand_query(self, query: str) -> List[str]:
        """쿼리를 유사어로 확장 (모드에 따라 다르게 동작)"""
        expanded_queries = [query]
        expansion_log = []  # 확장 과정 로그
        
        if self.expansion_mode == "normalize":
            # 정규화 모드: 유사어를 대표어로만 정규화
            normalized_query = self.normalize_synonyms_to_main_terms(query)
            if normalized_query != query:
                expanded_queries.append(normalized_query)
                expansion_log.append({
                    'type': 'normalization',
                    'original_query': query,
                    'normalized_query': normalized_query,
                    'description': '유사어를 대표어로 정규화'
                })
        
        elif self.expansion_mode == "expand":
            # 확장 모드: 대표어를 유사어로만 확장
            # 먼저 정규화
            normalized_query = self.normalize_synonyms_to_main_terms(query)
            working_query = normalized_query if normalized_query != query else query
            
            # 단일 용어 확장
            for main_term, synonyms in self.synonyms.items():
                if main_term in working_query:
                    for synonym in synonyms:
                        if synonym != main_term:
                            expanded_query = working_query.replace(main_term, synonym)
                            if expanded_query not in expanded_queries:
                                expanded_queries.append(expanded_query)
                                expansion_log.append({
                                    'type': 'single_expansion',
                                    'original_term': main_term,
                                    'synonym': synonym,
                                    'original_query': working_query,
                                    'expanded_query': expanded_query
                                })
            
            # 다중 용어 조합 확장
            query_terms = []
            for main_term in self.synonyms.keys():
                if main_term in working_query:
                    query_terms.append(main_term)
            
            if len(query_terms) >= 2:
                for i in range(len(query_terms)):
                    for j in range(i+1, len(query_terms)):
                        term1, term2 = query_terms[i], query_terms[j]
                        
                        for synonym1 in self.synonyms[term1]:
                            if synonym1 != term1:
                                for synonym2 in self.synonyms[term2]:
                                    if synonym2 != term2:
                                        expanded_query = working_query.replace(term1, synonym1).replace(term2, synonym2)
                                        if expanded_query not in expanded_queries:
                                            expanded_queries.append(expanded_query)
                                            expansion_log.append({
                                                'type': 'double_expansion',
                                                'original_terms': [term1, term2],
                                                'synonyms': [synonym1, synonym2],
                                                'original_query': working_query,
                                                'expanded_query': expanded_query
                                            })
        
        elif self.expansion_mode == "both":
            # 통합 모드: 정규화 + 확장 모두 수행
            # 1. 정규화
            normalized_query = self.normalize_synonyms_to_main_terms(query)
            if normalized_query != query:
                expanded_queries.append(normalized_query)
                expansion_log.append({
                    'type': 'normalization',
                    'original_query': query,
                    'normalized_query': normalized_query,
                    'description': '유사어를 대표어로 정규화'
                })
            
            # 2. 확장
            working_query = normalized_query if normalized_query != query else query
            
            # 단일 용어 확장
            for main_term, synonyms in self.synonyms.items():
                if main_term in working_query:
                    for synonym in synonyms:
                        if synonym != main_term:
                            expanded_query = working_query.replace(main_term, synonym)
                            if expanded_query not in expanded_queries:
                                expanded_queries.append(expanded_query)
                                expansion_log.append({
                                    'type': 'single_expansion',
                                    'original_term': main_term,
                                    'synonym': synonym,
                                    'original_query': working_query,
                                    'expanded_query': expanded_query
                                })
            
            # 다중 용어 조합 확장
            query_terms = []
            for main_term in self.synonyms.keys():
                if main_term in working_query:
                    query_terms.append(main_term)
            
            if len(query_terms) >= 2:
                for i in range(len(query_terms)):
                    for j in range(i+1, len(query_terms)):
                        term1, term2 = query_terms[i], query_terms[j]
                        
                        for synonym1 in self.synonyms[term1]:
                            if synonym1 != term1:
                                for synonym2 in self.synonyms[term2]:
                                    if synonym2 != term2:
                                        expanded_query = working_query.replace(term1, synonym1).replace(term2, synonym2)
                                        if expanded_query not in expanded_queries:
                                            expanded_queries.append(expanded_query)
                                            expansion_log.append({
                                                'type': 'double_expansion',
                                                'original_terms': [term1, term2],
                                                'synonyms': [synonym1, synonym2],
                                                'original_query': working_query,
                                                'expanded_query': expanded_query
                                            })
        
        # 확장 로그 저장
        self.last_expansion_log = expansion_log
        return expanded_queries
    
    def normalize_synonyms_to_main_terms(self, query: str) -> str:
        """유사어를 대표어로 정규화"""
        normalized_query = query
        
        # 유사어 → 대표어 매핑 생성
        synonym_to_main = {}
        for main_term, synonyms in self.synonyms.items():
            for synonym in synonyms:
                if synonym != main_term:  # 자기 자신은 제외
                    synonym_to_main[synonym] = main_term
        
        # 쿼리에서 유사어를 대표어로 교체
        for synonym, main_term in synonym_to_main.items():
            if synonym in normalized_query:
                normalized_query = normalized_query.replace(synonym, main_term)
        
        return normalized_query
    
    def get_normalization_info(self, query: str) -> Dict:
        """정규화 정보 반환"""
        original_query = query
        normalized_query = self.normalize_synonyms_to_main_terms(query)
        
        # 어떤 유사어가 어떤 대표어로 변경되었는지 추적
        changes = []
        synonym_to_main = {}
        for main_term, synonyms in self.synonyms.items():
            for synonym in synonyms:
                if synonym != main_term:
                    synonym_to_main[synonym] = main_term
        
        for synonym, main_term in synonym_to_main.items():
            if synonym in original_query:
                changes.append({
                    'synonym': synonym,
                    'main_term': main_term,
                    'position': original_query.find(synonym)
                })
        
        return {
            'original_query': original_query,
            'normalized_query': normalized_query,
            'changed': original_query != normalized_query,
            'changes': changes
        }
    
    def set_expansion_mode(self, mode: str):
        """확장 모드 설정"""
        valid_modes = ["normalize", "expand", "both"]
        if mode in valid_modes:
            self.expansion_mode = mode
            print(f"✅ 확장 모드가 '{mode}'로 설정되었습니다.")
        else:
            print(f"❌ 유효하지 않은 모드입니다. 사용 가능한 모드: {valid_modes}")
    
    def get_expansion_mode(self) -> str:
        """현재 확장 모드 반환"""
        return self.expansion_mode
    
    def get_expansion_log(self) -> List[Dict]:
        """마지막 확장 과정 로그 반환"""
        return getattr(self, 'last_expansion_log', [])
    
    def add_synonym(self, main_term: str, synonym: str):
        """새로운 유사어 추가"""
        if main_term not in self.synonyms:
            self.synonyms[main_term] = []
        
        if synonym not in self.synonyms[main_term]:
            self.synonyms[main_term].append(synonym)
            self._save_synonyms()
    
    def _save_synonyms(self):
        """유사어 파일 저장"""
        try:
            with open(self.synonym_file, 'w', encoding='utf-8') as f:
                json.dump(self.synonyms, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  유사어 파일 저장 중 오류: {e}")
    
    def show_synonyms(self, term: str = None):
        """유사어 Dictionary 상태 표시"""
        if term:
            if term in self.synonyms:
                print(f"\n📚 '{term}' 관련 유사어:")
                for i, synonym in enumerate(self.synonyms[term], 1):
                    print(f"  {i}. {synonym}")
            else:
                print(f"❌ '{term}'에 대한 유사어가 없습니다.")
        else:
            print(f"\n📚 전체 유사어 Dictionary 상태:")
            print(f"  총 용어 수: {len(self.synonyms)}")
            print(f"  총 유사어 수: {sum(len(synonyms) for synonyms in self.synonyms.values())}")
            print("\n  주요 용어들:")
            for term, synonyms in list(self.synonyms.items())[:10]:  # 상위 10개만 표시
                print(f"    {term}: {len(synonyms)}개 유사어")
            if len(self.synonyms) > 10:
                print(f"    ... 외 {len(self.synonyms) - 10}개 용어")
    
    def test_expansion(self, query: str):
        """쿼리 확장 테스트"""
        print(f"\n🧪 쿼리 확장 테스트:")
        print(f"원본 쿼리: '{query}'")
        
        # 정규화 정보 표시
        normalization_info = self.get_normalization_info(query)
        if normalization_info['changed']:
            print(f"\n📝 정규화 과정:")
            print(f"  원본: '{normalization_info['original_query']}'")
            print(f"  정규화: '{normalization_info['normalized_query']}'")
            print(f"  변경된 용어들:")
            for change in normalization_info['changes']:
                print(f"    '{change['synonym']}' → '{change['main_term']}'")
        
        expanded_queries = self.expand_query(query)
        expansion_log = self.get_expansion_log()
        
        print(f"\n📊 확장 결과:")
        print(f"  확장된 쿼리 수: {len(expanded_queries)}")
        print("  확장된 쿼리들:")
        for i, expanded_query in enumerate(expanded_queries, 1):
            print(f"    {i}. {expanded_query}")
        
        if expansion_log:
            print(f"\n🔄 확장 과정:")
            for i, log_entry in enumerate(expansion_log, 1):
                print(f"  {i}. {log_entry}")
        
        return expanded_queries

class QueryOptimizer:
    """쿼리 최적화 파이프라인 클래스"""
    
    def __init__(self, synonym_dict: SynonymDictionary):
        self.synonym_dict = synonym_dict
        self.query_history = []
    
    def preprocess_query(self, query: str) -> str:
        """쿼리 전처리"""
        # 1. 불필요한 공백 제거
        query = re.sub(r'\s+', ' ', query.strip())
        
        # 2. 특수문자 정리
        query = re.sub(r'[^\w\s가-힣]', ' ', query)
        
        # 3. 숫자 정규화
        query = re.sub(r'(\d+)천만원', r'\1천만원', query)
        query = re.sub(r'(\d+)만원', r'\1만원', query)
        
        return query
    
    def expand_query_with_synonyms(self, query: str) -> List[str]:
        """유사어를 사용한 쿼리 확장"""
        expanded_queries = self.synonym_dict.expand_query(query)
        
        # 확장 과정 시각화
        expansion_log = self.synonym_dict.get_expansion_log()
        if expansion_log:
            print("\n🔄 유사어 확장 과정:")
            print("=" * 60)
            for i, log_entry in enumerate(expansion_log, 1):
                print(f"  단계 {i} ({log_entry.get('type', 'unknown')}):")
                
                if log_entry.get('type') == 'normalization':
                    print(f"    설명: {log_entry.get('description', '정규화')}")
                    print(f"    변경 전: '{log_entry['original_query']}'")
                    print(f"    변경 후: '{log_entry['normalized_query']}'")
                elif log_entry.get('type') in ['single_replacement', 'single_expansion']:
                    print(f"    원본 용어: '{log_entry['original_term']}'")
                    print(f"    유사어: '{log_entry['synonym']}'")
                    print(f"    변경 전: '{log_entry['original_query']}'")
                    print(f"    변경 후: '{log_entry['expanded_query']}'")
                elif log_entry.get('type') in ['double_replacement', 'double_expansion']:
                    print(f"    원본 용어들: {log_entry['original_terms']}")
                    print(f"    유사어들: {log_entry['synonyms']}")
                    print(f"    변경 전: '{log_entry['original_query']}'")
                    print(f"    변경 후: '{log_entry['expanded_query']}'")
                
                print("-" * 40)
            print("=" * 60)
        
        return expanded_queries
    
    def generate_search_queries(self, original_query: str) -> List[str]:
        """검색용 쿼리 생성"""
        current_mode = self.synonym_dict.get_expansion_mode()
        mode_descriptions = {
            "normalize": "정규화 모드 (유사어 → 대표어)",
            "expand": "확장 모드 (대표어 → 유사어)",
            "both": "통합 모드 (정규화 + 확장)"
        }
        
        print("\n🔧 쿼리 최적화 파이프라인:")
        print(f"현재 모드: {current_mode} - {mode_descriptions.get(current_mode, '알 수 없음')}")
        print("=" * 60)
        
        # 1. 전처리
        processed_query = self.preprocess_query(original_query)
        print(f"1️⃣ 전처리:")
        print(f"   원본: '{original_query}'")
        print(f"   처리 후: '{processed_query}'")
        print("-" * 40)
        
        # 2. 유사어 처리 (모드에 따라 다름)
        expanded_queries = self.expand_query_with_synonyms(processed_query)
        
        if current_mode == "normalize":
            # 정규화 모드: 정규화 정보만 표시
            normalization_info = self.synonym_dict.get_normalization_info(processed_query)
            if normalization_info['changed']:
                print(f"2️⃣ 유사어 정규화:")
                print(f"   원본: '{normalization_info['original_query']}'")
                print(f"   정규화: '{normalization_info['normalized_query']}'")
                print(f"   변경된 용어들:")
                for change in normalization_info['changes']:
                    print(f"     '{change['synonym']}' → '{change['main_term']}'")
                print("-" * 40)
            else:
                print(f"2️⃣ 유사어 정규화:")
                print(f"   정규화할 유사어가 없습니다.")
                print("-" * 40)
        
        elif current_mode == "expand":
            # 확장 모드: 확장 정보만 표시
            print(f"2️⃣ 유사어 확장:")
            print(f"   확장된 쿼리 수: {len(expanded_queries)}")
            for i, query in enumerate(expanded_queries, 1):
                print(f"   {i}. {query}")
            print("-" * 40)
        
        elif current_mode == "both":
            # 통합 모드: 정규화 + 확장 모두 표시
            normalization_info = self.synonym_dict.get_normalization_info(processed_query)
            if normalization_info['changed']:
                print(f"2️⃣ 유사어 정규화:")
                print(f"   원본: '{normalization_info['original_query']}'")
                print(f"   정규화: '{normalization_info['normalized_query']}'")
                print(f"   변경된 용어들:")
                for change in normalization_info['changes']:
                    print(f"     '{change['synonym']}' → '{change['main_term']}'")
                print("-" * 40)
            
            print(f"3️⃣ 유사어 확장:")
            print(f"   확장된 쿼리 수: {len(expanded_queries)}")
            for i, query in enumerate(expanded_queries, 1):
                print(f"   {i}. {query}")
            print("-" * 40)
        
        # 쿼리 변형 생성 (모드에 따라 단계 번호 조정)
        variations = self._generate_query_variations(processed_query)
        step_number = 3 if current_mode in ["normalize", "expand"] else 4
        print(f"{step_number}️⃣ 쿼리 변형 생성:")
        print(f"   변형 쿼리 수: {len(variations)}")
        for i, query in enumerate(variations, 1):
            print(f"   {i}. {query}")
        print("-" * 40)
        
        # 중복 제거 및 정렬 (모드에 따라 단계 번호 조정)
        all_queries = [processed_query] + expanded_queries + variations
        unique_queries = list(dict.fromkeys(all_queries))  # 순서 유지하면서 중복 제거
        
        final_step = 4 if current_mode in ["normalize", "expand"] else 5
        print(f"{final_step}️⃣ 최종 정리:")
        print(f"   총 생성된 쿼리: {len(all_queries)}")
        print(f"   중복 제거 후: {len(unique_queries)}")
        print(f"   최종 선택: {min(5, len(unique_queries))}개")
        print("=" * 60)
        
        return unique_queries[:5]  # 최대 5개 쿼리 반환
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """쿼리 변형 생성"""
        variations = []
        
        # 1. 핵심 키워드 추출
        keywords = self._extract_keywords(query)
        
        # 2. 키워드 조합 생성
        if len(keywords) >= 2:
            # 2개 키워드 조합
            for i in range(len(keywords)):
                for j in range(i+1, len(keywords)):
                    combination = f"{keywords[i]} {keywords[j]}"
                    if combination not in variations:
                        variations.append(combination)
        
        # 3. 질문 형태 변형
        question_patterns = [
            f"{query} 계산 방법",
            f"{query} 세율",
            f"{query} 공제",
            f"{query} 신고"
        ]
        variations.extend(question_patterns)
        
        return variations
    
    def _extract_keywords(self, query: str) -> List[str]:
        """핵심 키워드 추출"""
        # 세금 관련 핵심 키워드
        tax_keywords = [
            "소득세", "연봉", "거주자", "과세표준", "종합소득",
            "세율", "공제", "신고", "납부", "계산",
            "법인세", "부가가치세", "양도소득세", "상속세", "증여세"
        ]
        
        extracted = []
        for keyword in tax_keywords:
            if keyword in query:
                extracted.append(keyword)
        
        return extracted
    
    def log_query(self, original_query: str, optimized_queries: List[str]):
        """쿼리 로깅"""
        self.query_history.append({
            'original': original_query,
            'optimized': optimized_queries,
            'timestamp': str(datetime.now())
        })

class EnhancedRAGSystem:
    """향상된 RAG 시스템 클래스"""
    
    def __init__(self, database, llm, synonym_dict: SynonymDictionary):
        self.database = database
        self.llm = llm
        self.synonym_dict = synonym_dict
        self.query_optimizer = QueryOptimizer(synonym_dict)
    
    def search_with_optimization(self, query: str, top_k: int = 5) -> List[Tuple]:
        """최적화된 검색 수행"""
        # 1. 쿼리 최적화
        optimized_queries = self.query_optimizer.generate_search_queries(query)
        
        # 2. 각 쿼리로 검색 수행
        all_results = []
        for opt_query in optimized_queries:
            try:
                results = self.database.similarity_search_with_score(opt_query, k=top_k)
                all_results.extend(results)
            except Exception as e:
                print(f"⚠️  쿼리 '{opt_query}' 검색 중 오류: {e}")
        
        # 3. 결과 중복 제거 및 정렬
        unique_results = self._deduplicate_results(all_results)
        
        # 4. 상위 결과 반환
        return sorted(unique_results, key=lambda x: x[1])[:top_k]
    
    def _deduplicate_results(self, results: List[Tuple]) -> List[Tuple]:
        """검색 결과 중복 제거"""
        seen_content = set()
        unique_results = []
        
        for doc, score in results:
            content_hash = hash(doc.page_content[:100])  # 내용의 처음 100자로 해시
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append((doc, score))
        
        return unique_results

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
        
        # 유사어 Dictionary 및 향상된 RAG 시스템 초기화
        print("\n📚 유사어 Dictionary 로딩 중...")
        synonym_dict = SynonymDictionary()
        enhanced_rag = EnhancedRAGSystem(database, llm, synonym_dict)
        print("✅ 유사어 Dictionary 로딩 완료")
        
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
            print("   (유사어 추가: add_synonym:주용어:유사어)")
            print("   (유사어 확인: show_synonyms 또는 show_synonyms:용어)")
            print("   (확장 테스트: test_expansion:테스트쿼리)")
            print("   (모드 설정: set_mode:normalize/expand/both)")
            print("   (모드 확인: show_mode)")
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
                
                # 유사어 관리 명령어 처리
                if query.lower().startswith('add_synonym:'):
                    try:
                        parts = query.split(':', 2)
                        if len(parts) == 3:
                            main_term = parts[1].strip()
                            synonym = parts[2].strip()
                            synonym_dict.add_synonym(main_term, synonym)
                            print(f"✅ 유사어 추가 완료: '{main_term}' → '{synonym}'")
                            continue
                        else:
                            print("❌ 형식: add_synonym:주용어:유사어")
                            continue
                    except Exception as e:
                        print(f"❌ 유사어 추가 중 오류: {e}")
                        continue
                
                # 유사어 Dictionary 상태 확인
                if query.lower().startswith('show_synonyms:'):
                    term = query.split(':', 1)[1].strip() if ':' in query else None
                    synonym_dict.show_synonyms(term)
                    continue
                
                if query.lower() == 'show_synonyms':
                    synonym_dict.show_synonyms()
                    continue
                
                # 쿼리 확장 테스트
                if query.lower().startswith('test_expansion:'):
                    test_query = query.split(':', 1)[1].strip()
                    synonym_dict.test_expansion(test_query)
                    continue
                
                # 확장 모드 설정
                if query.lower().startswith('set_mode:'):
                    mode = query.split(':', 1)[1].strip()
                    synonym_dict.set_expansion_mode(mode)
                    continue
                
                # 현재 모드 확인
                if query.lower() == 'show_mode':
                    current_mode = synonym_dict.get_expansion_mode()
                    mode_descriptions = {
                        "normalize": "정규화 모드 (유사어 → 대표어)",
                        "expand": "확장 모드 (대표어 → 유사어)",
                        "both": "통합 모드 (정규화 + 확장)"
                    }
                    print(f"\n📊 현재 확장 모드: {current_mode}")
                    print(f"설명: {mode_descriptions.get(current_mode, '알 수 없음')}")
                    continue
                
                # 대화 히스토리 관련 명령어
                if query.lower() == 'show_history':
                    conversation_history.show_history()
                    continue
                
                if query.lower() == 'clear_history':
                    conversation_history.clear_history()
                    continue
                
                if query.lower() == 'reset_conversation':
                    conversation_history.reset_conversation()
                    continue
                
                if query.lower() == 'clear_and_disable':
                    conversation_history.clear_and_disable()
                    continue
                
                if query.lower() == 'show_context':
                    if conversation_history.current_context:
                        print(f"\n📝 현재 대화 컨텍스트:")
                        print(f"{conversation_history.current_context}")
                    else:
                        print("📝 현재 대화 컨텍스트가 없습니다.")
                    continue
                
                # 히스토리 제어 명령어
                if query.lower() == 'disable_history':
                    conversation_history.disable_history()
                    continue
                
                if query.lower() == 'enable_history':
                    conversation_history.enable_history()
                    continue
                
                if query.lower() == 'remove_last':
                    conversation_history.remove_last_exchange()
                    continue
                
                if query.lower().startswith('remove_history:'):
                    try:
                        index = int(query.split(':', 1)[1].strip()) - 1  # 1-based to 0-based
                        conversation_history.remove_exchange_by_index(index)
                    except (ValueError, IndexError):
                        print("❌ 형식: remove_history:번호 (예: remove_history:1)")
                    continue
                
                if query.lower() == 'history_status':
                    status = conversation_history.get_history_status()
                    print(f"\n📊 대화 히스토리 상태:")
                    print(f"  활성화: {'✅' if status['enabled'] else '❌'}")
                    print(f"  저장된 대화: {status['count']}개")
                    print(f"  최대 저장: {status['max_history']}개")
                    print(f"  컨텍스트: {'있음' if status['has_context'] else '없음'}")
                    continue
                
                # 이전 대화 컨텍스트 확인 및 적용
                relevant_context = conversation_history.get_relevant_context(query)
                if relevant_context:
                    print(f"\n🔄 관련 이전 컨텍스트 발견:")
                    print(f"  {relevant_context}")
                    print("-" * 50)
                
                print(f"\n🤖 입력된 질문: {query}")
                print("-" * 50)
                
                # 향상된 유사도 검색으로 관련 문서 검색 (점수 포함)
                print("🔍 쿼리 최적화 및 검색 수행 중...")
                retrieved_docs_with_scores = enhanced_rag.search_with_optimization(query, top_k=5)
                retrieved_docs = [doc for doc, _ in retrieved_docs_with_scores]
                print(f"📚 검색된 관련 문서 수: {len(retrieved_docs)}")
                
                # 쿼리 최적화 정보 표시
                optimized_queries = enhanced_rag.query_optimizer.generate_search_queries(query)
                if len(optimized_queries) > 1:
                    print(f"🔄 쿼리 최적화: {len(optimized_queries)}개 변형 생성")
                    for i, opt_query in enumerate(optimized_queries[:3], 1):
                        print(f"   {i}. {opt_query}")
                    if len(optimized_queries) > 3:
                        print(f"   ... 외 {len(optimized_queries)-3}개")
                
                # 쿼리 최적화 과정을 로그에 저장
                enhanced_rag.query_optimizer.log_query(query, optimized_queries)
                
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
                
                # 최종 질의 로그 출력
                print("\n📝 최종 질의 로그:")
                print("=" * 60)
                print(f"원본 질문: {query}")
                print(f"최적화된 쿼리 수: {len(optimized_queries)}")
                print(f"검색된 문서 수: {len(retrieved_docs)}")
                print(f"컨텍스트 길이: {sum(len(doc.page_content) for doc in retrieved_docs)}자")
                print(f"평균 문서 길이: {sum(len(doc.page_content) for doc in retrieved_docs) // len(retrieved_docs) if retrieved_docs else 0}자")
                print("-" * 60)
                
                # 최적화된 쿼리들 표시
                print("🔄 최적화된 쿼리들:")
                for i, opt_query in enumerate(optimized_queries, 1):
                    print(f"  {i}. {opt_query}")
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