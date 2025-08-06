"""
대화 히스토리 관리 모듈
"""

from typing import List, Dict, Optional
from datetime import datetime
from .config import Config


class ConversationHistory:
    """대화 히스토리 관리 클래스"""
    
    def __init__(self, max_history: int = Config.MAX_HISTORY):
        self.history = []
        self.max_history = max_history
        self.current_context = ""
        self.history_enabled = True
    
    def add_exchange(self, question: str, answer: str, retrieved_docs: Optional[List] = None):
        """질문-답변 교환 추가"""
        if not self.history_enabled:
            return
        
        exchange = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs if retrieved_docs else [],
            'context_summary': self._extract_context(answer)
        }
        
        self.history.append(exchange)
        
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        self._update_context()
    
    def _extract_context(self, answer: str) -> str:
        """답변에서 핵심 컨텍스트 추출"""
        context_parts = []
        
        for keyword in Config.KEYWORDS:
            if keyword in answer:
                sentences = answer.split('.')
                for sentence in sentences:
                    if keyword in sentence:
                        context_parts.append(sentence.strip())
        
        return '. '.join(context_parts[:3])
    
    def _update_context(self):
        """전체 컨텍스트 업데이트"""
        if not self.history:
            self.current_context = ""
            return
        
        recent_contexts = []
        for exchange in self.history[-3:]:
            if exchange['context_summary']:
                recent_contexts.append(exchange['context_summary'])
        
        self.current_context = " ".join(recent_contexts)
    
    def get_relevant_context(self, new_question: str) -> str:
        """새 질문과 관련된 이전 컨텍스트 찾기"""
        if not self.history_enabled or not self.history:
            return ""
        
        question_keywords = self._extract_keywords(new_question)
        relevant_contexts = []
        
        for exchange in self.history:
            exchange_text = f"{exchange['question']} {exchange['answer']}"
            exchange_keywords = self._extract_keywords(exchange_text)
            
            common_keywords = set(question_keywords) & set(exchange_keywords)
            if common_keywords:
                relevant_contexts.append(exchange['context_summary'])
        
        return " ".join(relevant_contexts[-2:])
    
    def _extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 키워드 추출"""
        found_keywords = []
        for keyword in Config.KEYWORDS:
            if keyword in text:
                found_keywords.append(keyword)
        return found_keywords
    
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
        """대화 완전 초기화"""
        removed_count = len(self.history)
        self.history = []
        self.current_context = ""
        self.history_enabled = True
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