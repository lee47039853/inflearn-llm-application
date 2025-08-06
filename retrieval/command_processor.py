"""
명령어 처리 모듈
"""

from .conversation_history import ConversationHistory
from .enhanced_rag import EnhancedRAGSystem


class CommandProcessor:
    """명령어 처리 클래스"""
    
    def __init__(self, conversation_history: ConversationHistory, enhanced_rag: EnhancedRAGSystem):
        self.conversation_history = conversation_history
        self.enhanced_rag = enhanced_rag
    
    def process_command(self, command: str) -> bool:
        """명령어 처리"""
        command = command.lower()
        
        # 히스토리 제어 명령어들
        if command == 'disable_history':
            self.conversation_history.disable_history()
            return True
        elif command == 'enable_history':
            self.conversation_history.enable_history()
            return True
        elif command == 'clear_history':
            self.conversation_history.clear_history()
            return True
        elif command == 'reset_conversation':
            self.conversation_history.reset_conversation()
            return True
        elif command == 'clear_and_disable':
            self.conversation_history.clear_and_disable()
            return True
        elif command == 'remove_last':
            self.conversation_history.remove_last_exchange()
            return True
        elif command.startswith('remove_history:'):
            try:
                index = int(command.split(':', 1)[1].strip()) - 1
                self.conversation_history.remove_exchange_by_index(index)
            except (ValueError, IndexError):
                print("❌ 형식: remove_history:번호 (예: remove_history:1)")
            return True
        elif command == 'history_status':
            status = self.conversation_history.get_history_status()
            print(f"\n📊 대화 히스토리 상태:")
            print(f"  활성화: {'✅' if status['enabled'] else '❌'}")
            print(f"  저장된 대화: {status['count']}개")
            print(f"  최대 저장: {status['max_history']}개")
            print(f"  컨텍스트: {'있음' if status['has_context'] else '없음'}")
            return True
        elif command == 'show_history':
            self.conversation_history.show_history()
            return True
        elif command == 'show_context':
            if self.conversation_history.current_context:
                print(f"\n📝 현재 대화 컨텍스트:")
                print(f"{self.conversation_history.current_context}")
            else:
                print("📝 현재 대화 컨텍스트가 없습니다.")
            return True
        
        # 쿼리 최적화 제어 명령어들
        elif command == 'toggle_optimization':
            self.enhanced_rag.toggle_query_optimization()
            return True
        elif command == 'optimization_status':
            status = self.enhanced_rag.get_optimization_status()
            print(f"\n🔧 쿼리 최적화 상태:")
            print(f"  활성화: {'✅' if status['enabled'] else '❌'}")
            print(f"  체인 생성: {'✅' if status['has_chain'] else '❌'}")
            print(f"  사전 항목: {status['dictionary_count']}개")
            return True
        
        return False 