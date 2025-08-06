"""
Retrieval Package

RAG (Retrieval-Augmented Generation) 시스템을 위한 모듈화된 패키지
문서 기반 질의응답 시스템 - 대화 히스토리 및 LLM 쿼리 개선 포함

Version: 1.0.0
Author: RAG System Team
"""

__version__ = "1.0.0"
__author__ = "RAG System Team"
__description__ = "RAG (Retrieval-Augmented Generation) 시스템을 위한 모듈화된 패키지"

# 주요 클래스들을 패키지 레벨에서 import하여 사용자 편의성 제공
from .config import Config
from .embedding_manager import EmbeddingManager
from .database_manager import DatabaseManager
from .conversation_history import ConversationHistory
from .enhanced_rag import EnhancedRAGSystem
from .user_interface import UserInterface
from .command_processor import CommandProcessor
from .rag_application import RAGApplication

# 패키지에서 직접 사용할 수 있는 주요 클래스들
__all__ = [
    'Config',
    'EmbeddingManager', 
    'DatabaseManager',
    'ConversationHistory',
    'EnhancedRAGSystem',
    'UserInterface',
    'CommandProcessor',
    'RAGApplication'
]

# 패키지 정보
def get_package_info():
    """패키지 정보 반환"""
    return {
        'name': 'retrieval',
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'modules': [
            'config',
            'embedding_manager',
            'database_manager', 
            'conversation_history',
            'enhanced_rag',
            'user_interface',
            'command_processor',
            'rag_application'
        ]
    } 