"""
데이터베이스 관리 모듈
"""

import shutil
from pathlib import Path
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from .config import Config


class DatabaseManager:
    """데이터베이스 관리 클래스"""
    
    def __init__(self, chroma_dir: str = Config.CHROMA_DIR):
        self.chroma_dir = Path(chroma_dir)
    
    def check_existing_database(self) -> bool:
        """기존 Chroma 데이터베이스 존재 여부 확인"""
        return self.chroma_dir.exists() and any(self.chroma_dir.iterdir())
    
    def clear_database(self):
        """기존 데이터베이스 삭제"""
        if self.chroma_dir.exists():
            shutil.rmtree(self.chroma_dir)
            print("🗑️  기존 데이터베이스가 삭제되었습니다.")
    
    def create_database(self, embedding, collection_name: str = 'chroma-tax'):
        """Chroma 데이터베이스 생성"""
        return Chroma(
            collection_name=collection_name,
            persist_directory=str(self.chroma_dir),
            embedding_function=embedding
        )
    
    def load_documents(self, document_path: str = Config.DOCUMENT_PATH):
        """문서 로딩 및 분할"""
        print("📄 문서 로딩 중...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
        
        loader = Docx2txtLoader(document_path)
        document_list = loader.load_and_split(text_splitter=text_splitter)
        
        print(f"✅ {len(document_list)}개의 문서 청크로 분할 완료")
        return document_list 