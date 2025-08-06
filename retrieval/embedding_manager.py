"""
임베딩 모델 관리 모듈
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


class EmbeddingManager:
    """임베딩 모델 관리 클래스"""
    
    @staticmethod
    def create_embedding(choice: str):
        """임베딩 모델 생성"""
        if choice == '1':
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
            print("🌐 Google Gemini 임베딩 모델 로딩 중...")
            embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            print("✅ Google Gemini 임베딩 모델 로딩 완료")
        
        return embedding 