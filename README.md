# LLM 애플리케이션 프로젝트

LangChain을 사용한 LLM 애플리케이션 개발 프로젝트입니다.

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 활성화
.\Scripts\Activate.ps1

# 의존성 설치
pip install -r requirements.txt
```

### 2. API 키 설정

`.env` 파일을 생성하고 다음 내용을 추가하세요:

```
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. 개별 실행

#### Google Gemini 모델 사용

```bash
python 1.langchain_llm_gemini.py
```

#### Ollama Gemma3 모델 사용

```bash
python 1.langchain_llm_ollarma.py
```

#### RAG 시스템 (문서 기반 질의응답)

```bash
python 2.lrag_with_genimi_chroma.py
```

## 📁 프로젝트 구조

- `1.langchain_llm_gemini.py` - Google Gemini 모델 예제
- `1.langchain_llm_ollarma.py` - Ollama Gemma3 모델 예제
- `2.lrag_with_genimi_chroma.py` - RAG 시스템 구현
- `tax.docx` - 테스트용 문서
- `chroma/` - 벡터 데이터베이스 저장소

## 🔧 기술 스택

- **LangChain**: LLM 애플리케이션 프레임워크
- **Google Gemini**: Google의 LLM 모델
- **Ollama**: 로컬 LLM 실행 환경
- **Chroma**: 벡터 데이터베이스
- **Python 3.11**: 가상환경

## 📝 주의사항

1. Google API 키가 필요합니다 (Gemini 사용 시)
2. Ollama를 사용하려면 로컬에 Ollama가 설치되어 있어야 합니다
3. 가상환경이 활성화된 상태에서 실행하세요
4. 각 파일은 독립적으로 실행 가능합니다
