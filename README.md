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

#### 향상된 RAG 시스템 (유사어 Dictionary 및 쿼리 최적화)

```bash
python 3.retrival_with_gemini.py
```

## 📁 프로젝트 구조

- `1.langchain_llm_gemini.py` - Google Gemini 모델 예제
- `1.langchain_llm_ollarma.py` - Ollama Gemma3 모델 예제
- `2.lrag_with_genimi_chroma.py` - RAG 시스템 구현
- `3.retrival_with_gemini.py` - 향상된 RAG 시스템 (유사어 Dictionary, 쿼리 최적화, 대화 히스토리)
- `tax.docx` - 테스트용 문서
- `chroma/` - 벡터 데이터베이스 저장소

## 🔧 기술 스택

- **LangChain**: LLM 애플리케이션 프레임워크
- **Google Gemini**: Google의 LLM 모델
- **Ollama**: 로컬 LLM 실행 환경
- **Chroma**: 벡터 데이터베이스
- **Sentence Transformers**: 한국어 특화 임베딩 모델
- **Python 3.11**: 가상환경

## 🚀 향상된 RAG 시스템 기능

### 3.retrival_with_gemini.py 주요 기능

#### 🔍 유사어 Dictionary 시스템

- **정규화 모드**: 유사어를 대표어로 통일
- **확장 모드**: 대표어를 유사어로 확장
- **통합 모드**: 정규화와 확장을 모두 수행
- **동적 유사어 추가**: `add_synonym:주용어:유사어` 명령어로 실시간 추가

#### 🧠 쿼리 최적화 파이프라인

- **전처리**: 불필요한 공백 제거, 특수문자 정리
- **유사어 처리**: 모드에 따른 쿼리 변형 생성
- **쿼리 변형**: 핵심 키워드 조합 및 질문 형태 변형
- **중복 제거**: 최적화된 쿼리들의 중복 제거

#### 💬 대화 히스토리 관리

- **컨텍스트 유지**: 이전 대화 내용을 다음 질문에 반영
- **히스토리 제어**: 활성화/비활성화, 초기화, 특정 대화 제거
- **관련 컨텍스트 검색**: 새 질문과 관련된 이전 대화 자동 찾기

#### 🎯 고급 검색 기능

- **다중 쿼리 검색**: 최적화된 여러 쿼리로 동시 검색
- **유사도 점수 분석**: 검색 결과의 정확도 시각화
- **중복 제거**: 동일한 내용의 문서 자동 필터링
- **한국어 특화 임베딩**: ko-sroberta-multitask 모델 사용

#### 🔧 디버그 및 모니터링

- **프롬프트 확인 모드**: LLM에 전달되는 프롬프트 상세 분석
- **성능 분석**: 토큰 수, 메모리 사용량, API 비용 추정
- **최적화 제안**: 성능 개선을 위한 실시간 제안

### 📋 사용 가능한 명령어

#### 유사어 관리

- `add_synonym:주용어:유사어` - 새로운 유사어 추가
- `show_synonyms` - 전체 유사어 Dictionary 확인
- `show_synonyms:용어` - 특정 용어의 유사어 확인
- `test_expansion:테스트쿼리` - 쿼리 확장 테스트
- `set_mode:normalize/expand/both` - 확장 모드 설정
- `show_mode` - 현재 확장 모드 확인

#### 대화 히스토리

- `show_history` - 대화 히스토리 표시
- `clear_history` - 히스토리 초기화
- `disable_history` - 히스토리 비활성화
- `enable_history` - 히스토리 활성화
- `remove_last` - 마지막 대화 제거
- `remove_history:번호` - 특정 대화 제거
- `history_status` - 히스토리 상태 확인
- `reset_conversation` - 대화 완전 초기화

## 📝 주의사항

1. Google API 키가 필요합니다 (Gemini 사용 시)
2. Ollama를 사용하려면 로컬에 Ollama가 설치되어 있어야 합니다
3. 가상환경이 활성화된 상태에서 실행하세요
4. 각 파일은 독립적으로 실행 가능합니다
5. 향상된 RAG 시스템은 한국어 특화 임베딩 모델을 사용하여 더 정확한 검색 결과를 제공합니다
