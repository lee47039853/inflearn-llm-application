# RAG 시스템 리팩토링 완료 - 패키지 구조

## 📦 새로운 패키지 구조

원본의 785줄 단일 파일을 `retrieval` 패키지로 모듈화하여 리팩토링했습니다:

```
inflearn-llm-application/
├── retrieval/                    # 📦 메인 패키지
│   ├── __init__.py              # 패키지 초기화
│   ├── config.py                # 설정 관리
│   ├── embedding_manager.py     # 임베딩 모델 관리
│   ├── database_manager.py      # 데이터베이스 관리
│   ├── conversation_history.py  # 대화 히스토리 관리
│   ├── enhanced_rag.py          # 향상된 RAG 시스템
│   ├── user_interface.py        # 사용자 인터페이스
│   ├── command_processor.py     # 명령어 처리
│   └── rag_application.py       # 메인 애플리케이션
├── 3.retrival_with_gemini_2.py  # 🚀 메인 실행 파일 (26줄)
├── example_usage.py             # 📖 패키지 사용 예제
├── setup.py                     # 📦 패키지 설치 설정
└── README_REFACTORED.md         # 📋 이 파일
```

### 🔧 핵심 모듈들

1. **`retrieval/config.py`** - 설정 관리

   - 세금 관련 키워드 사전
   - 문서 처리 설정 (청크 크기, 오버랩 등)
   - 파일 경로 설정
   - 대화 히스토리 설정

2. **`retrieval/embedding_manager.py`** - 임베딩 모델 관리

   - 한국어 특화 모델 (ko-sroberta-multitask)
   - Google Gemini 임베딩
   - 모델 선택 및 초기화 로직

3. **`retrieval/database_manager.py`** - 데이터베이스 관리

   - Chroma 데이터베이스 생성/관리
   - 문서 로딩 및 분할
   - 기존 DB 확인/삭제 기능

4. **`retrieval/conversation_history.py`** - 대화 히스토리 관리

   - 질문-답변 교환 저장
   - 컨텍스트 추출 및 관리
   - 히스토리 제어 기능 (활성화/비활성화, 삭제 등)

5. **`retrieval/enhanced_rag.py`** - 향상된 RAG 시스템

   - 쿼리 개선 체인
   - 문서 검색 및 유사도 계산
   - RAG 처리 파이프라인

6. **`retrieval/user_interface.py`** - 사용자 인터페이스

   - 다중 라인 입력 처리
   - 결과 표시 및 분석 정보 출력
   - 사용자 선택 처리

7. **`retrieval/command_processor.py`** - 명령어 처리

   - 히스토리 제어 명령어
   - 쿼리 최적화 제어 명령어
   - 명령어 파싱 및 실행

8. **`retrieval/rag_application.py`** - 메인 애플리케이션
   - 모든 컴포넌트 조율
   - 초기화 및 실행 로직
   - 메인 루프 관리

### 🚀 메인 실행 파일

- **`3.retrival_with_gemini_2.py`** - 간소화된 메인 파일 (26줄)

## 📦 패키지 사용법

### 1. 기본 사용법

```python
# 전체 패키지에서 주요 클래스들 import
from retrieval import RAGApplication

# 애플리케이션 실행
app = RAGApplication()
app.run()
```

### 2. 개별 모듈 사용법

```python
# 특정 모듈에서만 import
from retrieval.config import Config
from retrieval.embedding_manager import EmbeddingManager
from retrieval.conversation_history import ConversationHistory

# 설정 사용
print(f"청크 크기: {Config.CHUNK_SIZE}")

# 대화 히스토리 사용
history = ConversationHistory()
history.add_exchange("질문", "답변")
```

### 3. 패키지 정보 확인

```python
from retrieval import get_package_info

info = get_package_info()
print(f"패키지 버전: {info['version']}")
print(f"모듈 목록: {info['modules']}")
```

## 🔄 실행 방법

### 기존과 동일하게 실행:

```bash
python 3.retrival_with_gemini_2.py
```

### 패키지 사용 예제 실행:

```bash
python example_usage.py
```

### 패키지 설치 (개발용):

```bash
pip install -e .
```

## ✨ 리팩토링 개선사항

### 1. **패키지화**

- 모든 모듈이 `retrieval` 패키지 안에 체계적으로 구성
- 상대 import를 사용하여 모듈 간 의존성 명확화
- `__init__.py`를 통한 패키지 레벨 import 지원

### 2. **중복 제거**

- 반복되는 코드 블록들을 함수와 클래스로 분리
- 설정값들을 중앙화된 Config 클래스로 관리
- 공통 로직을 재사용 가능한 모듈로 분리

### 3. **단일 책임 원칙 (SRP)**

- 각 클래스가 하나의 명확한 책임을 가짐
- 데이터베이스 관리, 임베딩 관리, 히스토리 관리 등이 분리됨

### 4. **모듈화**

- 기능별로 독립적인 모듈로 분리
- 각 모듈은 독립적으로 테스트 및 수정 가능
- 의존성이 명확하게 정의됨

### 5. **가독성 향상**

- 코드 구조가 명확해짐
- 각 모듈의 역할이 명확함
- 유지보수가 용이해짐

### 6. **확장성**

- 새로운 기능 추가가 쉬워짐
- 기존 기능 수정 시 영향 범위가 제한적
- 새로운 임베딩 모델이나 데이터베이스 추가 용이

### 7. **재사용성**

- 다른 프로젝트에서 `retrieval` 패키지 설치 후 사용 가능
- 필요한 모듈만 선택적으로 import 가능
- setup.py를 통한 표준 Python 패키지 구조

## 📊 코드 통계

- **원본**: 785줄 (단일 파일)
- **리팩토링 후**: 9개 파일로 분리
  - `retrieval/__init__.py`: 55줄
  - `retrieval/config.py`: 39줄
  - `retrieval/embedding_manager.py`: 31줄
  - `retrieval/database_manager.py`: 50줄
  - `retrieval/conversation_history.py`: 180줄
  - `retrieval/enhanced_rag.py`: 144줄
  - `retrieval/user_interface.py`: 136줄
  - `retrieval/command_processor.py`: 77줄
  - `retrieval/rag_application.py`: 203줄
  - `3.retrival_with_gemini_2.py`: 26줄 (메인)
  - `example_usage.py`: 65줄
  - `setup.py`: 65줄

**총 1,071줄** (약 36% 증가)하지만 **패키지화로 인한 재사용성 및 유지보수성 대폭 향상**

## 🎯 주요 개선점

1. **패키지화**: 표준 Python 패키지 구조로 재사용성 극대화
2. **유지보수성**: 각 기능이 독립적인 모듈로 분리되어 수정이 용이
3. **테스트 가능성**: 각 모듈을 독립적으로 테스트 가능
4. **재사용성**: 다른 프로젝트에서 필요한 모듈만 선택적으로 사용 가능
5. **가독성**: 코드 구조가 명확하고 이해하기 쉬움
6. **확장성**: 새로운 기능 추가 시 기존 코드에 영향 최소화
7. **설치 가능**: pip를 통한 표준 패키지 설치 지원
