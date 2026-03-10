# 🤖 Instana Chatbot

**Streamlit**과 **LangChain**으로 구축된 종합적인 챗봇 애플리케이션입니다. **IBM watsonx (Mistral AI)** 모델을 기반으로 하며, **Milvus** 벡터 데이터베이스를 활용한 **RAG(검색 증강 생성)** 기능을 통해 정교한 응답을 제공합니다.

---

## 🌟 주요 기능

* **대화형 채팅 UI**: 스트리밍 응답을 지원하는 현대적인 Streamlit 기반 인터페이스
* **RAG (Retrieval-Augmented Generation)**: Milvus 벡터 DB를 활용한 문서 검색 및 컨텍스트 인식 응답 ([활용 문서 링크](https://www.ibm.com/docs/en/SSE1JP5_1.0.301/pdf/instana-observability-1.0.301-documentation.pdf))
* **고성능 패키지 관리**: `uv`를 사용한 빠르고 안정적인 Python 의존성 관리

---

## 🚀 Quick Start

### 1. 사전 요구사항 (Prerequisites)

* **uv 설치**: ([공식 문서](https://docs.astral.sh/uv/) 참조)
    ```bash
    # macOS / Linux
    curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

    # Windows (PowerShell)
    powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
    ```
* **프로젝트 클론**:
    ```bash
    git clone [https://github.com/papooo-dev/instana-chatbot.git](https://github.com/papooo-dev/instana-chatbot.git)
    cd instana-chatbot
    ```

### 2. 환경 설정 (Configuration)

1.  **환경 파일 생성**:
    ```bash
    cp .env.example .env
    ```
2.  **`.env` 파일 설정**: 자신의 환경에 맞게 API 키와 URL을 수정합니다.
    ```ini
    # --- IBM watsonx ---
    WATSONX_API_KEY=YOUR_API_KEY
    WATSONX_URL=[https://us-south.ml.cloud.ibm.com](https://us-south.ml.cloud.ibm.com)
    WATSONX_PROJECT_ID=YOUR_PROJECT_GUID
    WATSONX_MODEL_ID=ibm/granite-20b-multilingual

    # --- 앱 설정 ---
    CHAT_TURNS_LIMIT=5
    QR_TEXT=YOUR_QR_CODE_URL

    # --- RAG용 Milvus ---
    MILVUS_URI=http://localhost:19530
    MILVUS_COLLECTION=instana_docs
    ```

### 3. 설치 및 실행 (Setup & Run)

1.  **의존성 설치**:
    ```bash
    uv sync
    ```
2.  **데이터 준비**: RAG 기능을 위해 PDF를 다운로드합니다.
    ```bash
    curl -o data/instana-observability-1.0.301-documentation.pdf \
      "[https://www.ibm.com/docs/en/SSE1JP5_1.0.301/pdf/instana-observability-1.0.301-documentation.pdf](https://www.ibm.com/docs/en/SSE1JP5_1.0.301/pdf/instana-observability-1.0.301-documentation.pdf)"
    ```
3.  **Milvus 서버 실행**: Docker Compose 사용
    ```bash
    docker-compose -f milvus-standalone-docker-compose.yml up -d
    ```
4.  **문서 벡터화 (Ingestion)**:
    ```bash
    uv run utils/ingest_pdf_to_milvus.py
    ```
5.  **애플리케이션 실행**:
    ```bash
    uv run streamlit run app.py
    ```

---

## 📚 프로젝트 구조

```text
instana-chatbot/
├── app.py                # 메인 Streamlit 애플리케이션
├── core/                 # 핵심 로직 모듈
│   ├── llm.py            # LLM 체인 및 스트리밍 설정
│   ├── rag.py            # RAG 프로세스 구현
│   ├── milvus_manager.py  # 벡터 DB 관리
│   ├── embedding.py       # watsonx 임베딩 통합
│   └── prompts.py         # 시스템 프롬프트 관리
├── utils/                # 유틸리티 (데이터 수집 등)
├── data/                 # PDF 등 정적 자산
├── config/               # 설정 파일 (Docker 등)
└── volumes/              # Milvus 데이터 영구 저장소
