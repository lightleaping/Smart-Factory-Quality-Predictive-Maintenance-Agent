````python
# JSON 문자열을 Python dict로 바꾸기 위해 사용합니다.
#
# intent classification 단계에서 OpenAI 응답은 JSON 형식의 "문자열"로 돌아옵니다.
# 예:
# '{"intent": "machine_failure_prediction", "confidence": 0.9}'
#
# 이 문자열을 Python dict로 바꿔야
# result["intent"], result["confidence"]처럼 key로 값을 꺼낼 수 있습니다.
import json

# 환경변수를 읽기 위해 사용합니다.
#
# 예:
# .env 파일에 아래처럼 값을 저장해둘 수 있습니다.
#
# OPENAI_API_KEY=...
# OPENAI_MODEL=gpt-4o
#
# os.getenv("OPENAI_MODEL", "gpt-4o")처럼 사용하면
# 환경변수에 OPENAI_MODEL이 있을 때는 그 값을 쓰고,
# 없을 때는 기본값으로 "gpt-4o"를 사용합니다.
import os

# typing 관련 import입니다.
#
# Any:
# - 어떤 타입이든 들어올 수 있다는 뜻입니다.
# - tool_result, evidence처럼 구조가 유동적인 값에 사용합니다.
#
# Literal:
# - 특정 문자열 값만 반환하도록 타입을 제한할 때 사용합니다.
# - route 함수에서 다음 node 이름을 제한하는 용도로 사용합니다.
from typing import Any, Literal

# uuid4는 랜덤 기반 UUID를 생성합니다.
#
# 예:
# "f4d9f1e2-6f77-4b9a-9b2c-1d17e8f4c123"
#
# 이 프로젝트에서는 trace_id와 session_id를 만들 때 사용합니다.
#
# trace_id:
# - Agent 실행 1회를 추적하기 위한 ID
#
# session_id:
# - server-managed 방식에서 하나의 대화방을 구분하기 위한 ID
from uuid import uuid4

# .env 파일에 저장된 환경변수를 현재 Python 실행 환경으로 불러오기 위해 사용합니다.
from dotenv import load_dotenv

# LangGraph workflow를 만들기 위한 구성 요소입니다.
#
# StateGraph:
# - AgentState를 들고 이동하는 workflow graph를 만듭니다.
#
# START:
# - graph의 시작 지점입니다.
#
# END:
# - graph의 종료 지점입니다.
from langgraph.graph import END, START, StateGraph

# OpenAI API를 호출하기 위한 클라이언트 클래스입니다.
from openai import OpenAI

# server-managed chat_history 방식에서 사용하는 저장/조회 함수입니다.
#
# load_chat_history:
# - session_id 기준으로 SQLite에서 이전 대화 이력을 불러옵니다.
#
# save_chat_history:
# - session_id 기준으로 업데이트된 chat_history를 SQLite에 저장합니다.
#
# 이 파일은 src/agent/history_store.py에 따로 작성되어 있어야 합니다.
from src.agent.history_store import load_chat_history, save_chat_history

# LangGraph node들이 공유하는 상태 타입입니다.
from src.agent.state import AgentState


# 현재 프로젝트의 .env 파일을 읽어옵니다.
#
# 이 코드가 있어야 .env 파일에 적어둔 OPENAI_API_KEY, OPENAI_MODEL 등을
# os.getenv() 또는 OpenAI SDK가 사용할 수 있습니다.
load_dotenv()

# OpenAI API를 호출할 client 객체를 만듭니다.
#
# 보통 OPENAI_API_KEY가 환경 변수에 있으면
# OpenAI()가 자동으로 해당 key를 사용합니다.
client = OpenAI()

# 환경변수에서 OPENAI_MODEL 값을 읽어옵니다.
#
# .env에 OPENAI_MODEL이 있으면 그 값을 사용하고,
# 없으면 기본값으로 "gpt-4o"를 사용합니다.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


# Agent가 지원하는 intent 목록입니다.
#
# LLM이 intent를 분류할 때 아래 값 중 하나를 반환하도록 프롬프트에서 지시합니다.
#
# 다만 LLM 출력은 런타임에 들어오는 문자열이므로
# 실제로는 예상 밖의 값이 들어올 수 있습니다.
#
# 그래서 classify_intent() 함수에서
# LLM이 반환한 intent가 SUPPORTED_INTENTS 안에 있는지 다시 검사합니다.
SUPPORTED_INTENTS = {
    # 설비 고장 위험 예측
    "machine_failure_prediction",

    # 고장 유형 또는 원인 분석
    "failure_type_analysis",

    # 모델 성능, F1-score, recall, threshold 질문
    "model_metric_query",

    # 데이터 컬럼, feature, target 질문
    "dataset_schema_query",

    # 지원하지 않는 질문
    "unknown",
}


# intent와 tool_name을 연결하는 mapping입니다.
#
# 이 mapping의 역할:
# - LLM이 분류한 intent를 보고 어떤 Tool을 호출할지 결정합니다.
# - AgentState의 tool_name 필드에는 여기서 선택된 Tool 이름이 저장됩니다.
#
# 예:
# intent = "machine_failure_prediction"
# → tool_name = "predict_machine_failure"
#
# intent = "dataset_schema_query"
# → tool_name = "get_dataset_schema"
#
# 중요한 점:
# - AgentState의 tool_name은 str로 둡니다.
# - 대신 실제 허용 Tool 목록과 연결 규칙은 INTENT_TO_TOOL 또는 Tool registry에서 관리합니다.
#
# 이렇게 하면 Tool이 추가되거나 이름이 바뀌어도
# AgentState 타입을 크게 수정하지 않고 mapping/registry만 수정해서 확장할 수 있습니다.
INTENT_TO_TOOL = {
    # 설비 고장 위험 예측 intent는 고장 예측 Tool로 연결합니다.
    "machine_failure_prediction": "predict_machine_failure",

    # 고장 유형 또는 원인 분석 intent는 고장 유형 분석 Tool로 연결합니다.
    #
    # 주의:
    # - 함수명/Tool 이름은 analyze_failure_type처럼 동사형으로 맞추는 것이 자연스럽습니다.
    # - 기존에 analysis_failure_type처럼 명사형으로 되어 있었다면,
    #   실제 Tool 함수 이름과 일치하도록 하나로 통일해야 합니다.
    "failure_type_analysis": "analyze_failure_type",

    # 모델 성능, threshold, F1-score 같은 질문은 metric 조회 Tool로 연결합니다.
    "model_metric_query": "get_model_metrics",

    # 데이터 컬럼, feature, target 질문은 dataset schema 조회 Tool로 연결합니다.
    "dataset_schema_query": "get_dataset_schema",

    # unknown은 지원하지 않는 질문이므로 실제 Tool을 호출하지 않습니다.
    #
    # "none"은 실제 Tool 이름이 아닙니다.
    # 호출할 Tool이 없다는 것을 나타내기 위한 sentinel 값입니다.
    #
    # sentinel 값이란?
    # - 일반 데이터처럼 보이지만,
    # - 코드 안에서 특별한 의미를 가지도록 약속한 표시용 값입니다.
    #
    # 예:
    # tool_name = "predict_machine_failure"
    # → 실제 고장 예측 Tool을 호출합니다.
    #
    # tool_name = "none"
    # → 호출할 Tool이 없으므로 Tool 호출 단계를 건너뜁니다.
    #
    # 여기서 "none"은 Python의 None 객체가 아니라,
    # workflow 안에서 "Tool 없음"을 표현하기 위해 정한 문자열입니다.
    "unknown": "none",
}


def prepare_state_only_history(state: AgentState) -> AgentState:
    """
    1번 방식: state-only chat_history

    의미:
    - 현재 LangGraph 실행 중 AgentState 안에서만 chat_history를 관리합니다.
    - 요청이 끝나면 서버나 DB에 자동 저장되지 않습니다.

    장점:
    - 가장 단순합니다.
    - graph.invoke() 테스트에 좋습니다.
    - 저장소가 필요 없습니다.

    단점:
    - 다음 요청에서 자동으로 이어지지 않습니다.
    - 프로그램이 꺼지면 대화 이력은 사라집니다.
    - 사실상 테스트용 또는 단일 실행 흐름 확인용입니다.

    첫 질문인지 판단:
    - state에 chat_history가 없거나 빈 리스트이면 첫 질문으로 봅니다.
    """

    # state-only 방식에서는 외부 저장소를 조회하지 않습니다.
    #
    # 현재 state에 chat_history가 있으면 그대로 사용하고,
    # 없으면 빈 리스트로 시작합니다.
    history = state.get("chat_history", [])

    # 현재 질문이 첫 질문인지 표시합니다.
    #
    # chat_history가 비어 있으면 이전 대화 맥락이 없으므로 첫 질문입니다.
    state["is_first_turn"] = len(history) == 0

    # 이후 classify_intent와 generate_answer에서 사용할 수 있도록
    # state에 chat_history를 확실히 넣어둡니다.
    state["chat_history"] = history

    return state


def prepare_client_managed_history(state: AgentState) -> AgentState:
    """
    2번 방식: client-managed chat_history

    의미:
    - 서버는 chat_history를 저장하지 않습니다.
    - 클라이언트가 이전 응답의 chat_history를 보관했다가
      다음 요청에 다시 포함해서 보냅니다.

    장점:
    - DB/Redis 없이도 멀티턴 Q&A를 구현할 수 있습니다.
    - Swagger나 간단한 프론트엔드에서 테스트하기 쉽습니다.
    - 서버가 대화 상태를 들고 있지 않아 구조가 단순합니다.

    단점:
    - 대화가 길어질수록 요청 body가 커집니다.
    - 클라이언트가 history를 잃어버리면 대화도 이어지지 않습니다.
    - 서버 입장에서는 이전 대화를 기억하지 않습니다.

    프로그램이 꺼지면?
    - 서버에는 저장된 history가 없으므로 서버 쪽 기록은 없습니다.
    - 다만 클라이언트가 localStorage, 앱 상태, 파일 등에 history를 보관하고 있다면
      서버 재시작 후에도 다시 보낼 수 있습니다.

    첫 질문인지 판단:
    - 요청 body에 chat_history가 없거나 빈 리스트이면 첫 질문입니다.
    - 이때 빈 리스트로 시작해서 현재 질문/답변을 새로 쌓습니다.
    """

    # client-managed 방식에서는 요청 body에 들어온 chat_history를 사용합니다.
    #
    # 첫 질문이면 클라이언트가 chat_history를 보내지 않거나 []로 보냅니다.
    # 이 경우 빈 리스트로 시작합니다.
    history = state.get("chat_history", [])

    # chat_history가 비어 있으면 현재 세션의 첫 질문입니다.
    state["is_first_turn"] = len(history) == 0

    # 이후 node들이 공통으로 사용할 수 있도록 state에 저장합니다.
    state["chat_history"] = history

    return state


def prepare_server_managed_history(state: AgentState) -> AgentState:
    """
    3번 방식: server-managed chat_history

    의미:
    - 서버가 session_id를 기준으로 chat_history를 저장하고 불러옵니다.
    - 여기서는 SQLite를 사용해 실제 파일 DB에 저장합니다.

    장점:
    - 실제 서비스 구조에 가장 가깝습니다.
    - 클라이언트는 chat_history 전체를 보낼 필요 없이 session_id만 보내면 됩니다.
    - SQLite 파일이 남아 있으면 서버를 껐다 켜도 대화 이력이 유지됩니다.

    단점:
    - 저장소가 필요합니다.
    - session_id 관리가 필요합니다.
    - 대화 리셋 정책을 정해야 합니다.

    첫 질문인지 판단:
    - 요청에 session_id가 없으면 새 대화로 보고 새 session_id를 생성합니다.
    - session_id가 있어도 DB에 저장된 history가 없으면 첫 질문으로 봅니다.
    """

    # session_id가 없으면 첫 질문 또는 새 대화로 보고 새 session_id를 생성합니다.
    #
    # 클라이언트는 이 session_id를 응답에서 받아 저장해두고,
    # 다음 질문부터 같은 session_id를 다시 보내면 됩니다.
    #
    # 예:
    # 첫 요청:
    # {
    #   "question": "이 설비 고장 위험 예측해줘"
    # }
    #
    # 서버 응답:
    # {
    #   "session_id": "새로 생성된 UUID",
    #   "answer": "...",
    #   ...
    # }
    #
    # 다음 요청:
    # {
    #   "session_id": "이전 응답에서 받은 UUID",
    #   "question": "왜 그렇게 판단했어?"
    # }
    if not state.get("session_id"):
        state["session_id"] = str(uuid4())

    session_id = state["session_id"]

    # 서버 저장소에서 기존 대화 이력을 불러옵니다.
    #
    # DB에 해당 session_id가 없으면 load_chat_history()가 빈 리스트를 반환합니다.
    # 즉, 이 경우가 해당 session_id의 첫 질문입니다.
    #
    # session_id는 있지만 DB에 해당 session_id가 없는 경우도 첫 질문으로 봅니다.
    #
    # DB에 row는 있지만 chat_history가 []인 경우도
    # 이전 대화 맥락이 없으므로 첫 질문처럼 처리합니다.
    history = load_chat_history(session_id)

    state["chat_history"] = history
    state["is_first_turn"] = len(history) == 0

    return state


def save_server_managed_history(state: AgentState) -> AgentState:
    """
    server-managed 방식에서 현재 chat_history를 저장소에 저장합니다.

    state-only 방식과 client-managed 방식에서는 이 함수가 필요 없습니다.
    server-managed 방식에서만 graph 마지막 단계에 연결하면 됩니다.
    """

    # server-managed 방식에서는 session_id가 반드시 필요합니다.
    #
    # session_id가 있어야 "어떤 대화방의 기록을 저장할지" 알 수 있습니다.
    session_id = state.get("session_id")

    if not session_id:
        # session_id가 없으면 저장할 key가 없으므로 DB 저장을 하지 않습니다.
        #
        # 대신 errors에 기록해 디버깅할 수 있게 합니다.
        state.setdefault("errors", []).append("missing_session_id")
        return state

    # 현재 state의 chat_history를 SQLite에 저장합니다.
    #
    # save_chat_history() 내부에서는:
    # - chat_history list를 JSON 문자열로 변환하고,
    # - session_id가 없으면 INSERT,
    # - session_id가 이미 있으면 UPDATE합니다.
    save_chat_history(
        session_id=session_id,
        chat_history=state.get("chat_history", []),
    )

    return state


def classify_intent(
    question: str,
    chat_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """
    LLM 기반 intent classification을 수행합니다.

    기존 single-turn 구조에서는 현재 question만 보고 intent를 분류했습니다.

    하지만 채팅 Q&A에서는 현재 질문만으로 intent를 알기 어려운 경우가 있습니다.

    예:
    - "왜 그렇게 판단했어?"
    - "그럼 조치는?"
    - "아까 말한 threshold가 뭐야?"

    이런 질문은 이전 대화 맥락을 봐야 어떤 intent인지 판단할 수 있습니다.

    따라서 현재 질문(question)과 이전 대화 이력(chat_history)을 함께
    intent classification prompt에 전달합니다.
    """

    # 질문이 비어 있으면 LLM을 호출할 필요가 없습니다.
    #
    # 이 경우 unknown intent로 처리하고 confidence는 0.0으로 둡니다.
    if not question or not question.strip():
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "reason": "질문이 비어 있습니다.",
        }

    # chat_history가 None이면 빈 리스트로 처리합니다.
    #
    # "chat_history를 입력값으로 받는데 왜 빈 리스트 처리가 필요한가?"
    #
    # 이유:
    # - 첫 질문에는 이전 대화가 없을 수 있습니다.
    # - 테스트 코드나 graph.invoke({"question": "..."})에서는 question만 넣고 실행할 수 있습니다.
    # - FastAPI 요청에서 chat_history 필드가 생략될 수도 있습니다.
    # - 아직 프론트엔드나 DB/Redis 대화 저장소가 붙지 않은 상태일 수 있습니다.
    #
    # 즉, chat_history가 없어도 single-turn 질문으로 정상 실행되도록
    # 기본값을 빈 리스트로 맞춰줍니다.
    #
    # 주의:
    # 함수 인자 기본값을 chat_history=[]로 직접 두지 않습니다.
    #
    # Python에서 함수의 기본값은 함수를 호출할 때마다 새로 만들어지는 것이 아니라,
    # 함수가 정의될 때 한 번만 만들어집니다.
    #
    # 따라서 리스트처럼 수정 가능한 mutable 객체를 기본값으로 두면
    # 여러 함수 호출이 같은 리스트를 공유할 수 있습니다.
    #
    # 예:
    # def func(history=[]):
    #     history.append("new message")
    #     return history
    #
    # 위처럼 작성하면 func()를 여러 번 호출할 때마다
    # 매번 빈 리스트에서 시작하는 것이 아니라,
    # 이전 호출에서 append된 값이 남아 있는 같은 리스트를 계속 사용하게 됩니다.
    #
    # 채팅 이력에서는 이런 문제가 특히 위험합니다.
    # 서로 다른 요청의 chat_history가 섞일 수 있기 때문입니다.
    #
    # 그래서 기본값은 None으로 두고,
    # 함수 내부에서 None인지 확인한 뒤 새 빈 리스트를 만들어 사용합니다.
    if chat_history is None:
        chat_history = []

    # 이 프롬프트는 사용자 질문을 정해진 intent 중 하나로 분류하기 위한 프롬프트입니다.
    #
    # LLM에게 자유롭게 답변을 생성하게 하는 것이 아니라,
    # predictive maintenance AI Agent의 intent classifier 역할만 수행하도록 제한합니다.
    #
    # 프롬프트 안에는 다음 요소를 넣었습니다.
    #
    # 1. LLM의 역할: intent classifier
    # 2. 지원하는 intent 목록
    # 3. 각 intent의 의미
    # 4. intent별 예시 질문
    # 5. 이전 대화 이력
    # 6. 현재 사용자 질문
    # 7. JSON 출력 형식
    #
    # 이렇게 작성한 이유:
    # - LLM의 응답을 다음 LangGraph node에서 안정적으로 파싱하기 위해서입니다.
    # - JSON으로 받으면 result["intent"]처럼 명확하게 값을 꺼낼 수 있습니다.
    #
    # 즉, 이 프롬프트의 목적은 답변 생성이 아니라
    # "질문을 어떤 Tool로 보낼지 결정하기 위한 의도 분류"입니다.
    prompt = f"""
You are an intent classifier for a predictive maintenance AI agent.

Classify the current user question into exactly one of the following intents.

Use the previous chat history only to understand references in the current question.
For example, if the current question says "왜 그렇게 판단했어?",
use the previous chat history to understand what "그렇게" refers to.

Do not answer the question.
Only classify the intent.

Supported intents:

1. machine_failure_prediction
- The user wants to predict whether a machine is likely to fail.
- Examples: "이 설비 고장 위험 예측해줘", "고장 가능성 알려줘"

2. failure_type_analysis
- The user wants to know the likely failure type or cause.
- Examples: "어떤 유형의 고장이야?", "고장 원인 후보 알려줘"

3. model_metric_query
- The user asks about model performance, metrics, threshold, recall, precision, F1-score, ROC-AUC.
- Examples: "모델 성능은 어때?", "F1-score 알려줘", "threshold 기준은 뭐야?"

4. dataset_schema_query
- The user asks about dataset columns, features, target, or meaning of fields.
- Examples: "이 데이터 컬럼 설명해줘", "Target은 뭐야?"

5. unknown
- The question is outside the supported scope.
- The current question cannot be resolved even with chat history.

[Previous Chat History]
{chat_history}

[Current User Question]
{question}

Return only valid JSON.
Do not include markdown.
Do not include explanation outside JSON.

JSON schema:
{{
  "intent": "one of the supported intents",
  "confidence": 0.0,
  "reason": "short reason in Korean"
}}
"""

    try:
        # OpenAI API를 호출해 사용자 질문의 intent를 분류합니다.
        #
        # 이 구간에서 오류가 발생할 수 있는 이유:
        # - OPENAI_API_KEY가 없거나 잘못되었을 수 있습니다.
        # - OpenAI API 호출이 실패할 수 있습니다.
        # - 모델 응답이 JSON 형식이 아닐 수 있습니다.
        # - confidence 값을 float으로 변환하지 못할 수 있습니다.
        #
        # 따라서 OpenAI API 호출, output_text 추출, JSON 파싱,
        # intent/confidence/reason 추출 과정을 try 안에서 처리합니다.
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,

            # 출력을 최대한 일관되게 만들기 위한 설정입니다.
            #
            # temperature가 높으면 모델 응답이 더 다양해질 수 있습니다.
            # intent classification에서는 일관성이 중요하므로 0으로 둡니다.
            temperature=0,
        )

        # response.output_text는 모델이 생성한 텍스트 출력만
        # 하나의 문자열로 꺼내기 위한 편의 속성입니다.
        text = response.output_text.strip()

        # 혹시 모델이 ```json ... ``` 형태로 반환했을 때를 대비한 간단한 정리입니다.
        #
        # 프롬프트에서 markdown을 쓰지 말라고 했더라도
        # LLM이 코드블록 형태로 답하는 경우가 있을 수 있습니다.
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        # JSON 문자열을 Python dict로 바꿉니다.
        #
        # response.output_text는 모델이 생성한 텍스트를 문자열로 반환합니다.
        # 이 프로젝트의 intent classification 프롬프트는
        # 모델에게 JSON 형식으로만 답하라고 지시합니다.
        #
        # 하지만 JSON처럼 보이는 응답도 처음에는 Python dict가 아니라 문자열입니다.
        #
        # 예:
        # text = '{"intent": "machine_failure_prediction", "confidence": 0.92}'
        #
        # 따라서 json.loads(text)를 사용해 JSON 문자열을 Python dict로 변환합니다.
        # 변환 후에는 result["intent"], result["confidence"]처럼 key로 값을 꺼낼 수 있습니다.
        result = json.loads(text)

        # dictionary에서 key로 값을 얻습니다.
        #
        # result.get("intent", "unknown")
        # - intent key가 있으면 해당 값을 가져옵니다.
        # - 없으면 기본값으로 "unknown"을 사용합니다.
        intent = result.get("intent", "unknown")

        # confidence는 숫자 연산이나 비교에 사용할 수 있도록 float으로 변환합니다.
        confidence = float(result.get("confidence", 0.0))

        # reason은 intent 분류 이유입니다.
        reason = result.get("reason", "")

    except Exception as error:
        # LLM intent classification 과정에서 오류가 발생해도
        # Agent workflow 전체가 중단되지 않도록 unknown intent를 반환합니다.
        #
        # 즉, 오류가 나면 프로그램을 멈추는 대신
        # "분류 실패" 상태로 안전하게 다음 단계로 넘깁니다.
        #
        # error 메시지는 반환값에 포함해
        # classify_intent_node에서 state["errors"]에 기록할 수 있게 합니다.
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "reason": "LLM intent classification failed.",
            "error": str(error),
        }

    # LLM이 반환한 intent가 프로젝트에서 지원하는 intent인지 확인합니다.
    #
    # LLM에게 정해진 intent만 반환하라고 해도,
    # 실제 실행에서는 오타나 예상 밖 문자열이 나올 수 있습니다.
    #
    # 예:
    # - "machine_failure"
    # - "failure_prediction"
    # - "maintenance_query"
    #
    # 이런 값은 workflow에서 처리할 수 없으므로 unknown으로 바꿉니다.
    if intent not in SUPPORTED_INTENTS:
        return {
            "intent": "unknown",
            "confidence": 0.0,
            "reason": f"지원하지 않는 intent가 반환되었습니다: {intent}",
        }

    return {
        "intent": intent,
        "confidence": confidence,
        "reason": reason,
    }


def validate_question(state: AgentState) -> AgentState:
    """
    사용자 질문이 비어 있는지 확인합니다.

    이 node는 workflow의 앞쪽에서 실행됩니다.

    역할:
    - trace_id가 없으면 새로 생성합니다.
    - question이 비어 있으면 intent를 unknown으로 설정합니다.
    - 오류 내용을 errors에 기록합니다.
    """

    # trace_id는 Agent 실행 1회에 부여하는 고유 추적 ID입니다.
    #
    # uuid4()는 랜덤 기반 UUID를 생성합니다.
    #
    # 예:
    # "f4d9f1e2-6f77-4b9a-9b2c-1d17e8f4c123"
    #
    # 왜 필요한가?
    # - 한 번의 사용자 질문이 여러 node를 거쳐 처리됩니다.
    # - validate_question → classify_intent → call_tool → generate_answer
    #   같은 여러 단계의 로그를 하나의 실행 흐름으로 묶어야 합니다.
    # - trace_id가 있으면 나중에 JSONL 로그, API 응답, 에러 기록을
    #   같은 요청 단위로 추적할 수 있습니다.
    #
    # state에 trace_id가 이미 있으면 기존 값을 유지하고,
    # 없으면 새 UUID를 생성합니다.
    #
    # str(uuid4())를 사용하는 이유:
    # - uuid4() 자체는 UUID 객체입니다.
    # - API 응답이나 JSON 로그에 저장하려면 문자열 형태가 다루기 쉽습니다.
    state.setdefault("trace_id", str(uuid4()))

    question = state.get("question", "")

    if not question or not question.strip():
        state["intent"] = "unknown"
        state["confidence"] = 0.0
        state["intent_reason"] = "질문이 비어 있습니다."

        # empty_question 오류를 errors 목록에 기록합니다.
        #
        # 주의:
        # state["errors"] = ["empty_question"] 후에 다시 append하면
        # 같은 오류가 중복 저장될 수 있습니다.
        #
        # 따라서 setdefault + append만 사용합니다.
        #
        # state.setdefault("errors", [])
        # - errors key가 이미 있으면 기존 list를 사용합니다.
        # - errors key가 없으면 빈 list를 새로 만들고 state["errors"]에 넣습니다.
        #
        # .append("empty_question")
        # - errors list에 이번 오류를 추가합니다.
        state.setdefault("errors", []).append("empty_question")

    return state


def classify_intent_node(state: AgentState) -> AgentState:
    """
    LLM 기반 intent classification 결과를 AgentState에 저장합니다.

    채팅 Q&A 확장:
    - 현재 question뿐 아니라 chat_history도 intent 분류에 사용합니다.
    - 후속 질문의 생략된 맥락을 이전 대화에서 보완하기 위함입니다.
    """

    question = state["question"]

    # 이전 대화 이력을 가져옵니다.
    #
    # 예:
    # [
    #   {"role": "user", "content": "이 설비 고장 위험 예측해줘"},
    #   {"role": "assistant", "content": "고장 가능성이 높습니다."},
    #   {"role": "user", "content": "왜 그렇게 판단했어?"}
    # ]
    #
    # 현재 질문이 "왜 그렇게 판단했어?"처럼 단독으로는 intent를 알기 어려운 경우,
    # chat_history를 함께 전달하면 LLM이 이전 대화 맥락을 참고해
    # 더 적절한 intent로 분류할 수 있습니다.
    chat_history = state.get("chat_history", [])

    result = classify_intent(
        question=question,
        chat_history=chat_history,
    )

    # classify_intent()가 반환한 결과를 AgentState에 저장합니다.
    state["intent"] = result["intent"]
    state["confidence"] = result["confidence"]
    state["intent_reason"] = result["reason"]

    if "error" in result:
        # classify_intent()에서 에러 정보가 반환된 경우,
        # AgentState의 errors 목록에 해당 에러 메시지를 추가합니다.
        #
        # 이렇게 하면 Agent 실행 중 발생한 여러 오류를
        # 하나의 errors list에 누적해서 추적할 수 있습니다.
        state.setdefault("errors", []).append(result["error"])

    return state


def prepare_tool_args(state: AgentState) -> AgentState:
    """
    intent에 따라 호출할 Tool 이름과 기본 입력값을 준비합니다.
    """

    intent = state.get("intent", "unknown")

    # intent를 기준으로 호출할 Tool 이름을 선택합니다.
    #
    # 여기서 INTENT_TO_TOOL은 "허용된 intent → Tool 이름" mapping 역할을 합니다.
    #
    # INTENT_TO_TOOL.get(intent, "none")을 사용하는 이유:
    # - intent가 mapping 안에 있으면 해당 Tool 이름을 가져옵니다.
    # - intent가 예상 밖의 값이면 "none"을 반환합니다.
    #
    # 즉, 예상하지 못한 intent가 들어와도 KeyError로 workflow가 중단되지 않고,
    # Tool 호출이 없는 상태로 안전하게 이어지도록 합니다.
    tool_name = INTENT_TO_TOOL.get(intent, "none")

    # 선택된 Tool 이름을 state에 저장합니다.
    #
    # 이 값은 다음 call_tool 단계에서 어떤 Tool을 호출할지 판단하는 데 사용됩니다.
    state["tool_name"] = tool_name

    # Tool에 전달할 입력값을 준비합니다.
    #
    # 지금은 mock Tool을 사용하지만,
    # 이후 실제 Tool 함수나 MCP Tool 호출로 바뀌어도
    # question, intent, dataset 같은 공통 입력값을 여기서 구성할 수 있습니다.
    state["tool_args"] = {
        "question": state["question"],
        "intent": intent,
        "dataset": "ai4i_2020",
    }

    return state


def call_tool_mock(state: AgentState) -> AgentState:
    """
    실제 MCP Tool과 모델이 붙기 전까지 사용하는 mock Tool입니다.

    이 함수의 목적:
    - 실제 모델 inference나 MCP Tool 서버가 붙기 전에도
      LangGraph workflow가 정상적으로 흐르는지 검증하기 위함입니다.
    - 이후 실제 구현에서는 call_tool_mock 대신
      Tool registry 기반 call_tool 또는 MCP Tool 호출 함수로 교체할 수 있습니다.

    면접 설명:
    - 실제 모델과 Tool을 붙이기 전 LangGraph workflow가 정상적으로 흐르는지 검증하기 위해
      mock Tool을 먼저 구성했습니다.
    - 이후 mock 부분을 실제 모델 inference 함수나 MCP Tool 호출 함수로 교체할 수 있도록
      단계를 분리했습니다.
    """

    intent = state.get("intent", "unknown")

    if intent == "machine_failure_prediction":
        state["tool_result"] = {
            "prediction": 1,
            "probability": 0.82,
            "threshold": 0.5,
            "evidence": [
                {
                    "feature": "Tool wear [min]",
                    "value": 220,
                    "interpretation": "공구 마모 시간이 높은 편입니다.",
                }
            ],
        }

    elif intent == "failure_type_analysis":
        state["tool_result"] = {
            "failure_type": "Tool Wear Failure",
            "evidence": [
                {
                    "feature": "Tool wear [min]",
                    "interpretation": "공구 마모 관련 고장 가능성을 우선 확인합니다.",
                }
            ],
        }

    elif intent == "model_metric_query":
        state["tool_result"] = {
            "model_name": "baseline_model",
            "roc_auc": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "message": "아직 모델 학습 전이므로 metric은 준비되지 않았습니다.",
        }

    elif intent == "dataset_schema_query":
        state["tool_result"] = {
            "dataset": "AI4I 2020 Predictive Maintenance Dataset",
            "target": "Target",
            "features": [
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
            ],
            "excluded_columns": ["UDI", "Product ID"],
        }

    else:
        state["tool_result"] = {
            "message": "지원하지 않는 질문입니다.",
        }

    return state


def evaluate_evidence(state: AgentState) -> AgentState:
    """
    Tool 결과를 AgentState의 prediction, probability, risk_level, evidence로 정리합니다.

    Tool 실행 결과를 AgentState의 표준 응답 필드로 정리하는 node입니다.

    이 단계가 필요한 이유:
    - call_tool node의 결과는 tool_result 안에 들어 있습니다.
    - 하지만 최종 답변 생성 node에서는 tool_result 내부 구조를 매번 직접 읽기보다,
      prediction, probability, risk_level, evidence처럼 공통 필드로 정리된 값을 사용하는 것이 좋습니다.
    - 즉, evaluate_evidence는 Tool의 raw result를 Agent가 설명 가능한 형태로 정리하는 중간 정리 단계입니다.
    """

    # 이전 단계인 call_tool node에서 저장한 Tool 실행 결과를 가져옵니다.
    #
    # 값이 없을 경우를 대비해 기본값으로 빈 dict를 사용합니다.
    tool_result = state.get("tool_result", {})

    # prediction은 모델 또는 Tool이 판단한 최종 예측 결과입니다.
    #
    # 예:
    # - 0 = 정상
    # - 1 = 실패/고장 위험
    # - "normal"
    # - "failure"
    #
    # 왜 state에 따로 저장하는가?
    # - 최종 답변 생성 단계에서 사용자가 바로 이해할 수 있는 핵심 결과이기 때문입니다.
    # - tool_result 안에 묻어두지 않고 state["prediction"]으로 올려두면,
    #   이후 generate_answer node가 결과를 쉽게 사용할 수 있습니다.
    if "prediction" in tool_result:
        state["prediction"] = tool_result["prediction"]

    # probability는 모델이 예측한 위험 가능성 또는 고장 가능성 점수입니다.
    #
    # 보통 0.0 ~ 1.0 사이의 값으로 사용합니다.
    #
    # 예:
    # 0.82 → 고장 가능성이 높음
    # 0.55 → 중간 수준의 위험
    # 0.21 → 낮은 위험
    if "probability" in tool_result:
        try:
            # Tool 결과가 문자열로 들어올 수도 있으므로 float으로 변환합니다.
            #
            # 예:
            # "0.82" → 0.82
            probability = float(tool_result["probability"])

            # probability 원본 값은 그대로 state에 저장합니다.
            #
            # 이 값은 모델의 수치 결과를 그대로 확인하기 위한 필드입니다.
            state["probability"] = probability

            # risk_score는 probability를 사람이 보기 쉬운 0~100 점수로 바꾼 값입니다.
            #
            # 예:
            # 0.82 → 82.0
            #
            # 왜 만드는가?
            # - probability보다 risk_score가 사용자 화면이나 보고서에서 직관적입니다.
            # - "고장 가능성 0.82"보다 "위험 점수 82점"이 이해하기 쉽습니다.
            state["risk_score"] = probability * 100

            # risk_level은 probability를 사람이 이해하기 쉬운 등급으로 바꾼 값입니다.
            #
            # 왜 필요한가?
            # - 실제 서비스에서는 사용자가 0.63 같은 숫자만 보고 판단하기 어렵습니다.
            # - HIGH / MEDIUM / LOW처럼 등급화하면 운영자가 빠르게 판단할 수 있습니다.
            #
            # 기준:
            # - 0.7 이상: HIGH
            #   → 고장 가능성이 높다고 보고 우선 점검 대상
            #
            # - 0.4 이상 0.7 미만: MEDIUM
            #   → 주의가 필요하지만 즉시 고장이라고 단정하기는 어려움
            #
            # - 0.4 미만: LOW
            #   → 현재 위험도가 낮은 상태
            #
            # 이 기준은 초기 프로젝트용 heuristic 기준입니다.
            # 실제 서비스에서는 검증 데이터, 비용, recall/precision 목표에 따라 조정해야 합니다.
            if probability >= 0.7:
                state["risk_level"] = "HIGH"

            elif probability >= 0.4:
                state["risk_level"] = "MEDIUM"

            else:
                state["risk_level"] = "LOW"

        except (TypeError, ValueError) as error:
            # probability가 None이거나 숫자로 바꿀 수 없는 값이면
            # float 변환 과정에서 오류가 발생합니다.
            #
            # 예:
            # - probability = None
            # - probability = "high"
            # - probability = "unknown"
            #
            # 이런 값 때문에 전체 Agent workflow가 중단되지 않도록
            # 오류 메시지만 state["errors"]에 기록하고 다음 단계로 넘어갑니다.
            state.setdefault("errors", []).append(
                f"invalid_probability: {error}"
            )

    # threshold는 모델이 normal/anomaly 또는 failure/no_failure를
    # 나누는 기준값입니다.
    if "threshold" in tool_result:
        try:
            # 비교와 설명에 사용할 수 있도록 float으로 변환해 state에 저장합니다.
            state["threshold"] = float(tool_result["threshold"])

        except (TypeError, ValueError) as error:
            # threshold가 없거나 숫자로 변환할 수 없는 값이면 오류가 발생합니다.
            #
            # threshold 변환 실패가 전체 workflow 중단으로 이어지지 않도록
            # 오류만 state["errors"]에 기록합니다.
            state.setdefault("errors", []).append(
                f"invalid_threshold: {error}"
            )

    # prediction과 risk_level을 분리해서 계산하는 이유:
    #
    # prediction은 threshold를 기준으로 normal/anomaly를 나누는 최종 판정입니다.
    # 예를 들어 probability가 threshold 이상이면 anomaly,
    # threshold 미만이면 normal로 판단합니다.
    #
    # 하지만 threshold 판정은 이진 결과만 알려줍니다.
    # 즉, normal인지 anomaly인지는 알 수 있지만,
    # 기준선에서 얼마나 가까운지 또는 얼마나 강하게 위험한지는 알기 어렵습니다.
    #
    # 그래서 probability를 함께 사용합니다.
    # probability는 모델이 계산한 위험 가능성의 크기입니다.
    #
    # 예:
    # - probability 0.49, threshold 0.5 → normal이지만 기준선에 가까움
    # - probability 0.51, threshold 0.5 → anomaly지만 기준선을 살짝 넘음
    # - probability 0.92, threshold 0.5 → anomaly이며 위험 강도가 높음
    #
    # 따라서 threshold는 "판정 기준",
    # probability는 "위험 가능성의 강도",
    # risk_level은 "사용자에게 보여주기 위한 위험 등급"으로 분리합니다.
    #
    # 다만 AutoEncoder처럼 reconstruction error > threshold로 이상을 판단하는 경우에는
    # probability라는 이름보다 anomaly_score나 reconstruction_error가 더 정확할 수 있고,
    # probability는 보통 분류 모델이 출력한 확률값일 때 쓰는 이름입니다.

    # evidence는 최종 답변의 근거가 되는 정보입니다.
    #
    # 왜 필요한가?
    # - Agent가 단순히 "위험합니다"라고 말하는 것이 아니라,
    #   어떤 입력값, 어떤 모델 결과, 어떤 기준 때문에 그렇게 판단했는지 설명하기 위해 필요합니다.
    #
    # evidence가 없을 수도 있으므로 기본값은 빈 리스트로 둡니다.
    state["evidence"] = tool_result.get("evidence", [])

    return state


def build_fallback_answer(state: AgentState) -> str:
    """
    LLM 답변 생성 또는 검증에 실패했을 때 사용하는 deterministic fallback 답변입니다.

    deterministic이라는 뜻:
    - LLM을 다시 호출하지 않고,
    - 현재 AgentState에 저장된 값만 사용해 답변을 만든다는 뜻입니다.

    fallback이 필요한 이유:
    - LLM 답변 생성이 실패할 수 있습니다.
    - LLM이 계산된 값과 다른 내용을 말할 수 있습니다.
    - LLM이 제공되지 않은 evidence를 만들어낼 수 있습니다.
    - 이때도 사용자에게 최소한의 분석 결과는 제공해야 합니다.
    """

    intent = state.get("intent", "unknown")
    tool_name = state.get("tool_name", "none")
    prediction = state.get("prediction")
    probability = state.get("probability")
    risk_score = state.get("risk_score")
    risk_level = state.get("risk_level")
    threshold = state.get("threshold")
    evidence = state.get("evidence", [])

    return (
        "분석 결과를 안전 모드로 반환합니다. "
        f"현재 intent는 {intent}, 호출된 tool은 {tool_name}입니다. "
        f"prediction={prediction}, probability={probability}, "
        f"risk_score={risk_score}, risk_level={risk_level}, "
        f"threshold={threshold}입니다. "
        f"제공된 판단 근거는 {evidence}입니다."
    )


def generate_answer(state: AgentState) -> AgentState:
    """
    evaluate_evidence에서 정리한 결과를 바탕으로
    OpenAI API를 사용해 사용자용 최종 답변을 생성합니다.

    중요한 점:
    - LLM은 prediction, probability, threshold, risk_level을 다시 계산하지 않습니다.
    - LLM은 이미 계산된 값을 바탕으로 설명 문장만 생성합니다.

    채팅 Q&A 확장:
    - 현재 질문뿐 아니라 chat_history를 함께 사용합니다.
    - 단, chat_history는 맥락 이해용입니다.
    - prediction, probability, threshold, risk_level은 현재 state 값을 우선합니다.

    추가 안전장치:
    - 생성된 답변은 바로 최종 반환하지 않고,
      validate_answer node에서 state 값과 일치하는지 한 번 더 검증합니다.
    """

    intent = state.get("intent", "unknown")
    tool_name = state.get("tool_name", "none")
    tool_result = state.get("tool_result", {})

    question = state.get("question", "")
    chat_history = state.get("chat_history", [])

    # evaluate_evidence에서 정리한 핵심 결과를 가져옵니다.
    prediction = state.get("prediction")
    probability = state.get("probability")
    risk_score = state.get("risk_score")
    risk_level = state.get("risk_level")
    threshold = state.get("threshold")
    evidence = state.get("evidence", [])

    # 이 프롬프트는 최종 답변 생성을 위한 프롬프트입니다.
    #
    # intent classification prompt와 달리,
    # 여기서는 질문의 의도를 다시 분류하지 않습니다.
    #
    # prediction, probability, threshold, risk_level, evidence는
    # 이전 node인 evaluate_evidence에서 이미 정리된 값입니다.
    #
    # 따라서 LLM의 역할은 이 값을 다시 계산하거나 바꾸는 것이 아니라,
    # 사용자가 이해하기 쉬운 한국어 설명으로 변환하는 것입니다.
    #
    # 프롬프트에서는 state에 저장된 값을 세 종류로 나누어 다룹니다.
    #
    # 1. prediction
    # - 모델 또는 Tool이 이미 결정한 최종 판정입니다.
    # - LLM이 이 값을 다시 판단하거나 바꾸면 안 됩니다.
    #
    # 2. probability, risk_score, risk_level, threshold
    # - prediction을 설명하기 위한 계산값입니다.
    # - 이미 이전 node에서 계산된 값이므로 LLM이 수정하면 안 됩니다.
    #
    # 3. evidence
    # - prediction을 설명하기 위한 근거 목록입니다.
    # - LLM은 제공된 evidence만 사용할 수 있고,
    #   새로운 근거를 생성하거나 추론해서 추가하면 안 됩니다.
    #
    # 즉, LLM의 역할은 값을 계산하거나 근거를 만드는 것이 아니라,
    # 이미 정리된 결과를 사용자에게 이해하기 쉬운 문장으로 설명하는 것입니다.
    prompt = f"""
You are an assistant for a predictive maintenance AI agent.

The system has already calculated the following result.
Your role is only to explain the result in Korean.

[Previous Chat History]
{chat_history}

[Current Question]
{question}

[Workflow Context]
- intent: {intent}
- tool_name: {tool_name}

[Calculated Result]
- prediction: {prediction}
- probability: {probability}
- risk_score: {risk_score}
- risk_level: {risk_level}
- threshold: {threshold}
- evidence: {evidence}

[Raw Tool Result]
{tool_result}

Strict rules:
1. Do not change any calculated values.
   This includes prediction, probability, risk_score, risk_level, and threshold.

2. Do not make a new judgment.
   The prediction is already decided by the system.

3. Use only the provided evidence list and raw tool result.
   Do not create, infer, assume, or add new evidence.

4. If a value is None or missing, do not pretend that it exists.
   Instead, explain only the available information.

5. If the evidence list is empty, say that no evidence was provided.

6. Use the previous chat history only to understand context.
   Do not invent new model results from previous messages.

7. Write the answer in natural Korean for a non-technical user.
"""

    # OpenAI API 호출 구조:
    #
    # FastAPI는 사용자가 우리 서버에 요청을 보내는 입구입니다.
    # 예:
    # POST /agent/query
    #
    # 반면 OpenAI API는 우리 서버 코드가 외부 OpenAI 서버로 요청을 보내는 구조입니다.
    #
    # 전체 흐름:
    # 사용자 요청
    # → FastAPI endpoint
    # → LangGraph workflow 실행
    # → classify_intent_node 또는 generate_answer node
    # → client.responses.create(...)로 OpenAI API 호출
    # → OpenAI 서버가 응답 반환
    # → response.output_text로 모델의 텍스트 결과를 꺼냄
    # → AgentState에 저장
    # → FastAPI가 최종 JSON 응답 반환
    #
    # 즉, FastAPI는 "내 서비스의 API 서버"이고,
    # OpenAI API는 "내 서버가 호출하는 외부 AI API"입니다.
    try:
        # OpenAI Responses API를 호출합니다.
        #
        # client.responses.create(...)는 OpenAI 모델에게 요청을 보내고,
        # 모델이 생성한 응답을 response 객체로 반환합니다.
        #
        # 주요 입력값:
        # - model: 사용할 OpenAI 모델 이름입니다.
        # - input: 모델에게 전달할 프롬프트입니다.
        # - temperature: 출력의 무작위성을 조절하는 값입니다.
        #
        # 여기서는 계산값을 바꾸지 않고 일관된 설명을 받기 위해 0으로 설정합니다.
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            temperature=0,
        )

        # OpenAI API 응답은 response 객체로 반환됩니다.
        #
        # response 객체 안에는 output 배열, 모델 정보, 응답 id 등 여러 정보가 들어갈 수 있습니다.
        #
        # 그중 response.output_text는 모델이 생성한 텍스트 답변만
        # 하나의 문자열로 꺼내기 위한 편의 속성입니다.
        #
        # 최종 답변 생성 단계에서는 JSON이 아니라
        # 사용자에게 보여줄 자연어 문장을 받습니다.
        #
        # intent classification 단계에서는 output_text가 JSON 문자열이므로
        # json.loads(text)를 사용해 Python dict로 변환합니다.
        #
        # answer generation 단계에서는 output_text가 사용자에게 보여줄 자연어 답변이므로
        # json.loads()로 dict 변환하지 않고,
        # response.output_text를 그대로 answer에 저장합니다.
        answer = response.output_text.strip()

        # API 호출은 성공했지만 응답 텍스트가 비어 있을 수도 있으므로
        # 빈 답변이면 명시적으로 오류를 발생시켜 fallback 답변을 사용합니다.
        #
        # 예:
        # - response.output_text == ""
        # - response.output_text == "   "
        #
        # 이런 경우 그대로 state["answer"]에 저장하면
        # 사용자는 빈 답변을 받게 됩니다.
        #
        # 따라서 빈 답변이면 명시적으로 ValueError를 발생시켜
        # 아래 except 블록으로 이동시킵니다.
        if not answer:
            raise ValueError("OpenAI API returned empty answer.")

        state["answer"] = answer

    except Exception as error:
        # 최종 답변 생성 중 오류가 발생해도
        # Agent API가 완전히 실패하지 않도록 fallback 답변을 생성합니다.
        #
        # 실패할 수 있는 경우:
        # - OpenAI API 호출 실패
        # - response.output_text가 비어 있음
        # - 네트워크/API key 문제
        #
        # 이때 workflow를 중단하지 않고,
        # 오류는 state["errors"]에 저장하고,
        # 사용자는 최소한 현재까지 계산된 prediction, probability,
        # risk_level, threshold, evidence를 확인할 수 있게 합니다.
        state.setdefault("errors", []).append(str(error))

        # fallback 답변:
        # - LLM이 생성한 자연어 답변은 아니지만,
        # - 현재까지 AgentState에 저장된 계산 결과를 사용해
        #   사용자에게 최소한의 결과를 반환합니다.
        state["answer"] = build_fallback_answer(state)

    return state


def validate_answer(state: AgentState) -> AgentState:
    """
    LLM이 생성한 최종 답변이 AgentState의 계산값과 일치하는지 검증합니다.

    이 node의 목적:
    - LLM 답변이 prediction, probability, risk_level, threshold를 바꾸어 말하지 않았는지 확인합니다.
    - 제공되지 않은 evidence를 새로 만들어내지 않았는지 확인합니다.
    - 검증에 실패하면 LLM 답변 대신 fallback 답변을 제공합니다.
    """

    answer = state.get("answer", "")

    if not answer or not answer.strip():
        state["answer_valid"] = False
        state["answer_validation_reason"] = "answer is empty"
        state.setdefault("answer_validation_errors", []).append("empty_answer")
        state["answer"] = build_fallback_answer(state)
        return state

    # 검증용 프롬프트입니다.
    #
    # generate_answer 단계에서는 LLM이 자연어 답변을 생성합니다.
    # 하지만 LLM은 종종 계산값을 바꾸거나,
    # 제공되지 않은 근거를 그럴듯하게 추가할 수 있습니다.
    #
    # 그래서 validate_answer 단계에서
    # 생성된 answer가 state의 계산값과 일치하는지 다시 확인합니다.
    #
    # 검증 결과는 JSON으로 받습니다.
    #
    # is_valid:
    # - 답변이 state와 일치하면 true
    # - 모순되거나 없는 근거를 추가하면 false
    #
    # issues:
    # - 발견된 문제 목록
    #
    # reason:
    # - 검증 이유
    validation_prompt = f"""
You are a strict validator for a predictive maintenance AI agent.

Your task is to check whether the generated answer is consistent with the calculated state.

[Calculated State]
- prediction: {state.get("prediction")}
- probability: {state.get("probability")}
- risk_score: {state.get("risk_score")}
- risk_level: {state.get("risk_level")}
- threshold: {state.get("threshold")}
- evidence: {state.get("evidence", [])}

[Generated Answer]
{answer}

Validation rules:
1. The answer must not change prediction.
2. The answer must not change probability, risk_score, risk_level, or threshold.
3. The answer must not add evidence that is not included in the evidence list.
4. The answer must not pretend that missing or None values exist.
5. The answer may explain the calculated state in natural Korean.
6. If the answer is consistent with the calculated state, return is_valid=true.
7. If the answer contradicts or adds unsupported facts, return is_valid=false.

Return only valid JSON.
Do not include markdown.

JSON schema:
{{
  "is_valid": true,
  "issues": [],
  "reason": "short reason in Korean"
}}
"""

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=validation_prompt,
            temperature=0,
        )

        text = response.output_text.strip()

        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        result = json.loads(text)

        is_valid = bool(result.get("is_valid", False))
        issues = result.get("issues", [])
        reason = result.get("reason", "")

        if not isinstance(issues, list):
            issues = [str(issues)]

        state["answer_valid"] = is_valid
        state["answer_validation_reason"] = reason
        state["answer_validation_errors"] = issues

        if not is_valid:
            # LLM 답변이 state의 계산값과 일치하지 않으면
            # 검증 실패로 기록하고 fallback 답변으로 교체합니다.
            state.setdefault("errors", []).append(
                f"answer_validation_failed: {reason}"
            )
            state["answer"] = build_fallback_answer(state)

    except Exception as error:
        # 검증용 LLM 호출 자체가 실패한 경우입니다.
        #
        # 이 경우에도 검증되지 않은 LLM 답변을 그대로 제공하지 않고,
        # 더 안전한 fallback 답변으로 교체합니다.
        state["answer_valid"] = False
        state["answer_validation_reason"] = "answer validation failed"
        state.setdefault("errors", []).append(str(error))
        state["answer"] = build_fallback_answer(state)

    return state


def append_current_turn_to_history(state: AgentState) -> AgentState:
    """
    현재 턴의 user 질문과 assistant 답변을 chat_history에 추가합니다.

    이 함수는 멀티턴 Q&A를 위한 공통 후처리 함수입니다.

    왜 generate_answer 안에서 바로 append하지 않고 node로 분리하는가?
    - generate_answer는 답변을 생성하는 역할입니다.
    - validate_answer는 답변이 state 값과 일치하는지 검증하는 역할입니다.
    - append_current_turn_to_history는 최종 답변을 대화 이력에 저장하는 역할입니다.
    - 이렇게 분리하면 각 node의 책임이 명확해집니다.

    저장 시점:
    - validate_answer 이후에 저장합니다.
    - LLM 답변이 유효하면 LLM 답변이 저장됩니다.
    - LLM 답변이 유효하지 않으면 fallback 답변으로 교체된 뒤 저장됩니다.
    - 즉, 사용자에게 실제로 반환될 최종 answer가 history에 남습니다.

    대화 이력 관리 방식별 의미:

    1. state-only 방식
    - 현재 AgentState 안에만 chat_history를 누적합니다.
    - graph.invoke() 결과 안에서는 history를 확인할 수 있지만,
      다음 요청에 자동으로 이어지지는 않습니다.
    - 테스트와 흐름 확인용으로 가장 단순합니다.

    2. client-managed 방식
    - 서버는 응답에 업데이트된 chat_history를 포함해 반환합니다.
    - 클라이언트는 이 chat_history를 저장했다가 다음 요청 때 다시 보냅니다.
    - DB 없이도 멀티턴 Q&A를 구현할 수 있습니다.

    3. server-managed 방식
    - 서버가 session_id를 기준으로 DB/Redis/SQLite에 chat_history를 저장합니다.
    - 클라이언트는 chat_history 전체를 보낼 필요 없이 session_id만 보내면 됩니다.
    """

    question = state.get("question", "")
    answer = state.get("answer", "")

    # chat_history가 없으면 빈 리스트로 초기화합니다.
    #
    # 첫 질문이거나,
    # 테스트 코드에서 question만 전달한 경우에도
    # 안전하게 history append가 가능하도록 하기 위함입니다.
    state.setdefault("chat_history", [])

    # 현재 사용자 질문을 대화 이력에 추가합니다.
    state["chat_history"].append(
        {
            "role": "user",
            "content": question,
        }
    )

    # 현재 assistant의 답변을 대화 이력에 추가합니다.
    #
    # 이때 저장되는 답변은 사용자에게 실제로 반환될 최종 answer입니다.
    # LLM 답변 생성이 실패했다면 fallback answer가 저장됩니다.
    # LLM 답변 검증에 실패했다면 검증 후 교체된 fallback answer가 저장됩니다.
    state["chat_history"].append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    return state


def route_after_validation(
    state: AgentState,
) -> Literal["classify_intent", "generate_answer"]:
    """
    validate_question 이후 다음 node를 결정합니다.

    질문이 비어 있으면 intent classification을 호출할 필요가 없습니다.
    따라서 바로 generate_answer로 이동합니다.
    """

    # validate_question에서 errors가 생겼다면
    # 이후 classify_intent나 Tool 호출을 진행하지 않고 generate_answer로 이동합니다.
    #
    # 예:
    # - empty_question
    #
    # 이 경우 generate_answer에서는 값이 없는 항목을 없는 대로 설명하거나,
    # fallback 답변을 생성할 수 있습니다.
    if state.get("errors"):
        return "generate_answer"

    return "classify_intent"


def route_after_classification(
    state: AgentState,
) -> Literal["prepare_tool_args", "generate_answer"]:
    """
    classify_intent 이후 다음 node를 결정합니다.

    intent가 unknown이면 호출할 Tool이 없습니다.
    따라서 prepare_tool_args와 call_tool 단계를 건너뛰고
    바로 generate_answer로 이동합니다.
    """

    # state의 errors 항목에 error가 있으면
    # Tool 호출을 진행하지 않고 바로 답변 생성 단계로 이동합니다.
    if state.get("errors"):
        return "generate_answer"

    # intent가 unknown이면 호출할 Tool이 없습니다.
    #
    # 이 조건이 없으면 unknown 질문도 prepare_tool_args → call_tool로 이동할 수 있습니다.
    # 그러면 불필요하게 Tool 호출 단계를 거치게 됩니다.
    #
    # unknown은 지원하지 않는 질문이므로 바로 generate_answer로 이동합니다.
    if state.get("intent") == "unknown":
        return "generate_answer"

    return "prepare_tool_args"


def build_base_graph(
    prepare_history_node,
    use_server_save: bool = False,
):
    """
    공통 LangGraph 구조를 생성합니다.

    prepare_history_node만 바꾸면
    state-only, client-managed, server-managed 방식을 모두 만들 수 있습니다.

    use_server_save:
    - False: state-only 또는 client-managed 방식
    - True: server-managed 방식

    이 함수로 공통 구조를 분리하는 이유:
    - 세 가지 graph는 대부분 같은 node 흐름을 사용합니다.
    - 다른 점은 "history를 어디서 준비하느냐"와
      "마지막에 서버 저장소에 저장하느냐"입니다.
    - 따라서 공통 node/edge 구성은 하나의 함수로 묶고,
      다른 부분만 인자로 바꾸는 구조가 중복을 줄입니다.
    """

    # AgentState를 상태 타입으로 사용하는 LangGraph workflow를 생성합니다.
    #
    # AgentState는 질문, intent, tool_name, tool_result, prediction,
    # evidence, answer 등을 담고 이동하는 상태 객체입니다.
    #
    # 즉, 각 node는 AgentState를 입력으로 받고,
    # 필요한 값을 추가하거나 수정한 뒤 다시 AgentState를 반환합니다.
    workflow = StateGraph(AgentState)

    # prepare_history node를 등록합니다.
    #
    # 역할:
    # - state-only/client-managed/server-managed 중 어떤 방식인지에 따라
    #   chat_history를 준비합니다.
    # - 첫 질문 여부를 is_first_turn에 기록합니다.
    workflow.add_node("prepare_history", prepare_history_node)

    # validate_question node를 등록합니다.
    #
    # 역할:
    # - 사용자 질문이 비어 있는지 확인합니다.
    # - 비어 있으면 intent를 unknown으로 설정하고 errors에 기록합니다.
    # - trace_id가 없으면 새로 생성합니다.
    workflow.add_node("validate_question", validate_question)

    # classify_intent node를 등록합니다.
    #
    # 역할:
    # - OpenAI API를 사용해 사용자 질문의 intent를 분류합니다.
    # - 현재 question뿐 아니라 chat_history도 함께 참고합니다.
    # - 분류 결과를 state["intent"], state["confidence"],
    #   state["intent_reason"]에 저장합니다.
    workflow.add_node("classify_intent", classify_intent_node)

    # prepare_tool_args node를 등록합니다.
    #
    # 역할:
    # - intent에 맞는 tool_name을 선택합니다.
    # - Tool 호출에 필요한 기본 입력값을 state["tool_args"]에 준비합니다.
    workflow.add_node("prepare_tool_args", prepare_tool_args)

    # call_tool node를 등록합니다.
    #
    # 역할:
    # - 현재는 실제 MCP Tool이나 모델이 붙기 전이므로 mock Tool을 호출합니다.
    # - intent에 따라 가상의 tool_result를 생성해 state["tool_result"]에 저장합니다.
    #
    # 이후 실제 구현에서는 call_tool_mock 대신
    # 실제 모델 inference 함수나 MCP Tool 호출 함수로 교체할 수 있습니다.
    workflow.add_node("call_tool", call_tool_mock)

    # evaluate_evidence node를 등록합니다.
    #
    # 역할:
    # - tool_result에 들어있는 raw 결과를 표준 state 필드로 정리합니다.
    # - prediction, probability, risk_score, risk_level, threshold, evidence를
    #   최종 답변 생성 node가 사용하기 쉽게 state에 저장합니다.
    workflow.add_node("evaluate_evidence", evaluate_evidence)

    # generate_answer node를 등록합니다.
    #
    # 역할:
    # - evaluate_evidence에서 정리한 값을 바탕으로
    #   OpenAI API를 사용해 사용자에게 보여줄 최종 답변을 생성합니다.
    # - 이 단계에서 LLM은 값을 새로 판단하지 않고,
    #   이미 계산된 결과를 자연어로 설명하는 역할만 합니다.
    workflow.add_node("generate_answer", generate_answer)

    # validate_answer node를 등록합니다.
    #
    # 역할:
    # - generate_answer에서 생성된 LLM 답변이
    #   state의 계산값과 일치하는지 검증합니다.
    # - 모순되거나 없는 근거를 추가한 경우 fallback 답변으로 교체합니다.
    workflow.add_node("validate_answer", validate_answer)

    # append_history node를 등록합니다.
    #
    # 역할:
    # - 검증까지 끝난 최종 answer를 chat_history에 추가합니다.
    # - 이 시점의 answer가 실제 사용자에게 반환될 답변입니다.
    workflow.add_node("append_history", append_current_turn_to_history)

    # server-managed 방식에서는 마지막에 chat_history를 SQLite에 저장해야 합니다.
    #
    # state-only/client-managed 방식에서는 서버 저장소에 저장하지 않으므로
    # 이 node를 등록하지 않습니다.
    if use_server_save:
        workflow.add_node("save_history", save_server_managed_history)

    # START는 LangGraph workflow의 시작 지점입니다.
    #
    # 그래프가 실행되면 가장 먼저 prepare_history node로 이동합니다.
    workflow.add_edge(START, "prepare_history")

    # history 준비가 끝나면 질문 검증 단계로 이동합니다.
    workflow.add_edge("prepare_history", "validate_question")

    # validate_question 이후 조건 분기:
    # - 정상 질문이면 classify_intent로 이동합니다.
    # - 질문이 비어 있거나 errors가 있으면 바로 generate_answer로 이동합니다.
    workflow.add_conditional_edges(
        "validate_question",
        route_after_validation,
        {
            "classify_intent": "classify_intent",
            "generate_answer": "generate_answer",
        },
    )

    # classify_intent 이후 조건 분기:
    # - 지원 가능한 intent면 prepare_tool_args로 이동합니다.
    # - unknown 또는 errors가 있으면 Tool 호출을 생략하고 generate_answer로 이동합니다.
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "prepare_tool_args": "prepare_tool_args",
            "generate_answer": "generate_answer",
        },
    )

    # Tool 이름과 입력값 준비가 끝나면 실제 Tool 호출 단계로 이동합니다.
    workflow.add_edge("prepare_tool_args", "call_tool")

    # Tool 실행 결과가 생성되면 evidence 평가 단계로 이동합니다.
    workflow.add_edge("call_tool", "evaluate_evidence")

    # prediction, probability, evidence 등이 정리되면 최종 답변 생성 단계로 이동합니다.
    workflow.add_edge("evaluate_evidence", "generate_answer")

    # 최종 답변 생성 후 바로 반환하지 않고,
    # 먼저 답변이 계산값과 일치하는지 검증합니다.
    workflow.add_edge("generate_answer", "validate_answer")

    # 답변 검증이 끝나면 현재 턴의 user/assistant 메시지를 chat_history에 추가합니다.
    workflow.add_edge("validate_answer", "append_history")

    if use_server_save:
        # server-managed 방식에서는 chat_history를 state에 추가한 뒤,
        # SQLite 저장소에 저장합니다.
        workflow.add_edge("append_history", "save_history")

        # 저장까지 끝나면 workflow를 종료합니다.
        workflow.add_edge("save_history", END)

    else:
        # state-only/client-managed 방식에서는 서버 저장소에 저장하지 않고 종료합니다.
        workflow.add_edge("append_history", END)

    # 지금까지 등록한 node와 edge를 실행 가능한 graph로 컴파일합니다.
    #
    # compile()을 호출해야 실제로 graph.invoke(...) 같은 방식으로
    # workflow를 실행할 수 있습니다.
    return workflow.compile()


def build_state_only_graph():
    """
    1번 방식: state-only graph

    특징:
    - 요청 1회 안에서만 chat_history를 유지합니다.
    - 서버나 클라이언트에 history를 저장하지 않습니다.
    - graph.invoke() 테스트용으로 가장 단순합니다.
    """

    return build_base_graph(
        prepare_history_node=prepare_state_only_history,
        use_server_save=False,
    )


def build_client_managed_graph():
    """
    2번 방식: client-managed graph

    특징:
    - 요청 body에 들어온 chat_history를 사용합니다.
    - 답변 후 업데이트된 chat_history를 응답으로 반환합니다.
    - 클라이언트가 다음 요청 때 chat_history를 다시 보내야 멀티턴이 이어집니다.
    """

    return build_base_graph(
        prepare_history_node=prepare_client_managed_history,
        use_server_save=False,
    )


def build_server_managed_graph():
    """
    3번 방식: server-managed graph

    특징:
    - session_id 기준으로 서버가 chat_history를 불러옵니다.
    - 답변 후 chat_history를 SQLite에 저장합니다.
    - 클라이언트는 다음 요청 때 session_id만 보내도 대화가 이어집니다.
    """

    return build_base_graph(
        prepare_history_node=prepare_server_managed_history,
        use_server_save=True,
    )


# 1번 방식 graph입니다.
#
# 학습용 / graph.invoke 테스트에 적합합니다.
state_only_graph = build_state_only_graph()

# 2번 방식 graph입니다.
#
# 현재 프로젝트 초기 구현 기준으로 추천하는 방식입니다.
# Swagger나 프론트엔드에서 chat_history를 직접 주고받으며 테스트할 수 있습니다.
client_managed_graph = build_client_managed_graph()

# 3번 방식 graph입니다.
#
# session_id 기준으로 SQLite에 chat_history를 저장하는 실제 서비스형 방식입니다.
server_managed_graph = build_server_managed_graph()


# 기본 graph는 현재 프로젝트 초기 구현 기준으로 client-managed 방식을 사용합니다.
#
# 이유:
# - DB 없이 멀티턴 Q&A를 테스트할 수 있습니다.
# - Swagger나 프론트엔드에서 chat_history를 직접 확인하기 쉽습니다.
# - 이후 server_managed_graph로 바꾸면 session_id 기반 저장 방식으로 확장할 수 있습니다.
#
# 사용 예:
# result = graph.invoke({
#     "question": "이 설비 고장 위험 예측해줘",
#     "chat_history": []
# })
#
# server-managed 방식으로 바꾸고 싶다면:
# graph = server_managed_graph
graph = client_managed_graph
