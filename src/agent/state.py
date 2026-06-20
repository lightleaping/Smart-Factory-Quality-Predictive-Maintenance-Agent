from typing import Any, Literal, NotRequired, Required, TypedDict


# total=False는 TypedDict에서 "모든 필드를 처음부터 반드시 넣지 않아도 된다"는 뜻입니다.
#
# LangGraph workflow에서는 처음부터 모든 값이 채워져 있지 않습니다.
# 처음에는 question만 들어오고,
# 각 node를 지나면서 intent, tool_name, tool_result, prediction, answer 등이
# 순서대로 추가됩니다.
class AgentState(TypedDict, total=False):
    """
    AgentState는 LangGraph workflow 안에서 각 node가 공유하는 상태입니다.

    사용자의 질문이 들어오면,
    각 node가 intent, tool_name, tool_result, prediction, answer 등을
    하나씩 채워가며 최종 응답을 만듭니다.
    """

    # 사용자가 이번 턴에 입력한 질문입니다.
    #
    # FastAPI /agent/query 또는 graph.invoke()에서 가장 먼저 들어오는 필수 값입니다.
    question: Required[str]

    # 여러 턴의 대화를 저장하기 위한 필드입니다.
    #
    # 기존 Agent 구조는 question 하나를 입력받고 answer 하나를 반환하는
    # single-turn Q&A 구조였습니다.
    #
    # 채팅 Q&A로 확장하려면 이전 user/assistant 메시지를 저장해야 합니다.
    #
    # 예:
    # [
    #   {"role": "user", "content": "이 설비 고장 위험 예측해줘"},
    #   {"role": "assistant", "content": "고장 가능성이 높습니다."},
    #   {"role": "user", "content": "왜 그렇게 판단했어?"}
    # ]
    #
    # chat_history는 두 곳에서 중요합니다.
    #
    # 1. classify_intent 단계
    # - "왜 그렇게 판단했어?", "그럼 조치는?" 같은 후속 질문은
    #   현재 문장만으로 intent를 알기 어렵습니다.
    # - 따라서 intent 분류 단계에서도 이전 대화 맥락을 참고해야 합니다.
    #
    # 2. generate_answer 단계
    # - 최종 답변을 생성할 때 이전 대화 흐름을 참고해
    #   자연스럽게 이어지는 답변을 만들 수 있습니다.
    #
    # chat_history 관리 방식은 크게 3가지입니다.
    #
    # 1. state-only 방식
    # - 현재 LangGraph 실행 중 AgentState 안에서만 대화 이력을 유지합니다.
    # - 장점: 가장 단순하고 graph.invoke() 테스트에 좋습니다.
    # - 단점: 요청이 끝나면 state도 끝나므로 다음 요청에서 자동으로 이어지지 않습니다.
    # - 프로그램이 꺼지면 당연히 사라집니다.
    #
    # 2. client-managed 방식
    # - 서버가 응답에 업데이트된 chat_history를 포함해 반환하고,
    #   클라이언트가 다음 요청 때 그 chat_history를 다시 보내는 방식입니다.
    # - 장점: DB/Redis 없이도 멀티턴 Q&A를 구현할 수 있습니다.
    # - 단점: 대화가 길어질수록 요청 body가 커지고,
    #   클라이언트가 history를 계속 들고 있어야 합니다.
    # - 서버 프로그램이 꺼져도 클라이언트가 history를 보관하고 있다면 다시 이어갈 수 있지만,
    #   클라이언트가 history를 잃어버리면 대화도 이어지지 않습니다.
    #
    # 3. server-managed 방식
    # - 서버가 session_id를 기준으로 DB/Redis/SQLite 등에 chat_history를 저장합니다.
    # - 장점: 실제 서비스 구조에 가깝고,
    #   클라이언트는 chat_history 전체가 아니라 session_id만 보내도 됩니다.
    # - 단점: 저장소, session 관리, 리셋 정책이 필요해 구현이 더 복잡합니다.
    # - DB 파일이나 Redis/PostgreSQL 같은 저장소가 유지되면 서버를 껐다 켜도 대화가 남습니다.
    #
    # 결론:
    # - 학습/테스트: state-only
    # - 초기 구현/Swagger 테스트: client-managed
    # - 실제 서비스 확장: server-managed
    chat_history: NotRequired[list[dict[str, str]]]

    # 서버 저장 방식에서 사용하는 세션 ID입니다.
    #
    # session_id는 "어떤 대화방의 기록인지" 구분하는 값입니다.
    #
    # server-managed 방식에서는 다음 흐름으로 동작합니다.
    #
    # 1. 첫 질문에서 session_id가 없으면 서버가 새 session_id를 생성합니다.
    # 2. 같은 대화를 이어갈 때는 클라이언트가 session_id를 다시 보냅니다.
    # 3. 서버는 session_id 기준으로 chat_history를 SQLite/DB에서 조회합니다.
    # 4. 저장된 history가 없으면 첫 질문으로 판단합니다.
    # 5. 답변이 끝나면 현재 질문과 답변을 같은 session_id에 저장합니다.
    #
    # 첫 질문으로 보는 경우:
    # - session_id가 없는 경우
    # - session_id는 있지만 DB에 해당 session_id가 없는 경우
    # - DB에 row는 있지만 chat_history가 빈 리스트인 경우
    # - DB에 저장된 chat_history JSON이 깨져서 사용할 수 없는 경우
    #
    # 사용자가 새 대화를 시작하려면:
    # - 새 session_id를 발급하거나
    # - 기존 session_id의 history를 삭제하면 됩니다.
    session_id: NotRequired[str]

    # 현재 질문이 해당 세션의 첫 번째 질문인지 표시합니다.
    #
    # 이 값은 필수는 아니지만,
    # 디버깅이나 응답 정책을 나눌 때 유용합니다.
    #
    # 예:
    # - 첫 질문이면 "이전 대화 맥락이 없습니다"라고 처리할 수 있습니다.
    # - 후속 질문이면 chat_history를 적극적으로 참고할 수 있습니다.
    is_first_turn: NotRequired[bool]

    # intent 분류 결과입니다.
    #
    # 예:
    # - machine_failure_prediction
    # - failure_type_analysis
    # - model_metric_query
    # - dataset_schema_query
    # - unknown
    #
    # 여기서는 Literal[...]로 제한하지 않고 str로 둡니다.
    #
    # 이유:
    # - intent는 Python 코드가 직접 정하는 값이 아니라 LLM 응답에서 가져오는 값입니다.
    # - LLM은 프롬프트에서 정해진 intent만 반환하라고 지시해도,
    #   오타나 예상하지 못한 문자열을 반환할 가능성이 있습니다.
    # - TypedDict의 Literal은 정적 타입 검사에는 도움이 되지만,
    #   런타임에서 들어오는 LLM 출력값을 실제로 막아주지는 않습니다.
    #
    # 따라서 state에서는 우선 str로 받은 뒤,
    # classify_intent() 안에서 SUPPORTED_INTENTS에 포함되는지 검사합니다.
    #
    # 만약 지원하지 않는 intent가 반환되면
    # "unknown"으로 바꿔 workflow가 안전하게 이어지도록 처리합니다.
    intent: NotRequired[str]

    # intent 분류에 대한 확신도입니다.
    #
    # 현재는 LLM이 반환한 confidence score를 저장합니다.
    confidence: NotRequired[float]

    # intent 분류 이유입니다.
    #
    # LLM이 왜 해당 intent로 판단했는지 짧게 저장합니다.
    intent_reason: NotRequired[str]

    # 호출할 Tool 이름입니다.
    #
    # 예:
    # - predict_machine_failure
    # - analyze_failure_type
    # - get_model_metrics
    # - get_dataset_schema
    # - none
    #
    # 여기서는 Literal[...]로 제한하지 않고 str로 둡니다.
    #
    # 이유:
    # - "none"은 실제 Tool 이름이 아니라, 호출할 Tool이 없다는 의미의 sentinel 값입니다.
    # - sentinel 값이란 코드 안에서 특별한 의미를 가지도록 약속한 표시용 값입니다.
    # - 여기서 "none"은 Python의 None 객체가 아니라,
    #   "Tool을 호출하지 않는다"는 의미로 정한 문자열입니다.
    #
    # 또한 Tool 이름은 실제 MCP Tool, DB 조회 Tool, 모델 inference Tool이 추가되면서
    # 계속 바뀔 수 있습니다.
    #
    # Literal로 너무 일찍 고정하면 새로운 Tool을 추가할 때마다
    # AgentState 타입도 함께 수정해야 합니다.
    #
    # 따라서 state에서는 tool_name을 str로 유연하게 저장하고,
    # 실제 허용 Tool 여부는 INTENT_TO_TOOL 또는 Tool registry에서 관리합니다.
    tool_name: NotRequired[str]

    # Tool에 전달할 입력값입니다.
    #
    # question, intent, dataset, feature 값 등이 들어갈 수 있으므로 dict로 둡니다.
    # tool_input보다 tool_args라는 이름이 더 자연스럽습니다.
    # Tool에 들어가는 값은 보통 문자열 하나가 아니라 여러 값이기 때문입니다.
    tool_args: NotRequired[dict[str, Any]]

    # Tool 실행 결과입니다.
    #
    # prediction, probability, threshold, evidence, metric 등이 들어갈 수 있습니다.
    # Tool 실행 결과는 단순 문장이 아니라 모델 결과, 근거, metric 등을 포함할 수 있습니다.
    tool_result: NotRequired[dict[str, Any]]

    # 모델 예측 결과입니다.
    #
    # 현재 mock에서는 0/1 숫자를 사용하지만,
    # 실제 모델이나 Tool에 따라 "normal", "failure" 같은 문자열이 들어올 수도 있습니다.
    #
    # 그래서 int로 강하게 고정하기보다 Any로 열어두면 확장에 유리합니다.
    prediction: NotRequired[Any]

    # 모델이 예측한 실패 또는 고장 가능성 점수입니다.
    #
    # 보통 0.0 ~ 1.0 사이 값으로 사용합니다.
    probability: NotRequired[float]

    # 예측 판단 기준값입니다.
    #
    # probability가 threshold 이상이면 위험으로 판단할 수 있습니다.
    threshold: NotRequired[float]

    # 숫자형 위험 점수입니다.
    #
    # 예:
    # probability 0.82 -> risk_score 82.0
    risk_score: NotRequired[float]

    # 사람이 이해하기 쉬운 위험 등급입니다.
    #
    # risk_level은 Python 코드에서 직접 LOW/MEDIUM/HIGH 중 하나로 넣는 값입니다.
    # LLM 출력값이 아니라 내부 계산 결과이므로 Literal로 제한해도 괜찮습니다.
    #
    # UNKNOWN은 Literal에 넣지 않는 것을 추천합니다.
    # 값이 없으면 risk_level 필드 자체가 없는 상태로 두는 편이 더 자연스럽습니다.
    risk_level: NotRequired[Literal["LOW", "MEDIUM", "HIGH"]]

    # 추천 조치입니다.
    #
    # 예:
    # "해당 설비의 최근 센서 로그와 점검 이력을 확인하세요."
    recommended_action: NotRequired[str]

    # 최종 답변입니다.
    answer: NotRequired[str]

    # 답변의 근거입니다.
    #
    # 근거는 하나가 아닐 수 있습니다.
    # 모델 결과, feature 값, metric, threshold 정보 등이 들어갈 수 있습니다.
    evidence: NotRequired[list[dict[str, Any]]]

    # LLM 답변 검증 결과입니다.
    #
    # generate_answer에서 만든 자연어 답변이
    # state의 prediction, probability, risk_level, threshold, evidence와
    # 일치하는지 검증한 결과를 저장합니다.
    answer_valid: NotRequired[bool]

    # LLM 답변 검증 이유입니다.
    answer_validation_reason: NotRequired[str]

    # LLM 답변 검증 중 발견된 문제 목록입니다.
    answer_validation_errors: NotRequired[list[str]]

    # 실행 추적 ID입니다.
    #
    # trace log와 API 응답을 연결하기 위한 값입니다.
    # trace_id는 보통 숫자보다 문자열 UUID 형태가 좋습니다.
    trace_id: NotRequired[str]

    # 에러 목록입니다.
    #
    # validate_question, OpenAI API 호출, tool call, model inference 중 발생한 문제를 저장합니다.
    errors: NotRequired[list[str]]