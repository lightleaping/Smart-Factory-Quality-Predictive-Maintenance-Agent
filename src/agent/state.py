from typing import Any, Literal, NotRequired, Required, TypedDict

# total=False는 'TypedDict에서 “모든 필드를 처음부터 반드시 넣지 않아도 된다”는 뜻'
class AgentState(TypedDict, total=False):
    """
    AgentState는 LangGraph workflow 안에서 각 node가 공유하는 상태입니다.

    사용자의 질문이 들어오면,
    각 node가 intent, tool_name, tool_result, prediction, answer 등을
    하나씩 채워가며 최종 응답을 만듭니다.
    """

    # 사용자 질문
    # FastAPI /agent/query 또는 graph.invoke()에서 가장 먼저 들어오는 값입니다.
    question: Required[str]
    
    # 이후 Literal로 제한한다.
    # intent 분류 결과
    # 예 : production_quality_failure, machine_failure_prediction, unknown
    intent: NotRequired[str]
    
    # intent 분류에 대한 확신도
    # rule 기반이면 1.0 또는 0.7처럼 임의 기준을 줄 수 있고,
    # LLM이 반환한 confidence score를 저장합니다.
    confidence: NotRequired[float]

    # intent 분류 이유
    # LLM이 왜 해당 intent로 판단했는지 짧게 저장합니다.
    intent_reason: NotRequired[str]

    # 이후 Literal로 제한한다.
    # 호출할 Tool 이름
    # 예 : predict_production_line_failure, predict_machine_failure
    tool_name: NotRequired[str]

    # Tool에 전달할 입력값
    # sample_id, dataset, feature 값 등이 들어갈 수 있으므로 dict로 둡니다.
    # tool_input보다는 tool_args. Tool에 들어가는 값은 보통 문자열 하나가 아니라 여러 값이 들어갈 수 있다.
    tool_args: NotRequired[dict[str, Any]]

    # Tool 실행 결과
    # prediction, probability, threshold, evidence 등이 들어갈 수 있습니다.
    # Tool 실행 결과는 단순 문장이 아니라 모델 결과, 근거, metric 등이 들어갈 수 있다.
    tool_result: NotRequired[dict[str, Any]]

    # 모델 예측 결과
    # 예 : 0 = 정상, 1 = 실패/고장 위험
    prediction: NotRequired[int]
    
    # 모델이 예측한 실패 또는 고장 확률
    probability: NotRequired[float]

    # 예측 판단 기준값
    # probability가 threshold 이상이면 위험으로 판단할 수 있습니다.
    threshold: NotRequired[float]

    # 숫자형 위험 점수
    # 예 : probability 0.82 -> risk_score 82.0
    risk_score: NotRequired[float]

    # 사람이 이해하기 쉬운 위험 등급
    risk_level: NotRequired[Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]]
    
    # 추천 조치
    # 예 : "해당 설비의 최근 센서 로그와 점검 이력을 확인하세요."
    recommended_action: NotRequired[str]

    # 답변
    answer: NotRequired[str]

    # 답변의 근거. 근거는 하나가 아닐 수 있음.
    # 모델 결과, feature importance, metric, threshold 정보 등이 들어갈 수 있습니다.
    evidence: NotRequired[list[dict[str, Any]]]
    
    # 실행 추적 ID
    # trace log와 API 응답을 연결하기 위한 값입니다.
    # trace_id는 보통 숫자보다 문자열 UUID 형태가 좋다.
    trace_id: NotRequired[str]
    
    # 에러 목록
    # validate_question, tool call, model inference 중 발생한 문제를 저장합니다.
    errors: NotRequired[list[str]]
