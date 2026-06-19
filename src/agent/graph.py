# JSON 문자열을 Python dict로 바꾸기 위해 사용한다.
import json
# 환경변수를 읽기 위해 사용한다. 예를 들어 .env에 저장한 OPENAI_MODEL 값을 가져올 수 있다.
import os

from dotenv import load_dotenv

# OpenAI API를 호출하기 위한 클라이언트 클래스를 가져온다.
from openai import OpenAI

from src.agent.state import AgentState

# 현재 프로젝트의 .env 파일을 읽어온다.
load_dotenv()

# OpenAI API를 호출할 client 객체를 만든다.
# 보통 OPENAI_API_KEY가 환경 변수에 있으면 자동으로 사용한다.
client = OpenAI()

# 환경변수에서 OPENAI_MODEL 값을 읽어오고, 없으면 기본값으로 gpt-4o 사용한다.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

SUPPORTED_INTENTS = {
    # 설비 고장 위험 예측
    "machine_failure_prediction",

    # 고장 유형 또는 원인 분석
    "failuere_type_analysis",

    # 모델 성능, F1-score, recall, threshold 질문
    "model_metric_query",

    # 데이터 컬럼, feature, target 질문
    "dataset_schema_query",

    # 지원하지 않는 질문
    "unknown",
}

def classify_intent(question: str) -> dict[str, Any]:
    """
    이번 프로젝트에서는 LLM 기반 classification을 이용합니다.

    question이라는 문자열을 입력 받고, intent 문자열을 반환합니다.

    즉, 사용자의 질문을 보고 아래 intent 중 하나로 분류합니다.

    - machine_failure_prediction
    - failure_type_analysis
    - model_metric_query
    - dataset_schema_query
    - unknown
    
    반환값은 반드시 SUPPORTED_INTENTS 중 하나여야 합니다.
    LLM이 잘못된 형식을 반환하거나 오류가 발생하면 unknown을 반환합니다.

    """

    if not question or not question.strip():
        return "unknown"
    
    prompt = f"""
You are an intent classifier for a predictive maintenance AI agent.

Classify the user's question into exactly one of the following intents:

1. machine_failure_prediction
- The user wants to predict whether a machine is likely to fail.
- Examples: "이 설비 고장 위험 예측해줘", "고장 가능성 알려줘"

2. failure_type_analysis
- The user wants to know the likely failure type or cause.
- Examples: "어떤 유형의 고장이야?", "고장 원인 후보 알려줘"

3. model_metric_query
- The user asks about model performance, metrics, threshold, recall, precision, F1-score, ROC-AUC.
- Examples: "모델 성능은 어때?", "F1-score 알려줘"

4. dataset_schema_query
- The user asks about dataset columns, features, target, or meaning of fields.
- Examples: "이 데이터 컬럼 설명해줘", "Target은 뭐야?"

5. unknown
- The question is outside the supported scope.

Return only valid JSON.
Do not include markdown.
Do not include explanation outside JSON.

JSON schema:
{{
"intent": "one of the supported intents"
}}

User question:
{question}
"""
    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,

            # 출력을 최대한 일관되게 만들기 위한 설정이다.
            temperature=0,
        )

        text = response.output_text.strip()

        # 혹시 모델이 ```json ... ``` 형태로 반환했을 때를 대비한 간단한 정리
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        # JSON 문자열을 Python dict로 바꾼다.
        result = json.loads(text)
        intent = result.get("intent", "unknown")

    except Exception:
        return "unknown"
    
    if intent not in SUPPORTED_INTENTS:
        return "unknown"
    
    return intent
