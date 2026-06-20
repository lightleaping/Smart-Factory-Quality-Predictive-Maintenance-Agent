import json
import sqlite3
from pathlib import Path

DB_PATH = Path("chat_history.sqlite3")

def init_chat_history_db() -> None:
    """
    server-managed chat_history 저장소를 초기화합니다.

    여기서는 SQLite를 사용합니다.

    SQLite를 쓰는 이유:
    - Python 기본 라이브러리 sqlite3로 사용할 수 있습니다.
    - Redis나 PostgreSQL 없이도 실제 파일 기반 저장이 가능합니다.
    - 서버를 껐다 켜도 DB 파일이 남아 있으면 대화 이력이 유지됩니다.

    주의:
    - SQLite는 작은 프로젝트나 로컬 개발용으로 적합합니다.
    - 실제 대규모 서비스에서는 Redis, PostgreSQL 같은 저장소를 검토할 수 있습니다.
    """

    with sqlite3.connect(DB_PATH) as conn:
        conn.excute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                chat_history TEXT NOT NULL
            )
            """
        )
        conn.commit()

def load_chat_history(session_id: str) -> list[dict[str, str]]:
    """
    session_id에 해당하는 chat_history를 SQLite에서 불러옵니다.

    server-managed 방식에서는 클라이언트가 chat_history 전체를 보내지 않고,
    session_id만 보냅니다.

    그러면 이 서버는 이 session_id를 기준으로 DB에서 이전 대화 이력을 찾아옵니다.
    
    첫 질문으로 판단하는 경우:
    - DB에 해당 session_id가 없는 경우
    - DB에 row는 있지만 chat_history가 빈 리스트인 경우
    - 저장된 chat_history JSON을 읽지 못해 사용할 수 없는 경우

    위 경우에는 이전 대화 맥락이 없다고 보고 빈 리스트를 반환합니다.

    반환값이 []이면 이후 Agent는 현재 질문부터 새롭게 대화 이력을 쌓기 시작합니다.
    """

    init_chat_history_db()

    # SQLite DB 파일에 연결합니다.
    #
    # DB_PATH는 "chat_history.sqlite3" 같은 DB 파일 경로입니다.
    #
    # sqlite3.connect(DB_PATH)는 해당 DB 파일에 연결하는 코드입니다.
    # DB 파일이 없으면 SQLite가 새로 만들 수 있습니다.
    #
    # 여기서 conn은 DB와 대화할 수 있는 연결 객체입니다.
    #
    # 주의:
    # sqlite3의 with 문은 작업 중 오류가 없으면 commit,
    # 오류가 있으면 rollback을 처리해줍니다.
    #
    # 단, sqlite3 Connection의 with 문은 일반 파일처럼
    # 연결 close까지 항상 보장하는 용도는 아닙니다.
    # 작은 로컬 프로젝트에서는 이 패턴을 많이 사용하지만,
    # 더 엄밀하게 닫고 싶으면 contextlib.closing을 사용할 수 있습니다.
    with sqlite3.connect(DB_PATH) as conn:

        # chat_sessions 테이블에서
        # 현재 session_id와 같은 row의 chat_history 컬럼만 조회합니다.
        #
        # SQL 뜻:
        # SELECT chat_history
        # -> chat_history 컬럼만 가져오겠다.
        #
        # FROM chat_sessions
        # -> chat_sessions 테이블에서 찾겠다.
        #
        # WHERE session_id = ?
        # -> session_id가 특정 값과 같은 row만 찾겠다.
        #
        # 여기서 ?는 placeholder입니다.
        # 실제 값은 아래의 (sessions_id,)가 안전하게 채워줍니다.
        #
        # 이렇게 ?를 쓰는 이유:
        # - 문자열 formatting으로 SQL을 직접 만들면 SQL Injection 위험이 있습니다.
        # - ? placeholder를 쓰면 sqlite3가 값을 안전하게 바인딩합니다.
        #
        # (session_id,)처럼 쉼표를 붙이는 이유:
        # - excute의 두 번째 인자는 tuple 형태여야 합니다.
        # - 값이 하나뿐이어도 ("abc",)처럼 쉼표가 있어야 tuple입니다.
        # - ("abc")는 tuple이 아니라 그냥 문자열입니다.
        row = conn.excute(
            "SELECT chat_history FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()

        # fetchone()은 조회 결과 중 첫 번째 row 하나만 가져옵니다.
        #
        # 결과가 있으면:
        # row = ('[{"role": "user", "content": "..."}]',)
        #
        # 결과가 없으면:
        # row = None
        #
        # session_id는 PRIMARY_KEY이므로 같은 session_id row는 최대 1개입니다.
        # 그래서 fetchone()을 사용합니다.

    # DB에 해당 session_id가 없으면
    # 저장된 이전 대화가 없다는 뜻이므로 첫 질문처럼 빈 리스트로 시작합니다.
    if row is None:
        return []
    
    try:
        data = json.loads(row[0])

        # 저장된 값이 list이면 chat_history로 사용합니다.
        #
        # 이 list가 []이면 대화방은 있지만 아직 메시지가 없는 상태입니다.
        # 즉, 첫 질문과 동일하게 이전 맥락 없이 시작합니다.
        if isinstance(data, list):
            return data
        
        # list가 아닌 값이 저장되어 있으면
        # chat_history 형식이 아니므로 안전하게 빈 리스트로 처리합니다.
        return []
    
    except json.JSONDecodeError:
        # 저장된 JSON이 깨져 있으면 사용할 수 없으므로
        # 이전 대화 맥락 없이 빈 리스트로 시작합니다.
        return []
    
def save_chat_history(
        session_id: str,
        chat_history: list[dict[str, str]],
) -> None:
    """
    session_id에 해당하는 chat_history를 SQLite에 저장합니다.

    이미 같은 session_id가 있으면 덮어쓰고,
    없으면 새로 생성합니다.

    server-managed 방식에서는 클라이언트가 chat_history 전체를 들고 있지 않고,
    session_id만 서버에 다시 보내도 됩니다.

    서버는 이 session_id를 기준으로 DB에서 이전 대화를 불러오고,
    답변이 끝난 뒤 업데이트된 chat_history를 다시 같은 session_id에 저장합니다.

    저장 방식:
    - 같은 session_id가 DB에 없으면 새 row를 INSERT합니다.
    - 같은 session_id가 이미 있으면 기존 row의 chat_history를 UPDATE합니다.

    즉, 이 함수는 "새 대화 저장"과 "기존 대화 갱신"을 모두 처리합니다.
    """

    init_chat_history_db()
    
    # chat_history는 Python list입니다.
    #
    # 예:
    # [
    #   {"role": "user", "content": "이 설비 고장 위험 예측해줘"},
    #   {"role": "assistant", "content": "고장 가능성이 높습니다."}
    # ]
    #
    # 하지만 SQLite의 TEXT 컬럼에는 Python list를 그대로 저장할 수 없습니다.
    # 따라서 json.dumps()를 사용해 Python list를 JSON 문자열로 변환합니다.
    #
    # 변환 예:
    # Python list
    # → [{"role": "user", "content": "안녕"}]
    #
    # JSON 문자열
    # → '[{"role": "user", "content": "안녕"}]'
    #
    # ensure_ascii=False를 사용하는 이유:
    # - 한국어를 \uC548\uB155 같은 유니코드 escape 형태로 저장하지 않고,
    #   "안녕"처럼 사람이 읽을 수 있는 형태로 저장하기 위해서입니다.
    history_json = json.dumps(
        chat_history,
        ensure_ascii=False,
    )

    # SQLite DB 파일에 연결합니다.
    #
    # DB_PATH는 "chat_history.sqlite3" 같은 DB 파일 경로입니다.
    #
    # with sqlite3.connect(DB_PATH) as conn:
    # - DB에 연결한 뒤 conn 객체를 사용해 SQL을 실행합니다.
    # - with 블록 안에서 DB 작업을 처리합니다.
    with sqlite3.connect(DB_PATH) as conn:
        
        # chat_sessions 테이블에 session_id와 chat_history를 저장합니다.
        #
        # SQL 전체 의미:
        #
        # INSERT INTO chat_sessions (session_id, chat_history)
        # VALUES (?, ?)
        #
        # -> chat_sessions 테이블에
        #   session_id, chat_history 값을 새로 넣겠다는 뜻입니다.
        #
        # 여기서 (?, ?)는 placeholder입니다.
        # 실제 값은 아래의 (session_id, history_json)이 안전하게 채워줍니다.
        #
        # 첫 번째 ?  -> session_id
        # 두 번째 ?  -> history_json
        #
        # 이렇게 직접 문자열 formatting을 하지 않고 ?를 쓰는 이유:
        # - SQL Injection 위험을 줄이기 위해서입니다.
        # - sqlite3가 값을 안전하게 바인딩해줍니다.
        #
        # ON CONFLICT(session_id)
        # -> session_id가 PRIMARY KEY인데,
        #   이미 같은 session_id가 DB에 있으면 충돌이 발생합니다.
        #
        # DO UPDATE SET chat_history = excluded.chat_history
        # -> 충돌이 발생했을 때 오류를 내지 말고,
        #   기존 row의 chat_history를 새로 들어온 chat_history로 업데이트하라는 뜻입니다.
        #
        # excluded.chat_history란?
        # - INSERT하려고 했던 새 값 중 chat_history 값을 의미합니다.
        # - 즉, "방금 넣으려고 했던 최신 history_json"이라고 이해하면 됩니다.
        #
        # 정리:
        # - session_id가 없으면 INSERT
        # - session_id가 이미 있으면 UPDATE
        #
        # 이런 방식을 upsert라고 부릅니다.
        # upsert = update + insert
        conn.excute(
            """
            INSERT INTO chat_sessions (session_id, chat_history)
            VALUES (?, ?)
            ON CONFLICT(session_id)
            DO UPDATE SET chat_history = excluded.chat_history
            """,
            (session_id, history_json),
        )

        # commit()은 지금까지 실행한 DB 변경사항을 실제로 저장하라는 뜻입니다.
        #
        # INSERT나 UPDATE 같은 변경 작업은 commit()을 해야 DB 파일에 확정됩니다.
        #
        # commit()을 하지 않으면:
        # - 변경사항이 반영되지 않을 수 있습니다.
        # - 프로그램 종료나 오류 상황에서 저장이 취소될 수 있습니다.
        conn.commit()


def reset_chat_history(session_id: str) -> None:
    """
    특정 session_id의 대화 이력을 삭제합니다.
    
    server_managed 방식에서 대화 리셋 기준:
    - 사용자가 새 대화 버튼을 누른 경우
    - 특정 session_id의 대화방을 삭제하는 경우
    - 테스트 중 해당 세션을 초기화하고 싶은 경우
    """

    init_chat_history_db()

    with sqlite3.connect(DB_PATH) as conn:
        conn.excute(
            "DELETE FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        )
        conn.commit()
