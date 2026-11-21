"""폴더 위치를 한곳에 모은 파일.

어느 폴더에서 실행해도 같은 곳에 읽고 쓰도록, 불러오는 순간 프로젝트 맨 위 폴더로 이동한다.
기록 파일에 내 PC의 전체 경로가 남지 않게 경로는 맨 위 폴더 기준으로 적는다.
"""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

DATA = Path("data")
REPORTS = Path("docs/reports")            # 사람이 읽는 결과 글 (점검 · 평가 · 진단 · 정책 해석)
RESULTS = Path("results")
LOGS = RESULTS / "logs"                   # 규칙별 날짜 · 고객 단위 기록
CURVES = RESULTS / "curves"               # 학습 곡선과 학습 기록
DIAGNOSIS = RESULTS / "diagnosis"         # 진단 그림과 화면용 진단 숫자
EXPERIENCE = RESULTS / "experience"       # 학습 경험 표본
PUBLIC_DATA = Path("public/data")         # 화면이 읽는 재생 데이터
REWARD_HISTORY = Path("src/reward_history")   # 앞 보상 버전 보관
