"""보상 v3 — 피로 위험 구간 벌점 + 수신거부 벌점 (3회차 2-1, 두 번째 수정).
v1 · v2는 reward_v1.py · reward_v2.py에 그대로 보관.

무엇을 왜 바꿨나 (쉬운 말):
- v2(위험 구간 벌점)는 수신거부 86.8명으로 허용 기준(10명 이하)을 넘어 되돌렸다. 진단(results/diagnosis_v2.md):
  학습 초반에는 수신거부 9%의 좋은 행동을 찾았다가, 그 뒤 수신거부 87%로 옮겨 갔다 — 보상의 허점 때문.
  · 피로 80 이상인 유저는 알림을 안 보내도 75 아래로 내려오기까지 며칠 동안 매일 −3점
  · 그런데 알림을 끄면 판이 끝나 벌점도 끝나고, 수신거부 자체에는 벌점이 없다
  → '리워드를 계속 보내 이익을 챙기고 빨리 끄게 하기'가 유리했다.
- v3는 v2에 항목을 하나만 더한다: 알림을 끈 날 −20점. 끄게 해서 벌점을 끝내는 꼼수가 더 이상 이득이 아니다.
- 크기 근거 (가정): '현재 운영' 기준 유저 1명의 90일 순이익 ≈ 18,278원 ≈ 18점 → 20점.
  언제 끄든 같은 벌점이라 85일째에 떠나게 해도 10일째와 똑같이 손해 → 막판 몰아주기도 함께 막는다.
- v3 하루 보상 = 단기(그날 순이익 ÷ 1,000) + 장기(피로 위험 구간 −3점) + 장기(수신거부 −20점)
- 성공 척도(90일 순이익) 계산은 그대로다. 벌점은 보상에만 더한다.
- 걱정되는 점: 벌점이 두 개라 너무 조심해서 순이익이 '현재 운영'(183만 원) 가까이로 낮아질 수 있다.
"""

VERSION = "v3"
REWARD_SCALE = 1000.0      # 단기 항목: 1,000원 = 1점 (v1 · v2와 같음)
DANGER_PENALTY = 3.0       # 장기 항목: 알림 받은 뒤 피로도가 위험선 이상인 날마다 −3점 (v2와 같음)  # 가정
OPT_OUT_PENALTY = 20.0     # 장기 항목: 알림을 끈 날 −20점 (v3에서 추가)  # 가정
PARTS = ("profit", "fatigue_danger", "opt_out")   # 보상 항목 이름 (학습 곡선·평가에 항목별로 기록)
SUMMARY = "그날 순이익 ÷ 1,000 − 3점 × [피로도 75 이상] − 20점 × [알림을 끔]"   # 화면에 보여 줄 한 줄 식

_danger = None


def _fatigue_danger() -> float:
    """위험선은 env.py의 반응 규칙 3 값(FATIGUE_DANGER = 75)을 그대로 쓴다 (숫자를 두 곳에 적지 않게)."""
    global _danger
    if _danger is None:
        from env import FATIGUE_DANGER
        _danger = FATIGUE_DANGER
    return _danger


def reward_parts(info: dict) -> dict:
    return {
        "profit": info["profit"] / REWARD_SCALE,
        "fatigue_danger": -DANGER_PENALTY if info["fatigue_after"] >= _fatigue_danger() else 0.0,
        "opt_out": -OPT_OUT_PENALTY if info["opted_out"] else 0.0,
    }


def compute_reward(info: dict) -> float:
    return sum(reward_parts(info).values())
