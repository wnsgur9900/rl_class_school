"""보상 v2 — 피로 위험 구간 벌점 (3회차 2-1). v1은 reward_v1.py에 그대로 보관.

무엇을 왜 바꿨나 (쉬운 말):
- v1(하루 보상 = 그날 순이익 ÷ 1,000)으로 학습한 AI는 순이익은 가장 높았지만 수신거부 90.8명으로
  허용 기준(10명 이하)을 크게 넘었다. 진단(docs/reports/diagnosis_v1.md)에서 찾은 원인:
  · 피로도 75 이상인 유저(매일 20% 확률로 알림을 끌 수 있는 유저)에게 98% 리워드를 보냄 — 1,328번
  · 판 끝(76~90일)에 리워드를 72%로 몰아 보냄 — 끝나기 직전 이탈은 잃을 게 적다고 배움
- 반응 규칙 3에서 수신거부는 '알림을 받은 뒤 피로도가 75 이상'일 때만 생긴다.
  그래서 v2는 위험 구간에 들어가는 것 자체에 벌점을 준다 (자료의 '위험 구간에서만 벌점').
- v2 하루 보상 = 단기 항목 + 장기 항목
    단기: 그날 순이익 ÷ 1,000            (v1과 같음 — 제보 1건 +1.5점, 리워드 제보 +1.2점)
    장기: 알림 받은 뒤 피로도가 75 이상인 날 −3점
- 벌점 3점의 근거 (가정): 위험 구간에 하루 있으면 20% 확률로 그 유저의 남은 이익(약 1.5만 원)을 잃는다
  → 약 3,000원 = 3점. 리워드 알림 하루 기대 이익(약 0.5점)보다 훨씬 커서 위험 구간으로 미는 알림을 피하게 된다.
- 성공 척도(90일 순이익) 계산은 그대로다. 벌점은 보상에만 더한다.
- 걱정되는 점: 이미 위험 구간 깊이 들어간 유저는 벌점이 며칠 이어지므로 '빨리 끄게 해서 벌점을 끝내는'
  쪽을 배울 수 있다 (판이 끝나면 벌점도 끝나서). 결과의 수신거부 수로 확인한다.
"""

VERSION = "v2"
REWARD_SCALE = 1000.0      # 단기 항목: 1,000원 = 1점 (v1과 같음)
DANGER_PENALTY = 3.0       # 장기 항목: 알림 받은 뒤 피로도가 위험선 이상인 날마다 −3점  # 가정
PARTS = ("profit", "fatigue_danger")   # 보상 항목 이름 (학습 곡선·평가에 항목별로 기록)
SUMMARY = "그날 순이익 ÷ 1,000 − 3점 × [알림 받은 뒤 피로도 75 이상]"   # 화면에 보여 줄 한 줄 식

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
    }


def compute_reward(info: dict) -> float:
    return sum(reward_parts(info).values())
