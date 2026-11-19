"""가상 매장 — 론팟 마포 상권 푸시 알림 환경 (Gymnasium 형식).

한 스텝 = 하루 1번 결정(18:00), 한 에피소드 = 고객 한 명의 90일.
고객마다 에피소드를 따로 진행하고 정책 하나를 같이 쓴다. 성공 척도는 100명 결과를 합산한다.
명세: project_spec.md 1 매장 정의 · 2 반응 규칙 · 3 고객 성향 항목 · 4 설계 카드
"""
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

import config as C
import reward as R

# ---------------------------------------------------------------------------
# 반응 규칙 가정값 (project_spec.md 2절, 규칙 번호를 주석으로)  # 모두 가정
# 고객마다 다른 값은 data/customer_traits.csv (3절), 여기엔 모두에게 같은 값만 둔다.
# ---------------------------------------------------------------------------
GENERAL_PUSH_MULTIPLIER = 2.5        # 규칙1: 일반 푸시 날 제보 확률 = 기본 제보 성향 × 2.5 (8% → 20%)
REWARD_PROB_BASE = 0.25              # 규칙2: 리워드 제보 확률 = 0.25 + 0.40 × 리워드 민감도 (평균 45%)
REWARD_PROB_SLOPE = 0.40             # 규칙2
FATIGUE_PER_PUSH = 15.0              # 규칙3: 발송 1회 피로도 +15 × 알림 피로 민감도
FATIGUE_RECOVERY = 5.0               # 규칙3: 미발송 날 피로도 −5
FATIGUE_DANGER = 75.0                # 규칙3: 피로도 75 이상이면
OPT_OUT_PROB = 0.20                  # 규칙3: 매일 20% 확률로 수신거부(영구 이탈)
DEPENDENCY_PER_REWARD = 20.0         # 규칙4: 리워드 1회 의존도 +20 × (0.5 + 리워드 민감도)
DEPENDENCY_RECOVERY = 5.0            # 규칙4: 리워드 없는 날 의존도 −5
DEPENDENCY_DANGER = 60.0             # 규칙4: 의존도 60 이상이면
DEPENDENCY_PENALTY = 0.5             # 규칙4: 일반 알림·미발송 제보 확률 × 0.5
RAIN_MM = 1.0                        # 규칙5: 비 = 강수량 1mm 이상
DEMAND_UP = 1.4                      # 규칙5: 비·공휴일·주말 × 1.4 (겹쳐도 1회)
COLD_TEMP_MIN = -5.0                 # 규칙5: 한파 = 일 최저기온 −5℃ 이하
COLD_MULTIPLIER = 0.7                # 규칙5: 한파 × 0.7
MONTH_END_REWARD_MULTIPLIER = 1.2    # 규칙5: 월말 리워드 제보 확률만 × 1.2
CHANCE_LOW, CHANCE_HIGH = 0.85, 1.15 # 규칙6: 매일 유저마다 × 0.85~1.15 균등 무작위
LEVEL_MAX = 100.0                    # 피로도·의존도 범위 0~100

OBS_NAMES = [
    "report_propensity", "reward_sensitivity", "fatigue_sensitivity",
    "fatigue", "dependency",
    "is_rain", "is_holiday_or_weekend", "is_cold", "is_month_end", "day_of_week",
    "days_left",
]

DATA_DIR = Path(__file__).parent / "data"


def load_traits() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "customer_traits.csv", encoding="utf-8-sig").set_index("customer_id")


def load_factors() -> pd.DataFrame:
    f = pd.read_csv(DATA_DIR / "external_factors.csv", encoding="utf-8-sig")
    f["is_rain"] = (f["precipitation_mm"] >= RAIN_MM).astype(int)
    f["is_holiday_or_weekend"] = ((f["is_holiday"] == 1) | (f["is_weekend"] == 1)).astype(int)
    f["is_cold"] = (f["temp_min"] <= COLD_TEMP_MIN).astype(int)
    return f.set_index("date")


class LaunPotEnv(gym.Env):
    """고객 한 명의 90일. reset(options={"customer_id", "start_date"})로 평가 조건을 지정한다."""

    metadata = {"render_modes": []}

    def __init__(self, traits: pd.DataFrame | None = None, factors: pd.DataFrame | None = None):
        super().__init__()
        self.traits = load_traits() if traits is None else traits
        self.factors = load_factors() if factors is None else factors
        self.dates = list(self.factors.index)
        self.date_pos = {d: i for i, d in enumerate(self.dates)}
        # 스텝마다 표를 뒤지지 않도록 숫자 배열로 미리 바꿔 둔다 (속도용, 값은 같음)
        self.fx = {c: self.factors[c].to_numpy(dtype=np.float64)
                   for c in ["is_rain", "is_holiday_or_weekend", "is_cold", "is_month_end", "day_of_week"]}
        self.trait_rows = {int(cid): {k: float(v) for k, v in row.items() if k in
                                      ("report_propensity", "reward_sensitivity", "fatigue_sensitivity")}
                           for cid, row in self.traits.iterrows()}
        last_start = self.date_pos[C.TRAIN_END] - C.EPISODE_LENGTH_DAYS + 1
        self.train_start_range = (self.date_pos[C.TRAIN_START], last_start)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(len(OBS_NAMES),), dtype=np.float32)
        self.action_space = spaces.Discrete(C.NUM_ACTIONS)

    # --- 관측: 결정할 때(18:00) 알 수 있는 값만, 0~1로 맞춤 ---
    def _obs(self) -> np.ndarray:
        t, fx, i = self.trait, self.fx, self.pos
        return np.array([
            t["report_propensity"] / 0.20,
            t["reward_sensitivity"],
            t["fatigue_sensitivity"] - 0.5,
            self.fatigue / LEVEL_MAX,
            self.dependency / LEVEL_MAX,
            fx["is_rain"][i], fx["is_holiday_or_weekend"][i], fx["is_cold"][i], fx["is_month_end"][i],
            fx["day_of_week"][i] / 6.0,
            (C.EPISODE_LENGTH_DAYS - self.day) / C.EPISODE_LENGTH_DAYS,
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        cid = options.get("customer_id")
        self.customer_id = int(cid) if cid is not None else int(self.np_random.choice(self.traits.index))
        self.trait = self.trait_rows[self.customer_id]
        start = options.get("start_date")
        self.start_pos = self.date_pos[start] if start else int(self.np_random.integers(*self.train_start_range, endpoint=True))
        self.day, self.pos = 0, self.start_pos
        self.fatigue, self.dependency = 0.0, 0.0
        return self._obs(), {}

    def report_probability(self, action: int) -> float:
        """규칙 1·2·4·5: 오늘 행동에 따른 제보 확률 (우연 배수 전)."""
        t, i = self.trait, self.pos
        f = {c: self.fx[c][i] for c in ("is_rain", "is_holiday_or_weekend", "is_cold", "is_month_end")}
        if action == C.ACTION_REWARD_PUSH:
            p = REWARD_PROB_BASE + REWARD_PROB_SLOPE * t["reward_sensitivity"]           # 규칙2
            if f["is_month_end"]:
                p *= MONTH_END_REWARD_MULTIPLIER                                        # 규칙5 월말
        else:
            p = t["report_propensity"]                                                  # 규칙1 미발송
            if action == C.ACTION_GENERAL_PUSH:
                p *= GENERAL_PUSH_MULTIPLIER                                            # 규칙1 일반
            if self.dependency >= DEPENDENCY_DANGER:
                p *= DEPENDENCY_PENALTY                                                 # 규칙4
        if f["is_rain"] or f["is_holiday_or_weekend"]:
            p *= DEMAND_UP                                                              # 규칙5
        if f["is_cold"]:
            p *= COLD_MULTIPLIER                                                        # 규칙5
        return p

    def step(self, action):
        action = int(action)
        state = {"fatigue": self.fatigue, "dependency": self.dependency}
        p = self.report_probability(action) * self.np_random.uniform(CHANCE_LOW, CHANCE_HIGH)  # 규칙6
        p = float(np.clip(p, 0.0, 1.0))
        reported = bool(self.np_random.random() < p)
        value = C.DATA_VALUE_PER_REPORT if reported else 0
        cost = C.REWARD_COST_PER_REPORT if (reported and action == C.ACTION_REWARD_PUSH) else 0

        pushed = action in (C.ACTION_GENERAL_PUSH, C.ACTION_REWARD_PUSH)
        if pushed:                                                                      # 규칙3
            self.fatigue += FATIGUE_PER_PUSH * self.trait["fatigue_sensitivity"]
        else:
            self.fatigue -= FATIGUE_RECOVERY
        if action == C.ACTION_REWARD_PUSH:                                              # 규칙4
            self.dependency += DEPENDENCY_PER_REWARD * (0.5 + self.trait["reward_sensitivity"])
        else:
            self.dependency -= DEPENDENCY_RECOVERY
        self.fatigue = float(np.clip(self.fatigue, 0.0, LEVEL_MAX))
        self.dependency = float(np.clip(self.dependency, 0.0, LEVEL_MAX))
        opted_out = bool(self.fatigue >= FATIGUE_DANGER and self.np_random.random() < OPT_OUT_PROB)  # 규칙3

        info = {
            "date": self.dates[self.pos], "customer_id": self.customer_id, "action": action,
            "fatigue_before": state["fatigue"], "dependency_before": state["dependency"],
            "report_prob": p, "reported": int(reported), "opted_out": int(opted_out),
            "value": value, "reward_cost": cost, "profit": value - cost,
            "fatigue_after": self.fatigue, "dependency_after": self.dependency,
        }
        rew = R.compute_reward(info)
        self.day += 1
        self.pos += 1
        terminated = opted_out
        truncated = self.day >= C.EPISODE_LENGTH_DAYS
        obs = self._obs() if not (terminated or truncated) else np.zeros(len(OBS_NAMES), dtype=np.float32)
        return obs, float(rew), terminated, truncated, info
