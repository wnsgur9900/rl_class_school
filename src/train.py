"""학습 — 가상 매장(env.py)에서 PPO로 정책을 학습한다 (4 설계 카드: PPO · CPU).

학습 기간: 1 매장 정의의 학습 기간(2016~2024년)에서 시작일·고객을 무작위로 골라 에피소드를 반복.
출력
- results/models/<버전>/model.zip           : 학습된 정책
- results/curves/learning_curve_<버전>.png / .csv  : 학습 곡선 (가로 학습 스텝, 세로 에피소드 평균 보상)
- results/experience/experience_sample_<버전>.csv      : 학습 중 처음 1,000스텝의 경험 (상태 · 행동 · 보상 · 다음 상태)

실행: uv run python src/train.py --trial      (1분 안쪽 시험 학습: 오류 없이 도는지, 스텝당 시간)
      uv run python src/train.py              (본 학습)
"""
import argparse
import time
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

import config as C
import reward as R
from env import LaunPotEnv, OBS_NAMES, load_factors, load_traits
from paths import CURVES, EXPERIENCE, RESULTS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# 학습 설정  # 가정 (4-6에서 시험 학습으로 정함)
TOTAL_TIMESTEPS = 4_000_000      # 본 학습 스텝 수 — 3회차 v2 진단 '학습 부족'으로 200만 → 400만 (이 PC 약 11분)
TRIAL_TIMESTEPS = 30_000         # 시험 학습
N_ENVS = 8                       # 동시에 돌리는 가상 매장 수
PPO_KWARGS = dict(n_steps=1024, batch_size=1024, n_epochs=5, gamma=0.99, learning_rate=3e-4, ent_coef=0.01)
SAMPLE_STEPS = 1_000


class ExperienceRecorder(gym.Wrapper):
    """첫 번째 가상 매장의 처음 1,000스텝 경험을 모아 둔다."""

    def __init__(self, env, store: list):
        super().__init__(env)
        self.store, self.obs = store, None

    def reset(self, **kw):
        self.obs, info = self.env.reset(**kw)
        return self.obs, info

    def step(self, action):
        nxt, rew, term, trunc, info = self.env.step(action)
        if len(self.store) < SAMPLE_STEPS:
            row = {"step": len(self.store) + 1, "date": info["date"], "customer_id": info["customer_id"]}
            row |= {f"s_{n}": round(float(v), 4) for n, v in zip(OBS_NAMES, self.obs)}
            row |= {"action": int(action), "reward": round(rew, 4), "terminated": term, "truncated": trunc}
            row |= {f"next_{n}": round(float(v), 4) for n, v in zip(OBS_NAMES, nxt)}
            self.store.append(row)
        self.obs = nxt
        return nxt, rew, term, trunc, info


PARTS = getattr(R, "PARTS", ())          # 보상 항목 이름 (v2부터)
EPISODE_KEYS = ("ep_profit", "ep_opt_out") + tuple(f"ep_part_{k}" for k in PARTS)


class EpisodeStats(gym.Wrapper):
    """한 판(고객 1명 × 90일)이 끝날 때 성공 척도(순이익)·수신거부·보상 항목별 합계를 info에 넣는다
    → VecMonitor가 학습 곡선 파일에 함께 기록 (보상 점수와 성공 척도를 두 선으로 그리기 위해)."""

    def __init__(self, env):
        super().__init__(env)
        self.sums = {}

    def reset(self, **kw):
        self.sums = dict.fromkeys(EPISODE_KEYS, 0.0)
        return self.env.reset(**kw)

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self.sums["ep_profit"] += info["profit"]
        self.sums["ep_opt_out"] += info["opted_out"]
        if PARTS:
            for k, v in R.reward_parts(info).items():
                self.sums[f"ep_part_{k}"] += v
        if term or trunc:
            info.update(self.sums)
        return obs, rew, term, trunc, info


def plot_curve(monitor_csv: Path, version: str) -> pd.DataFrame:
    m = pd.read_csv(monitor_csv, skiprows=1)
    m["timesteps"] = m["l"].cumsum()
    window = max(50, len(m) // 100)
    roll = lambda s: s.rolling(window, min_periods=1).mean()
    curve = pd.DataFrame({"timesteps": m["timesteps"], "episode_return": m["r"], "return_mean": roll(m["r"])})
    if "ep_profit" in m:
        curve["episode_profit"] = m["ep_profit"]
        curve["profit_mean"] = roll(m["ep_profit"])            # 성공 척도: 한 판 순이익(원)
        curve["opt_out_rate"] = roll(m["ep_opt_out"])          # 그 판에서 수신거부한 비율
        for k in PARTS:
            curve[f"part_{k}"] = m[f"ep_part_{k}"]
    curve.to_csv(CURVES / f"learning_curve_{version}.csv", index=False)
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curve["timesteps"], curve["return_mean"], color="#2563eb", label="보상 (점)")
    ax.set_xlabel("학습 스텝")
    ax.set_ylabel(f"에피소드 평균 보상 (최근 {window}판, 1점 = {R.REWARD_SCALE:,.0f}원)")
    if "profit_mean" in curve:
        ax2 = ax.twinx()
        ax2.plot(curve["timesteps"], curve["profit_mean"], color="#64748b", ls="--", label="성공 척도: 순이익 (원)")
        ax2.set_ylabel("에피소드 평균 순이익 (원)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower right")
    ax.set_title(f"학습 곡선 — 보상 {version} (고객 1명 × 90일)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CURVES / f"learning_curve_{version}.png", dpi=120)
    plt.close(fig)
    return curve


def train(version: str, timesteps: int, trial: bool) -> dict:
    for d in (CURVES, EXPERIENCE):
        d.mkdir(parents=True, exist_ok=True)
    traits, factors = load_traits(), load_factors()
    store: list = []
    fns = [(lambda i=i: EpisodeStats(ExperienceRecorder(LaunPotEnv(traits, factors), store) if i == 0 else LaunPotEnv(traits, factors)))
           for i in range(N_ENVS)]
    tag = f"{version}_trial" if trial else version
    monitor = CURVES / f"train_{tag}"
    venv = VecMonitor(DummyVecEnv(fns), filename=str(monitor), info_keywords=EPISODE_KEYS)
    venv.seed(C.TRAIN_SEED)
    model = PPO("MlpPolicy", venv, device="cpu", seed=C.TRAIN_SEED, verbose=0, **PPO_KWARGS)
    t = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    sec = time.time() - t
    venv.close()   # 학습 기록 파일(monitor.csv)을 닫는다
    out = {"steps": timesteps, "seconds": sec, "ms_per_step": sec / timesteps * 1000}
    if trial:
        Path(f"{monitor}.monitor.csv").unlink(missing_ok=True)
        return out
    (RESULTS / "models" / version).mkdir(parents=True, exist_ok=True)
    model.save(RESULTS / "models" / version / "model.zip")
    pd.DataFrame(store).to_csv(EXPERIENCE / f"experience_sample_{version}.csv", index=False, encoding="utf-8-sig")
    curve = plot_curve(Path(f"{monitor}.monitor.csv"), version)
    n = max(1, len(curve) // 10)
    out |= {"episodes": len(curve), "first_return": curve["episode_return"].head(n).mean(),
            "last_return": curve["episode_return"].tail(n).mean()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default=R.VERSION)
    ap.add_argument("--trial", action="store_true")
    ap.add_argument("--timesteps", type=int)
    a = ap.parse_args()
    steps = a.timesteps or (TRIAL_TIMESTEPS if a.trial else TOTAL_TIMESTEPS)
    r = train(a.version, steps, a.trial)
    print(f"{'시험 학습' if a.trial else '본 학습'} {r['steps']:,}스텝 · {r['seconds']:.0f}초 · 스텝당 {r['ms_per_step']:.3f}ms")
    if not a.trial:
        print(f"에피소드 {r['episodes']:,}판 · 평균 보상 처음 10% {r['first_return']:.2f}점 → 마지막 10% {r['last_return']:.2f}점")
        print(f"저장: results/models/{a.version}/model.zip, results/curves/learning_curve_{a.version}.png/.csv, "
              f"results/experience/experience_sample_{a.version}.csv")


if __name__ == "__main__":
    main()
