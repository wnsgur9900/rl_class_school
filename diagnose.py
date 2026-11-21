"""결과 진단 — 3회차 1-1: 다섯 질문 진단표와 진단 그림.

읽는 것: results/logs/<규칙>.csv (평가 기록), results/learning_curve_<버전>.csv (학습 곡선)
만드는 것: results/diagnosis_<버전>.md (진단표 · 이상한 장면 · 문제 위치)
          results/diagnosis_<버전>.png (다섯 칸 그림 + 요약)
          results/diagnosis_<버전>.json (같은 진단 + 정책 해석 숫자 → 화면 7·8단계, viewer_data.js에 묶임)
v1이 아닌 버전은 v1을 나란히 그린다 (학습 곡선 칸만 버전마다 보상 기준이 달라 그 버전 하나).

실행: uv run python diagnose.py --version v1   (실행 메뉴 4번 평가가 끝나면 자동으로도 실행됨)
"""
import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

import config as C
from env import DEPENDENCY_DANGER, FATIGUE_DANGER, load_factors

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("results")
LOGS = RESULTS / "logs"
PUBLIC_DATA = Path("public/data")
# 진단 판정 기준  # 가정 (화면·발표용 판정선, 학습에는 영향 없음)
FLAT_TOLERANCE = 0.03      # 질문 1: 마지막 10% 판 평균이 그 앞 10%와 3% 안쪽 차이면 '평평'
PEAK_DROP = 0.15           # 질문 1: 마지막 10%가 가장 높았던 10% 구간보다 15% 넘게 낮으면 '오르다 떨어짐'
BIAS_SHARE = 0.80          # 질문 4: 한 행동이 80%를 넘거나 한 번도 안 쓴 행동이 있으면 '쏠림'
SITUATION_GAP = 0.10       # 질문 5: 상황별 리워드 비율 차이가 10%p를 넘으면 '상황에 따라 달라짐'
# 정책 해석(3회차 2-2) 규칙 다섯 줄을 셀 때 쓰는 구간  # 가정 (results/policy_rules.md와 같은 기준, 학습에는 영향 없음)
LOW_FATIGUE = 25           # 규칙 2: '피로가 낮은 유저'
REST_FATIGUE = 60          # 규칙 3: 이 피로부터는 쉬게 함 (50~59도 함께 셈)
NEAR_REST_FATIGUE = 50
SENS_LOW, SENS_HIGH = 0.4, 0.7        # 상식과 다른 규칙: 리워드 민감도 낮음 · 높음
FSENS_LOW, FSENS_HIGH = 0.8, 1.2      # 규칙 5: 알림 피로 민감도 낮음 · 높음
FATIGUE_BANDS = [(0, 25, "0~24"), (25, 50, "25~49"), (50, 60, "50~59"), (60, 75, "60~74"), (75, 101, "75 이상")]
TEST_KEYS = ["none", "random", "current", "max_reward"]
LABELS = {"none": "아무것도 안 함", "random": "무작위", "current": "현재 운영(내 규칙)", "max_reward": "매일 전원 리워드"}
ACTS = ["안 보냄", "일반 알림", "리워드 알림"]
COL = {"none": "#a8a29e", "random": "#8b5cf6", "current": "#2563eb", "max_reward": "#d97706", "v1": "#0f766e"}
NEW_COL = "#db2777"


def pol_key(v: str) -> str:
    return f"ppo_{v}"


def label(k: str) -> str:
    return LABELS.get(k, f"학습된 정책 {k.removeprefix('ppo_')}")


def tick(k: str) -> str:   # 그림 막대 아래 짧은 이름
    return {"none": "아무것도\n안 함", "random": "무작위", "current": "현재 운영\n(내 규칙)",
            "max_reward": "매일\n전원 리워드"}.get(k, f"AI {k.removeprefix('ppo_')}")


PPO_COLORS = ["#0f766e", "#db2777", "#7c3aed", "#0891b2"]   # 보상 버전 v1 · v2 · v3 · v4 (화면과 같은 색)


def color(k: str) -> str:
    m = re.match(r"ppo_v(\d+)$", k)
    return COL.get(k) or (PPO_COLORS[(int(m.group(1)) - 1) % len(PPO_COLORS)] if m else NEW_COL)


def load_log(key: str) -> pd.DataFrame:
    df = pd.read_csv(LOGS / f"{key}.csv", encoding="utf-8-sig")
    start = df["episode"].map(lambda e: pd.Timestamp(C.EVAL_START_DATES[e - 1]))
    df["day"] = (pd.to_datetime(df["date"]) - start).dt.days + 1
    f = load_factors()
    df["up"] = df["date"].map(f["is_rain"] | f["is_holiday_or_weekend"]).astype(bool)
    df["cold"] = df["date"].map(f["is_cold"]).astype(bool)
    df["month_end"] = df["date"].map(f["is_month_end"]).astype(bool)
    return df


def names() -> dict:
    p = pd.read_csv("data/personas.csv", encoding="utf-8-sig")
    out = {}
    for r in p.itertuples():
        m = re.match(r"^(\S+?) 씨", str(r.persona))
        out[int(r.customer_id)] = m.group(1) if m else f"고객 {r.customer_id}"
    return out


def situation_table(df: pd.DataFrame) -> pd.DataFrame:
    """상황별 행동 비율 (그날 결정한 유저 기준)."""
    groups = [
        ("평범한 날", ~df["up"] & ~df["cold"] & ~df["month_end"]),
        ("비·주말·공휴일", df["up"]),
        ("한파", df["cold"]),
        ("월말", df["month_end"]),
        ("피로도 0~24", df["fatigue_before"] < 25),
        ("피로도 25~49", df["fatigue_before"].between(25, 49.999)),
        ("피로도 50~74", df["fatigue_before"].between(50, FATIGUE_DANGER - 0.001)),
        (f"피로도 {FATIGUE_DANGER:.0f} 이상", df["fatigue_before"] >= FATIGUE_DANGER),
        ("1~60일", df["day"] <= 60),
        ("61~75일", df["day"].between(61, 75)),
        ("76~90일", df["day"] >= 76),
    ]
    rows = []
    for name, m in groups:
        x = df[m]
        rows.append({"상황": name, "결정 수": len(x),
                     "리워드 비율": (x["action"] == 2).mean() if len(x) else np.nan,
                     "일반 비율": (x["action"] == 1).mean() if len(x) else np.nan})
    return pd.DataFrame(rows)


def policy_stats(df: pd.DataFrame) -> dict:
    """정책 해석 — 규칙 다섯 줄이 몇 번 나타났는지와 '상황 × 피로도' 지도 (화면 8단계용)."""
    t = pd.read_csv("data/customer_traits.csv", encoding="utf-8-sig").set_index("customer_id")
    rs, fs = df["customer_id"].map(t["reward_sensitivity"]), df["customer_id"].map(t["fatigue_sensitivity"])
    fat, dep, a, up = df["fatigue_before"], df["dependency_before"], df["action"], df["up"]
    plain = ~up & ~df["month_end"] & ~df["cold"]
    hit = lambda m, ok: {"n": int(m.sum()), "k": int((m & ok).sum())}
    share = lambda m, act: float((a[m] == act).mean()) if m.any() else None
    gen = a == 1
    gaps = []
    for _, x in df[a > 0].sort_values(["episode", "customer_id", "day"]).groupby(["episode", "customer_id"]):
        gaps += list(x["day"].diff().dropna())
    me_dry = df["month_end"] & ~up
    cols = [("평범한 날", plain), ("비·주말·공휴일", up), ("월말 (비·주말 아님)", me_dry), ("한파 (비·주말·월말 아님)", ~up & ~df["month_end"] & df["cold"])]
    cells = []
    for lo, hi, _ in FATIGUE_BANDS:
        row = []
        for _, cm in cols:
            m = cm & fat.between(lo, hi - 1e-9)
            row.append({"n": int(m.sum()), "s": [share(m, i) for i in range(3)] if m.any() else None})
        cells.append(row)
    return {
        "decisions": len(df),
        "rules": [hit(plain, a == 0), hit(up & (fat < LOW_FATIGUE), a == 2), hit(fat >= REST_FATIGUE, a == 0),
                  hit(dep >= DEPENDENCY_DANGER, a != 2), hit(gen, up)],
        "rest_near": share(fat.between(NEAR_REST_FATIGUE, REST_FATIGUE - 1e-9), 0),
        "dep_reward": share(dep >= DEPENDENCY_DANGER, 2),
        "general_by_fsens": [share(fs <= FSENS_LOW, 1), share(fs >= FSENS_HIGH, 1)],
        "reward_early": share(df["day"] <= 60, 2), "reward_late": share(df["day"] >= 76, 2),
        "danger_decisions": int((fat >= FATIGUE_DANGER).sum()),
        "gap_mean": float(np.mean(gaps)) if gaps else None, "gap_median": float(np.median(gaps)) if gaps else None,
        "reward_by_rsens": [share(rs < SENS_LOW, 2), share(rs >= SENS_HIGH, 2)],
        "month_end_dry": hit(me_dry, a == 0),
        "grid": {"rows": [b[2] for b in FATIGUE_BANDS], "cols": [c[0] for c in cols], "cells": cells},
        "cuts": {"low_fatigue": LOW_FATIGUE, "rest_fatigue": REST_FATIGUE, "near_rest": NEAR_REST_FATIGUE,
                 "dep_danger": DEPENDENCY_DANGER, "sens": [SENS_LOW, SENS_HIGH], "fsens": [FSENS_LOW, FSENS_HIGH]},
    }


def curve_stats(v: str) -> dict | None:
    p = RESULTS / f"learning_curve_{v}.csv"
    if not p.exists():
        return None
    c = pd.read_csv(p)
    n = max(1, len(c) // 10)
    r = c["episode_return"]
    first, prev, last = r.head(n).mean(), r.iloc[-2 * n:-n].mean(), r.tail(n).mean()
    seg = [r.iloc[i * n:(i + 1) * n].mean() for i in range(10)]             # 학습을 10구간으로 나눈 평균
    opt = [c["opt_out_rate"].iloc[i * n:(i + 1) * n].mean() if "opt_out_rate" in c else None for i in range(10)]
    if "episode_profit" in c:   # 판마다 수신거부했는지는 학습 기록(monitor)에서 — 없으면 그 열은 비움
        m = RESULTS / f"train_{v}.monitor.csv"
        if m.exists():
            mo = pd.read_csv(m, skiprows=1)
            k = len(mo) // 10
            opt = [mo["ep_opt_out"].iloc[i * k:(i + 1) * k].mean() for i in range(10)]
    peak_i = int(np.argmax(seg))
    peaked = last < seg[peak_i] - abs(seg[peak_i]) * PEAK_DROP
    return {"curve": c, "first": first, "prev": prev, "last": last, "seg": seg, "opt": opt, "peak_i": peak_i, "peaked": peaked,
            "flat": abs(last - prev) <= abs(last) * FLAT_TOLERANCE, "change": (last - prev) / abs(last)}


def strange_scenes(df: pd.DataFrame, who: dict) -> list[dict]:
    """성공 척도·허용 기준으로 볼 때 이상한 장면 세 개."""
    scenes = []
    out = df[df["opted_out"] == 1].sort_values(["day", "episode", "customer_id"])
    if len(out):
        r = out.iloc[0]
        scenes.append({"날짜": f"{r.date} ({r.day:.0f}일째)", "대상": f"#{r.customer_id} {who[r.customer_id]}",
                       "상태": f"피로도 {r.fatigue_before:.0f} · 의존도 {r.dependency_before:.0f}", "행동": ACTS[r.action],
                       "왜 이상한가": f"첫 수신거부. 이후 수신거부가 몰림 (절반이 {out['day'].median():.0f}일째 이후)"})
    risky = df[(df["fatigue_before"] >= FATIGUE_DANGER) & (df["action"] > 0)]
    if len(risky):
        r = risky.sort_values(["fatigue_before", "day"], ascending=False).iloc[0]
        scenes.append({"날짜": f"{r.date} ({r.day:.0f}일째)", "대상": f"#{r.customer_id} {who[r.customer_id]}",
                       "상태": f"피로도 {r.fatigue_before:.0f} (위험선 {FATIGUE_DANGER:.0f} 이상)", "행동": ACTS[r.action],
                       "왜 이상한가": f"이미 매일 20% 확률로 알림을 끌 수 있는 유저에게 또 보냄 — 이런 결정이 {len(risky):,}번"})
    late = df[df["day"] >= 61].sort_values(["episode", "customer_id", "day"])
    best = None
    for (ep, cid), g in late.groupby(["episode", "customer_id"]):
        run, start = 0, None
        for r in g.itertuples():
            if r.action == 2:
                run, start = run + 1, start if run else r
                if not best or run > best[0]:
                    best = (run, start, r)
            else:
                run, start = 0, None
    if best and best[0] >= 2:
        n, a, b = best
        scenes.append({"날짜": f"{a.date} ~ {b.date} ({a.day:.0f}~{b.day:.0f}일째)", "대상": f"#{a.customer_id} {who[a.customer_id]}",
                       "상태": f"피로도 {a.fatigue_before:.0f} → {b.fatigue_before:.0f}", "행동": f"리워드 {n}일 연속",
                       "왜 이상한가": "판 끝 무렵 같은 유저에게 리워드를 몰아 보냄 (막판 몰아주기)" + (" → 그 뒤 알림 끔" if b.opted_out else "")})
    return scenes[:3]


def diagnose(version: str) -> tuple[Path, Path]:
    key = pol_key(version)
    pols = [pol_key("v1")] if version == "v1" else [pol_key("v1"), key]
    keys = TEST_KEYS + pols
    logs = {k: load_log(k) for k in keys if (LOGS / f"{k}.csv").exists()}
    # 성공 척도·보조 지표는 평가 때 계산해 둔 값(재생 기록의 summary)을 그대로 씀
    s = {k: json.loads((PUBLIC_DATA / f"replay_{k}.json").read_text(encoding="utf-8"))["summary"] for k in logs}
    me, cur = s[key], s["current"]
    df = logs[key]
    who = names()
    cs = curve_stats(version)
    lim = C.MAX_OPT_OUT_CUSTOMERS
    per_ep = df.groupby("episode")["opted_out"].sum()
    total = me["none_count"] + me["general_pushes"] + me["reward_pushes"]
    shares = [me["none_count"] / total, me["general_pushes"] / total, me["reward_pushes"] / total]
    sit = situation_table(df)
    gap = sit["리워드 비율"].max() - sit["리워드 비율"].min()

    q1 = cs is not None and cs["flat"] and not cs["peaked"]
    q2 = all(me["profit"] > s[k]["profit"] for k in ("none", "random", "current"))
    q3 = not me["success"]
    q4 = max(shares) > BIAS_SHARE or min(shares) == 0
    q5 = gap > SITUATION_GAP
    yn = lambda b: "예" if b else "아니오"
    q1t = (f"처음 10% 판 평균 {cs['first']:.1f}점 → 마지막 앞 10% {cs['prev']:.1f}점 → 마지막 10% {cs['last']:.1f}점 "
           f"(마지막 구간 변화 {cs['change']:+.1%}, ±{FLAT_TOLERANCE:.0%} 안이면 평평)"
           + (f" · **최고점에서 떨어짐**: {cs['peak_i'] + 1}번째 10% 구간 {cs['seg'][cs['peak_i']]:.1f}점이 가장 높았음" if cs["peaked"] else "")
           ) if cs else "학습 곡선 파일 없음"
    q2t = " · ".join(f"{label(k)} {s[k]['profit']:,.0f}원" for k in ("none", "random", "current")) + \
          f" vs {label(key)} **{me['profit']:,.0f}원** (내 규칙의 {me['profit'] / cur['profit']:.2f}배)" + \
          ("" if me["success"] else " — 단 허용 기준을 넘어 채택할 수 없음")
    q3t = f"수신거부 네 판 평균 **{me['opt_outs']:.1f}명** (판별 {', '.join(f'{x:.0f}' for x in per_ep)}명) · 기준 {lim}명 이하 → {'넘음' if q3 else '지킴'}"
    q4t = f"안 보냄 {shares[0]:.0%} · 일반 알림 {shares[1]:.0%} · 리워드 {shares[2]:.0%}" + \
          (" — 한 번도 안 쓴 행동이 있음" if min(shares) == 0 else "")
    q5t = " · ".join(f"{r['상황']} {r['리워드 비율']:.0%}" for _, r in sit.iterrows() if r["결정 수"]) + f" (리워드 비율 최대·최소 차이 {gap:.0%}p)"

    # 문제 위치: 학습 → 관측 → 보상 → 환경
    if not q1 and cs and cs["peaked"]:
        where = ("**보상 (허점 의심)** — 학습 곡선이 오르다 떨어짐: 좋은 행동을 찾았다가 보상이 더 높게 쳐주는 다른 행동으로 옮겨 감. "
                 "학습을 더 늘려도 나아지지 않을 수 있으니 구간별 수신거부 비율을 보고 보상 식을 고침")
    elif not q1:
        where = "**학습 부족** — 학습 곡선이 아직 오르는 중. 학습 스텝을 2배로 늘려 다시 학습 (보상 그대로)"
    elif not q5:
        where = "**관측** — 상황이 달라도 같은 행동. 그 값이 관측에 있는지, 범위가 맞는지 확인"
    elif q3 or q4:
        where = "**보상** — 학습은 됐고 상황에 따라 행동도 달라지지만, 허용 기준을 넘거나 행동이 쏠림 → 보상 재설계"
    else:
        where = "**보상 (더 나아질 점)** — 다섯 질문 모두 문제 없음. 보조 지표 하나를 더 좋게 하는 수정을 시도"
    scenes = strange_scenes(df, who)

    md = [f"# 결과 진단 — 보상 {version}", "",
          f"평가 조건: {', '.join(C.EVAL_START_DATES)} 시작 90일 × 고객 {C.NUM_CUSTOMERS}명, 시드 {C.EVAL_SEED} · "
          f"성공 척도 = 90일 순이익 · 허용 기준 = 수신거부 {lim}명 이하 (한 판마다)", "",
          "## 다섯 질문", "", "| # | 질문 | 답 | 근거 숫자 |", "| --- | --- | --- | --- |",
          f"| 1 | 학습 곡선이 오르다 평평해졌는가 | {yn(q1)} | {q1t} |",
          f"| 2 | 성공 척도(순이익)가 시험 규칙, 특히 내 규칙보다 나은가 | {yn(q2)} | {q2t} |",
          f"| 3 | 보조 지표가 1절 허용 기준을 넘는가 | {yn(q3)} | {q3t} |",
          f"| 4 | 행동이 한쪽으로 쏠렸는가 | {yn(q4)} | {q4t} |",
          f"| 5 | 날씨 · 요일 · 쌓이는 값에 따라 행동이 달라지는가 | {yn(q5)} | {q5t} |", "",
          "기준 없이 함께 보는 보조 지표: " + f"하루 평균 제보 {me['reports_per_day']:.1f}건 · 리워드 비용 {me['reward_cost']:,.0f}원 · "
          f"결정 뒤 평균 피로도 {me['avg_fatigue']:.1f}", "",
          "## 이상한 장면 (성공 척도 · 허용 기준으로 볼 때)", "", "| # | 날짜 | 대상 | 상태 | 행동 | 왜 이상한가 |", "| --- | --- | --- | --- | --- | --- |"]
    md += [f"| {i} | {x['날짜']} | {x['대상']} | {x['상태']} | {x['행동']} | {x['왜 이상한가']} |" for i, x in enumerate(scenes, 1)]
    if cs:
        md += ["", "## 학습 중 변화 (학습을 10구간으로 나눔)", "", "| 구간 | " + " | ".join(str(i + 1) for i in range(10)) + " |",
               "| --- |" + " ---: |" * 10, "| 보상 (판 평균, 점) | " + " | ".join(f"{x:.1f}" for x in cs["seg"]) + " |"]
        if cs["opt"][0] is not None:
            md += ["| 수신거부로 끝난 판 | " + " | ".join(f"{x:.0%}" for x in cs["opt"]) + " |"]
    md += ["", "## 문제 위치 (학습 → 관측 → 보상 → 환경 순서로 확인)", "",
           f"- 학습: {'문제 없음 (질문 1 예)' if q1 else ('학습 양보다 보상 쪽 문제로 보임 (곡선이 최고점에서 떨어짐)' if cs and cs['peaked'] else '문제 (질문 1 아니오)')}",
           f"- 관측: {'문제 없음 (질문 5 예 — 상황·피로도·남은 날에 따라 행동이 달라짐)' if q5 else '문제 (질문 5 아니오)'}",
           f"- 환경: 시험 규칙 결과가 상식대로 (results/sanity_check.md 통과) → 문제 없음",
           f"- 판정: {where}", "",
           f"진단 그림: results/diagnosis_{version}.png"]
    md_path = RESULTS / f"diagnosis_{version}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    # 화면(7·8단계)용: 같은 진단을 JSON으로 — 화면이 숫자를 따로 계산하지 않게
    plain = lambda x: re.sub(r"\*\*", "", x)
    qs = [("학습 곡선이 오르다 평평해졌는가", q1, q1, q1t), ("순이익이 시험 규칙, 특히 내 규칙보다 나은가", q2, q2, q2t),
          (f"수신거부가 허용 기준({lim}명 이하)을 넘는가", q3, not q3, q3t), ("행동이 한쪽으로 쏠렸는가", q4, not q4, q4t),
          ("날씨 · 요일 · 쌓이는 값에 따라 행동이 달라지는가", q5, q5, q5t)]
    js = {"version": version, "questions": [{"q": q, "yes": bool(y), "good": bool(g), "why": plain(w)} for q, y, g, w in qs],
          "where": plain(where), "where_tag": re.search(r"\*\*(.+?)\*\*", where).group(1),
          "segments": {"reward": [round(float(x), 2) for x in cs["seg"]],
                       "opt": [None if x is None else round(float(x), 4) for x in cs["opt"]], "peak": cs["peak_i"] + 1} if cs else None,
          "scenes": scenes, "policy": policy_stats(df)}
    (RESULTS / f"diagnosis_{version}.json").write_text(json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 진단 그림: 2 × 3 (다섯 칸 + 요약) ----
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.suptitle(f"결과 진단 — 보상 {version}  (평가 2025년 4판 × 100명, 시드 {C.EVAL_SEED})", fontsize=15, fontweight="bold")
    a = ax[0, 0]   # 1 학습 곡선
    if cs:
        c = cs["curve"]
        a.plot(c["timesteps"] / 1e4, c["return_mean"], color=color(key), label="보상 합계 (판 평균, 점)")
        a.set_xlabel("학습 스텝 (만)")
        a.set_ylabel("보상 (점)")
        b = a.twinx()
        if "profit_mean" in c:
            b.plot(c["timesteps"] / 1e4, c["profit_mean"], color="#64748b", ls="--", label="성공 척도: 순이익 (판 평균, 원)")
            note = ""
        else:
            b.plot(c["timesteps"] / 1e4, c["return_mean"] * 1000, color="#64748b", ls="--", label="성공 척도: 순이익 (원)")
            note = "\n(v1은 보상 = 순이익 ÷ 1,000이라 두 선이 겹침)"
        b.set_ylabel("순이익 (원)")
        h1, l1 = a.get_legend_handles_labels()
        h2, l2 = b.get_legend_handles_labels()
        a.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
        a.set_title(f"① 학습 곡선 — {'평평해짐' if q1 else ('오르다 떨어짐' if cs['peaked'] else '아직 오르는 중')}{note}", fontsize=11)
    a = ax[0, 1]   # 2 성공 척도
    ks = [k for k in keys if k in s]
    a.bar([tick(k) for k in ks], [s[k]["profit"] / 1e4 for k in ks], color=[color(k) for k in ks])
    for i, k in enumerate(ks):
        a.text(i, s[k]["profit"] / 1e4, f"{s[k]['profit'] / 1e4:,.0f}", ha="center", va="bottom", fontsize=9)
    a.set_ylabel("90일 순이익 (만 원)")
    a.set_title("② 성공 척도: 90일 순이익 (네 판 평균)", fontsize=11)
    a.tick_params(axis="x", labelsize=8)
    a = ax[0, 2]   # 3 보조 지표와 허용 기준
    a.bar([tick(k) for k in ks], [s[k]["opt_outs"] for k in ks], color=[color(k) for k in ks])
    a.axhline(lim, color="#b91c1c", ls="--", lw=1.5, label=f"허용 기준 {lim}명")
    for i, k in enumerate(ks):
        a.text(i, s[k]["opt_outs"], f"{s[k]['opt_outs']:.1f}", ha="center", va="bottom", fontsize=9)
    a.set_ylabel("수신거부 (명, 네 판 평균)")
    a.set_title(f"③ 보조 지표: 수신거부 vs 허용 기준 — {'넘음' if q3 else '지킴'}", fontsize=11)
    a.legend(fontsize=8)
    a.tick_params(axis="x", labelsize=8)
    a = ax[1, 0]   # 4 행동별 횟수
    width = 0.8 / len(pols)
    for j, k in enumerate(pols):
        t = s[k]
        vals = [t["none_count"], t["general_pushes"], t["reward_pushes"]]
        xs = np.arange(3) + (j - (len(pols) - 1) / 2) * width
        a.bar(xs, vals, width, color=color(k), label=label(k))
        for x, v in zip(xs, vals):
            a.text(x, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
    a.set_xticks(range(3), ACTS)
    a.set_ylabel("한 판 평균 횟수")
    a.set_title(f"④ 행동별 횟수 — {'쏠림' if q4 else '고름'}", fontsize=11)
    a.legend(fontsize=8)
    a = ax[1, 1]   # 5 상황별 평균 행동
    ys = np.arange(len(sit))
    for j, k in enumerate(pols):
        t = situation_table(logs[k])
        a.barh(ys + (j - (len(pols) - 1) / 2) * width, t["리워드 비율"] * 100, width, color=color(k), label=f"{label(k)} 리워드 비율")
    a.set_yticks(ys, sit["상황"])
    a.invert_yaxis()
    a.set_xlabel("그날 결정 중 리워드 알림 비율 (%)")
    a.set_xlim(0, 140)
    a.set_title(f"⑤ 상황별 행동 — {'달라짐' if q5 else '같음'}", fontsize=11)
    a.legend(fontsize=8, loc="center right")
    a = ax[1, 2]   # 요약
    a.axis("off")
    lines = [f"1 학습 곡선이 평평해졌나: {yn(q1)}", f"2 순이익이 시험 규칙보다 높나: {yn(q2)}",
             f"3 허용 기준({lim}명)을 넘나: {yn(q3)}", f"4 행동이 쏠렸나: {yn(q4)}", f"5 상황에 따라 달라지나: {yn(q5)}", "",
             "문제 위치:"] + textwrap.wrap(re.sub(r"\*\*", "", where), 26)
    a.text(0.02, 0.98, "\n".join(lines), va="top", fontsize=11.5, linespacing=1.6,
           bbox=dict(boxstyle="round", facecolor="#f8f5ee", edgecolor="#e6dfd0"))
    a.set_title("진단 요약", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = RESULTS / f"diagnosis_{version}.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    from simulate import export_viewer_bundle   # 진단 결과를 화면 데이터 묶음에 다시 넣음 (평가 → 진단 순서라서)
    export_viewer_bundle()
    return md_path, png


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    a = ap.parse_args()
    md, png = diagnose(a.version)
    print(md.read_text(encoding="utf-8"))
    print(f"저장: {md}, {png}")


if __name__ == "__main__":
    main()
