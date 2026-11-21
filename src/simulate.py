"""가상 데이터 생성 — 시험 규칙(또는 학습된 정책)으로 가상 매장을 평가 조건대로 돌린다.

평가 조건(1 매장 정의): 2025년 4개 시작일 × 고객 100명 × 90일, 시드 고정 → 다시 돌려도 같은 결과.
모든 규칙이 같은 시작일·같은 난수(날씨·우연)를 쓰므로 규칙끼리 공정하게 비교된다.

출력
- results/logs/<규칙>.csv        : 날짜 · 판 · 고객 · 상태 · 행동 · 반응 · 이익 (사람이 보는 기록)
- docs/reports/sanity_check.md        : 시험 규칙 비교표와 통과 기준 (환경 타당성 검증)
- docs/reports/summary_<버전>.md      : 학습된 정책 vs 시험 규칙 (평가)
- public/data/replay_<규칙>.json : 매장 화면(뷰어) 재생용, 날마다 받은 보상 포함
- public/data/personas.json, public/data/project.json : 화면에 그릴 손님·프로젝트 정보
- public/data/viewer_data.js     : 위 JSON + 학습 곡선 + 검증 결과를 한 파일로 (public/index.html을 더블클릭해도 열리게)

실행: uv run python src/simulate.py            (시험 규칙 4개 → 환경 타당성 검증)
      uv run python src/simulate.py --policy v1 (학습된 v1 정책 평가 → summary_v1.md)
"""
import argparse
import json

import numpy as np
import pandas as pd

import config as C
import env as E
import reward as R
from check import MEAN_TOLERANCE, TRAITS
from env import LaunPotEnv, OBS_NAMES, load_factors
from paths import CURVES, DIAGNOSIS, LOGS, PUBLIC_DATA, REPORTS, RESULTS, REWARD_HISTORY, SEED_CHECK

VIEWER_BUNDLE = PUBLIC_DATA / "viewer_data.js"
ACTION_KEYS = {C.ACTION_NO_PUSH: "none", C.ACTION_GENERAL_PUSH: "general", C.ACTION_REWARD_PUSH: "reward"}
CURVE_POINTS = 400   # 화면에 넘기는 학습 곡선 점 개수 (2만여 줄을 줄여 담음)  # 가정
# 화면의 '직접 해 보기'가 env.py와 같은 숫자로 계산하도록 넘기는 반응 규칙 가정값 이름
RULE_NAMES = ["GENERAL_PUSH_MULTIPLIER", "REWARD_PROB_BASE", "REWARD_PROB_SLOPE",
              "FATIGUE_PER_PUSH", "FATIGUE_RECOVERY", "FATIGUE_DANGER", "OPT_OUT_PROB",
              "DEPENDENCY_PER_REWARD", "DEPENDENCY_RECOVERY", "DEPENDENCY_DANGER", "DEPENDENCY_PENALTY",
              "RAIN_MM", "DEMAND_UP", "COLD_TEMP_MIN", "COLD_MULTIPLIER", "MONTH_END_REWARD_MULTIPLIER",
              "CHANCE_LOW", "CHANCE_HIGH", "LEVEL_MAX"]


# ---------------------------------------------------------------------------
# 시험 규칙: (env, obs, rng) → 행동
# ---------------------------------------------------------------------------
def rule_none(env, obs, rng):
    return C.ACTION_NO_PUSH


def rule_max(env, obs, rng):
    return C.ACTION_REWARD_PUSH


def rule_random(env, obs, rng):
    return int(rng.integers(C.NUM_ACTIONS))


def rule_current(env, obs, rng):
    """지금의 운영 방식 (가정): 매월 25일 전원 리워드, 매주 토요일 전원 일반 알림."""
    date = pd.Timestamp(env.dates[env.pos])
    if date.day == C.CURRENT_REWARD_PUSH_DAY:
        return C.ACTION_REWARD_PUSH
    if date.day_of_week == C.CURRENT_GENERAL_PUSH_WEEKDAY:
        return C.ACTION_GENERAL_PUSH
    return C.ACTION_NO_PUSH


TEST_RULES = {
    "none": ("아무것도 안 함", rule_none),
    "max_reward": ("매일 최대로 행동 (매일 전원 리워드)", rule_max),
    "random": ("무작위", rule_random),
    "current": ("현재 운영 (토요일 일반 + 25일 리워드, 가정)", rule_current),
}


def policy_rule(version: str, path=None):
    from stable_baselines3 import PPO
    model = PPO.load(path or RESULTS / "models" / version / "model.zip", device="cpu")

    def act(env, obs, rng):
        return int(model.predict(obs, deterministic=True)[0])
    return act


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def run(policy) -> pd.DataFrame:
    env = LaunPotEnv()
    parts = getattr(R, "reward_parts", None)   # 보상 항목별 기록 (v2부터)
    rows = []
    for ep, start in enumerate(C.EVAL_START_DATES):
        for cid in sorted(env.traits.index):
            seed = C.EVAL_SEED * 10_000 + ep * 1_000 + int(cid)
            obs, _ = env.reset(seed=seed, options={"customer_id": int(cid), "start_date": start})
            rng = np.random.default_rng([C.EVAL_SEED, ep, int(cid), 7])
            done = False
            while not done:
                action = policy(env, obs, rng)
                obs, rew, term, trunc, info = env.step(action)
                info.update(episode=ep + 1, reward=rew)
                if parts:
                    info.update({f"rp_{k}": v for k, v in parts(info).items()})
                rows.append(info)
                done = term or trunc
    df = pd.DataFrame(rows)
    df["action_name"] = df["action"].map(C.ACTION_NAMES)
    return df


def summarize(df: pd.DataFrame) -> dict:
    """성공 척도와 보조 지표 — 평가 4판의 평균 (한 판 = 고객 100명 × 90일)."""
    n_ep = df["episode"].nunique()
    days = C.EPISODE_LENGTH_DAYS
    factors = load_factors()
    up = df["date"].map(factors["is_rain"] | factors["is_holiday_or_weekend"]).astype(bool)
    cold = df["date"].map(factors["is_cold"]).astype(bool)
    return {
        "profit": df["profit"].sum() / n_ep,
        "opt_outs": df["opted_out"].sum() / n_ep,
        "success": bool(df.groupby("episode")["opted_out"].sum().le(C.MAX_OPT_OUT_CUSTOMERS).all()),   # 허용 기준을 네 판 모두 지킴
        "reports_per_day": df["reported"].sum() / n_ep / days,
        "reward_cost": df["reward_cost"].sum() / n_ep,
        "general_pushes": (df["action"] == C.ACTION_GENERAL_PUSH).sum() / n_ep,
        "reward_pushes": (df["action"] == C.ACTION_REWARD_PUSH).sum() / n_ep,
        "none_count": (df["action"] == C.ACTION_NO_PUSH).sum() / n_ep,
        "active_end": C.NUM_CUSTOMERS - df["opted_out"].sum() / n_ep,
        "report_rate_up": df.loc[up, "reported"].mean(),
        "report_rate_normal": df.loc[~up & ~cold, "reported"].mean(),
        "report_rate_cold": df.loc[cold & ~up, "reported"].mean(),
        "avg_fatigue": df["fatigue_after"].mean(),
        "total_reward": df["reward"].sum() / n_ep,
        "reward_parts": {c[3:]: df[c].sum() / n_ep for c in df.columns if c.startswith("rp_")},   # 보상 항목별 한 판 합계
    }


def compare_table(summaries: dict[str, tuple[str, dict]]) -> str:
    head = (f"| 규칙 | 90일 순이익 | 수신거부 | 허용 기준 (수신거부 {C.MAX_OPT_OUT_CUSTOMERS}명 이하) | 하루 평균 제보 | 리워드 비용 | 일반 발송 | 리워드 발송 | 미발송 |\n"
            "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    lines = [head]
    for key, (label, s) in summaries.items():
        lines.append(
            f"| {label} | {s['profit']:,.0f}원 | {s['opt_outs']:.1f}명 | {'지킴' if s['success'] else '넘음'} "
            f"| {s['reports_per_day']:.1f}건 | {s['reward_cost']:,.0f}원 | {s['general_pushes']:,.0f} "
            f"| {s['reward_pushes']:,.0f} | {s['none_count']:,.0f} |")
    return "\n".join(lines)


def sanity_report(summaries: dict) -> tuple[str, bool]:
    s = {k: v for k, (_, v) in summaries.items()}
    best = max(s, key=lambda k: s[k]["profit"])
    up_ratio = s["none"]["report_rate_up"] / s["none"]["report_rate_normal"]
    cold_ratio = s["none"]["report_rate_cold"] / s["none"]["report_rate_normal"]
    profits = [v["profit"] for v in s.values()]
    spread = (max(profits) - min(profits)) / max(abs(min(profits)), 1)
    checks = [
        ("매일 최대로 행동이 최선이 아님", best != "max_reward",
         f"최고 이익 규칙 = {summaries[best][0]}, 매일 최대의 수신거부 {s['max_reward']['opt_outs']:.0f}명"),
        ("날씨·공휴일에 따라 결과가 달라짐", 1.25 <= up_ratio <= 1.55 and 0.6 <= cold_ratio <= 0.8,
         f"미발송 기준 제보율: 비·주말·공휴일 {up_ratio:.2f}배, 한파 {cold_ratio:.2f}배 (규칙5: 1.4배, 0.7배)"),
        ("시험 규칙 사이에 차이가 있음", spread > 0.1, f"이익 최고·최저 차이 {spread:.0%}"),
        ("하루 제보 수가 상식적인 범위 (100명 앱 유저, 3~40건)", all(3 <= v["reports_per_day"] <= 40 for v in s.values()),
         ", ".join(f"{summaries[k][0].split(' (')[0]} {v['reports_per_day']:.1f}건" for k, v in s.items())),
    ]
    ok = all(c[1] for c in checks)
    lines = ["| 통과 기준 | 결과 | 근거 |", "| --- | --- | --- |"]
    lines += [f"| {name} | {'통과' if passed else '실패'} | {why} |" for name, passed, why in checks]
    return "\n".join(lines), ok


# ---------------------------------------------------------------------------
# 화면(뷰어) 재생용 데이터
# ---------------------------------------------------------------------------
def export_replay(key: str, label: str, df: pd.DataFrame, summary: dict) -> None:
    PUBLIC_DATA.mkdir(parents=True, exist_ok=True)
    f = load_factors()
    episodes = []
    for ep, g in df.groupby("episode"):
        start = C.EVAL_START_DATES[ep - 1]
        dates = pd.date_range(start, periods=C.EPISODE_LENGTH_DAYS).strftime("%Y-%m-%d")
        by_date = {d: x.set_index("customer_id") for d, x in g.groupby("date")}
        days = []
        for d in dates:
            x = by_date.get(d)
            ids = range(1, C.NUM_CUSTOMERS + 1)
            get = (lambda col, default: [int(round(x[col].get(i, default))) if x is not None else default for i in ids])
            days.append({
                "date": d, "weekday": int(f.at[d, "day_of_week"]),
                "holiday": f.at[d, "holiday_name"] if isinstance(f.at[d, "holiday_name"], str) else "",
                "weather": {"temp_max": f.at[d, "temp_max"], "temp_min": f.at[d, "temp_min"],
                            "precip_mm": f.at[d, "precipitation_mm"], "snow_cm": f.at[d, "snowfall_cm"]},
                "action": get("action", -1),          # -1 = 이미 수신거부해 결정 없음
                "reported": get("reported", 0),
                "opted_out": get("opted_out", 0),
                "fatigue": get("fatigue_after", -1),
                "dependency": get("dependency_after", -1),
                "profit": int(x["profit"].sum()) if x is not None else 0,
                "reward": round(float(x["reward"].sum()), 3) if x is not None else 0.0,
            })
        episodes.append({"episode": int(ep), "start": start, "days": days,
                         "profit": int(g["profit"].sum()), "opt_outs": int(g["opted_out"].sum())})
    out = {"rule": key, "label": label, "reward_version": R.VERSION, "is_my_rule": key == "current",
           "summary": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()},
           "episodes": episodes}
    (PUBLIC_DATA / f"replay_{key}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def reward_versions() -> list[dict]:
    """보관된 보상 파일(reward_v1.py …)과 지금 reward.py의 버전·한 줄 식 — 화면의 버전 설명용."""
    import importlib
    out = {}
    names = [f"reward_history.{p.stem}" for p in sorted(REWARD_HISTORY.glob("reward_v*.py"))] + ["reward"]
    for name in names:
        m = importlib.import_module(name)
        out[m.VERSION] = getattr(m, "SUMMARY", "그날 순이익 ÷ 1,000")
    return [{"version": v, "formula": f} for v, f in sorted(out.items(), key=lambda x: int(x[0][1:]))]


def export_static() -> None:
    PUBLIC_DATA.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv("data/personas.csv", encoding="utf-8-sig")
    t = pd.read_csv("data/customer_traits.csv", encoding="utf-8-sig", keep_default_na=False)
    m = p.merge(t, on="customer_id")
    people = [{
        "id": int(r.customer_id), "sex": r.sex, "age": int(r.age), "occupation": r.occupation,
        "family_type": r.family_type, "housing_type": r.housing_type, "summary": r.persona,
        "report_propensity": r.report_propensity, "reward_sensitivity": r.reward_sensitivity,
        "fatigue_sensitivity": r.fatigue_sensitivity, "confidence": r.confidence, "evidence": r.evidence,
    } for r in m.itertuples()]
    (PUBLIC_DATA / "personas.json").write_text(json.dumps(people, ensure_ascii=False), encoding="utf-8")
    project = {
        "store": C.STORE_NAME, "business": C.BUSINESS_TYPE, "location": f"{C.CITY} {C.DISTRICT} · {C.TRADE_AREA}",
        "business_hours": list(C.BUSINESS_HOURS), "push_hour": C.PUSH_SEND_HOUR,
        "agent": "CRM 매니저 — 매일 18:00 유저마다 오늘 보낼 푸시를 정함",
        "state": OBS_NAMES,
        "state_labels": ["기본 제보 성향", "리워드 민감도", "알림 피로 민감도", "알림 피로도", "보상 의존도",
                         "비", "공휴일·주말", "한파", "월말", "요일", "남은 날 수"],
        "actions": {str(k): v for k, v in C.ACTION_NAMES.items()},
        "weather_rules": {"rain_mm": E.RAIN_MM, "cold_temp_min": E.COLD_TEMP_MIN},
        "location": {"lat": C.STORE_LAT, "lon": C.STORE_LON},
        "reward": f"{R.VERSION}: {getattr(R, 'SUMMARY', '그날 순이익 ÷ 1,000')}",
        "reward_version": R.VERSION, "reward_scale": R.REWARD_SCALE,
        "reward_versions": reward_versions(),
        "success_metric": C.SUCCESS_METRIC + f", 수신거부 {C.MAX_OPT_OUT_CUSTOMERS}명 이하",
        "episode_days": C.EPISODE_LENGTH_DAYS, "eval_starts": C.EVAL_START_DATES,
        "rules": {k: getattr(E, k) for k in RULE_NAMES},
        "money": {"value_per_report": C.DATA_VALUE_PER_REPORT, "reward_cost": C.REWARD_COST_PER_REPORT,
                  "max_opt_outs": C.MAX_OPT_OUT_CUSTOMERS},
        "month_end_start_day": C.MONTH_END_START_DAY,
        "current_rule": {"general_weekday": C.CURRENT_GENERAL_PUSH_WEEKDAY, "reward_day": C.CURRENT_REWARD_PUSH_DAY},
        "traits": {k: {"min": lo, "max": hi, "target": tg} for k, (lo, hi, tg) in TRAITS.items()},
        "trait_tolerance": MEAN_TOLERANCE,
    }
    (PUBLIC_DATA / "project.json").write_text(json.dumps(project, ensure_ascii=False, indent=1), encoding="utf-8")


def export_viewer_bundle() -> None:
    """public/data의 JSON과 학습 곡선·검증 결과를 viewer_data.js 하나로 묶는다.

    브라우저는 파일을 더블클릭해 열면 JSON을 읽지 못하지만 <script>로 넣은 JS 파일은 읽으므로,
    public/index.html을 서버 없이 열 수 있다.
    """
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))
    out = {"project": load(PUBLIC_DATA / "project.json"), "personas": load(PUBLIC_DATA / "personas.json"),
           "replays": {}, "curves": {}, "sanity": [], "diagnosis": {}, "seed_checks": {}}
    for p in sorted(SEED_CHECK.glob("seed_check_v*.json")):   # 채택 버전 다시 확인 (화면 7·10단계)
        out["seed_checks"][p.stem.removeprefix("seed_check_")] = load(p)
    for p in sorted(DIAGNOSIS.glob("diagnosis_v*.json")):   # diagnose.py가 만든 진단 · 정책 해석 (화면 7·8단계)
        out["diagnosis"][p.stem.removeprefix("diagnosis_")] = load(p)
    keys = list(TEST_RULES) + sorted(p.stem.removeprefix("replay_") for p in PUBLIC_DATA.glob("replay_ppo_*.json"))
    for k in keys:
        if (PUBLIC_DATA / f"replay_{k}.json").exists():
            out["replays"][k] = load(PUBLIC_DATA / f"replay_{k}.json")
    for p in sorted(CURVES.glob("learning_curve_*.csv")):
        c = pd.read_csv(p)
        pts = c.iloc[::max(1, len(c) // CURVE_POINTS)]
        if pts.index[-1] != c.index[-1]:
            pts = pd.concat([pts, c.tail(1)])
        n = max(1, len(c) // 10)      # train.py와 같이 처음·마지막 10% 판의 평균
        out["curves"][p.stem.removeprefix("learning_curve_")] = {
            # [학습 스텝, 보상(점), 순이익 ÷ 1,000(점)] — 순이익은 v2부터 기록 (v1은 보상 = 순이익 ÷ 1,000)
            "points": [[int(t), round(float(r), 2), round(float(p) / 1000, 2)] for t, r, p in
                       zip(pts["timesteps"], pts["return_mean"], pts["profit_mean"] if "profit_mean" in pts else pts["return_mean"] * 1000)],
            "episodes": len(c), "timesteps": int(c["timesteps"].iloc[-1]),
            "first_return": round(float(c["episode_return"].head(n).mean()), 2),
            "last_return": round(float(c["episode_return"].tail(n).mean()), 2)}
    sanity = REPORTS / "sanity_check.md"
    if sanity.exists():   # '통과 기준' 표의 줄을 그대로 옮김 (기준 숫자를 화면에 따로 적지 않게)
        rows = sanity.read_text(encoding="utf-8").split("## 통과 기준")[-1].splitlines()
        for line in rows:
            cells = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(cells) == 3 and cells[1] in ("통과", "실패"):
                out["sanity"].append({"name": cells[0], "passed": cells[1] == "통과", "why": cells[2]})
    VIEWER_BUNDLE.write_text("window.VIEWER_DATA = " + json.dumps(out, ensure_ascii=False) + ";\n", encoding="utf-8")


# ---------------------------------------------------------------------------
def run_rules(rules: dict) -> dict:
    LOGS.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for key, (label, fn) in rules.items():
        df = run(fn)
        cols = ["date", "episode", "customer_id", "fatigue_before", "dependency_before", "action", "action_name",
                "report_prob", "reported", "opted_out", "value", "reward_cost", "profit", "reward"]
        cols += [c for c in df.columns if c.startswith("rp_")]   # 보상 항목별 (v2부터)
        df[cols].round(4).to_csv(LOGS / f"{key}.csv", index=False, encoding="utf-8-sig")
        s = summarize(df)
        summaries[key] = (label, s)
        export_replay(key, label, df, s)
    export_static()
    return summaries


def sanity_check() -> tuple[dict, bool]:
    summaries = run_rules(TEST_RULES)
    table = compare_table(summaries)
    checks, ok = sanity_report(summaries)
    md = (f"# 환경 타당성 검증\n\n평가 조건: {', '.join(C.EVAL_START_DATES)} 시작 90일 × 고객 {C.NUM_CUSTOMERS}명, "
          f"시드 {C.EVAL_SEED} · 값은 4판 평균 · 보상 버전 {R.VERSION}\n\n## 시험 규칙 비교\n\n{table}\n\n"
          f"## 통과 기준\n\n{checks}\n\n**종합: {'통과' if ok else '실패'}**\n")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "sanity_check.md").write_text(md, encoding="utf-8")
    export_viewer_bundle()
    return summaries, ok


def evaluate(version: str) -> dict:
    key = f"ppo_{version}"
    rules = {**TEST_RULES, key: (f"학습된 정책 {version} (PPO)", policy_rule(version))}
    summaries = run_rules(rules)
    # 앞 보상 버전들은 저장된 재생 기록의 결과를 그대로 읽어 나란히 둔다
    # (다시 돌리면 그 버전 재생 기록의 '받은 보상'이 지금 보상 식으로 바뀌어 버리므로 다시 돌리지 않음)
    pols = {}
    for p in sorted(PUBLIC_DATA.glob("replay_ppo_*.json"), key=lambda p: int(p.stem.split("_v")[-1])):
        k = p.stem.removeprefix("replay_")
        if k == key:
            pols[k] = summaries.pop(k)
        else:
            r = json.loads(p.read_text(encoding="utf-8"))
            pols[k] = (f"{r['label']} — 보상 {r['reward_version']}", r["summary"])
    summaries.update(pols)
    table = compare_table(summaries)
    s, cur = summaries[key][1], summaries["current"][1]
    acts = "| 행동 | " + " | ".join(k.removeprefix("ppo_") for k in pols) + " |\n| --- |" + " ---: |" * len(pols) + "\n"
    for name, col in [("보내지 않음", "none_count"), ("일반 알림", "general_pushes"), ("리워드 알림", "reward_pushes")]:
        cells = []
        for k in pols:
            t = pols[k][1]
            tot = t["none_count"] + t["general_pushes"] + t["reward_pushes"]
            cells.append(f"{t[col]:,.0f} ({t[col] / tot:.0%})")
        acts += f"| {name} | " + " | ".join(cells) + " |\n"
    parts = s.get("reward_parts") or {"total": s["total_reward"]}
    part_names = {"profit": "단기: 순이익 ÷ 1,000", "fatigue_danger": "장기: 피로 위험 구간 벌점", "opt_out": "장기: 수신거부 벌점",
                  "total": "보상 합계"}
    part_rows = "".join(f"| {part_names.get(k, k)} | {v:,.1f}점 |\n" for k, v in parts.items())
    ok = {k: t for k, (_, t) in pols.items() if t["success"]}
    adopt = max(ok, key=lambda k: ok[k]["profit"]) if ok else "ppo_v1"
    adopt_why = (f"허용 기준을 지키는 버전 가운데 순이익이 가장 높음 ({ok[adopt]['profit']:,.0f}원)" if ok
                 else "허용 기준을 지키는 버전이 없어 초기 보상 v1이 채택 버전")
    v1 = pols.get("ppo_v1", (None, s))[1]
    verdict = "낫다" if (s["profit"] > cur["profit"] and s["success"]) else "낫지 않다"
    md = (f"# 평가 결과 — 보상 {version}\n\n평가 조건: {', '.join(C.EVAL_START_DATES)} 시작 90일 × 고객 {C.NUM_CUSTOMERS}명, "
          f"시드 {C.EVAL_SEED} (환경 타당성 검증과 같음) · 값은 4판 평균 · 성공 척도 = 90일 순이익 · "
          f"허용 기준 = 수신거부 {C.MAX_OPT_OUT_CUSTOMERS}명 이하 (한 판마다)\n\n"
          f"## 성공 척도와 보조 지표 (보상 버전 나란히)\n\n{table}\n\n"
          f"## 학습된 정책의 행동별 횟수 (한 판 평균)\n\n{acts}\n"
          f"## 보상 {version}의 항목별 합계 (한 판 평균, 고객 100명 합)\n\n| 보상 항목 | 합계 |\n| --- | ---: |\n{part_rows}\n"
          f"보상 점수는 버전마다 기준이 달라 버전끼리 비교하지 않는다 — 비교는 위의 성공 척도와 허용 기준으로.\n\n"
          f"## 판별\n\n"
          f"- v1 대비: 순이익 {s['profit']:,.0f}원 vs {v1['profit']:,.0f}원 ({s['profit'] - v1['profit']:+,.0f}원), "
          f"수신거부 {s['opt_outs']:.1f}명 vs {v1['opt_outs']:.1f}명 · 허용 기준 {'지킴' if s['success'] else '넘음'}\n"
          f"- **학습된 정책 {version}은 '현재 운영'보다 {verdict}** "
          f"(순이익 {s['profit']:,.0f}원 vs {cur['profit']:,.0f}원, 수신거부 {s['opt_outs']:.1f}명 vs {cur['opt_outs']:.1f}명)\n"
          f"- **지금까지의 채택 버전: {adopt.removeprefix('ppo_')}** — {adopt_why}\n")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"summary_{version}.md").write_text(md, encoding="utf-8")
    export_viewer_bundle()
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", help="학습된 정책 버전 (예: v1)")
    a = ap.parse_args()
    if a.policy:
        summaries = evaluate(a.policy)
        print((REPORTS / f"summary_{a.policy}.md").read_text(encoding="utf-8"))
    else:
        summaries, ok = sanity_check()
        print((REPORTS / "sanity_check.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
