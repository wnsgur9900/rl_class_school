"""채택 버전 다시 확인 — 학습 시드(무작위 출발 번호)만 바꿔 다시 학습하고, 같은 평가 조건에서 결론이 유지되는지 본다.

채택한 모델과 결과 파일은 덮어쓰지 않는다. 새 학습은 results/seed_check/<버전>_seed<시드>/ 에만 저장한다.
만드는 것: docs/reports/seed_check_<버전>.md   (사람이 읽는 비교표와 판정)
          results/seed_check/seed_check_<버전>.json (같은 숫자 → 화면 7·10단계, viewer_data.js에 묶임)

실행: uv run python src/seed_check.py                  (본 학습과 같은 스텝, 약 8~10분)
      uv run python src/seed_check.py --timesteps 20000 (동작만 확인)
"""
import argparse
import json

import numpy as np
import pandas as pd

import config as C
import reward as R
import simulate as S
import train as T
from paths import CURVES, PUBLIC_DATA, REPORTS, SEED_CHECK

# 다시 확인 설정  # 가정
RECHECK_SEED = 1042        # 원래 학습 시드 42와 멀리 (가상 환경 8개가 시드+0~7을 쓰므로 가까운 번호는 일부 겹침)
PROFIT_TOLERANCE = 0.05    # 판정 3: 채택 모델과 순이익 차이가 5% 안이면 '비슷함'
CURVE_GRID = 200           # 화면에서 두 학습 곡선을 겹쳐 그릴 점 개수


def replay(key: str) -> dict:
    return json.loads((PUBLIC_DATA / f"replay_{key}.json").read_text(encoding="utf-8"))


def brief(s: dict) -> dict:
    """비교에 쓰는 값만 — 한 판 평균."""
    total = s["none_count"] + s["general_pushes"] + s["reward_pushes"]
    return {"profit": round(float(s["profit"]), 2), "opt_outs": round(float(s["opt_outs"]), 2), "success": bool(s["success"]),
            "reports_per_day": round(float(s["reports_per_day"]), 2), "reward_cost": round(float(s["reward_cost"]), 2),
            "share": [round(float(s[k]) / float(total), 4) for k in ("none_count", "general_pushes", "reward_pushes")]}


def overlay(base_csv, new_csv) -> dict:
    """두 학습 곡선을 같은 가로축 점에 맞춘다 (판 길이가 달라 그대로는 겹쳐 그릴 수 없음)."""
    a, b = pd.read_csv(base_csv), pd.read_csv(new_csv)
    end = min(a["timesteps"].iloc[-1], b["timesteps"].iloc[-1])
    t = np.linspace(end / CURVE_GRID, end, CURVE_GRID)
    return {"t": [int(x) for x in t],
            "base": [round(float(x), 2) for x in np.interp(t, a["timesteps"], a["return_mean"])],
            "recheck": [round(float(x), 2) for x in np.interp(t, b["timesteps"], b["return_mean"])]}


def write_report(r: dict) -> None:
    a, b, v = r["base"], r["recheck"], r["version"]
    won = lambda x: f"{x:,.0f}원"
    rows = [("90일 순이익 (성공 척도)", won(a["profit"]), won(b["profit"]), f"{b['profit'] - a['profit']:+,.0f}원 ({r['profit_gap']:+.1%})"),
            ("수신거부 (네 판 평균)", f"{a['opt_outs']:.1f}명", f"{b['opt_outs']:.1f}명", f"{b['opt_outs'] - a['opt_outs']:+.1f}명"),
            (f"허용 기준 ({C.MAX_OPT_OUT_CUSTOMERS}명 이하)", "지킴" if a["success"] else "넘음", "지킴" if b["success"] else "넘음", ""),
            ("리워드 비용", won(a["reward_cost"]), won(b["reward_cost"]), f"{b['reward_cost'] - a['reward_cost']:+,.0f}원"),
            ("하루 평균 제보", f"{a['reports_per_day']:.1f}건", f"{b['reports_per_day']:.1f}건", f"{b['reports_per_day'] - a['reports_per_day']:+.1f}건"),
            ("행동 비율 (안 보냄 · 일반 · 리워드)", " · ".join(f"{x:.0%}" for x in a["share"]), " · ".join(f"{x:.0%}" for x in b["share"]), "")]
    md = [f"# 채택 버전 다시 확인 — 보상 {v}", "",
          f"학습 시드(무작위 출발 번호)만 {r['base_seed']} → {r['seed']}로 바꿔 {r['steps']:,}스텝을 다시 학습하고, "
          f"같은 평가 조건({', '.join(C.EVAL_START_DATES)} 시작 90일 × 고객 {C.NUM_CUSTOMERS}명, 시드 {C.EVAL_SEED})에서 비교했다. "
          "채택 모델과 그 결과 파일은 덮어쓰지 않았다.", "",
          "## 비교 (네 판 평균)", "", f"| 항목 | 채택 모델 (시드 {r['base_seed']}) | 다시 학습 (시드 {r['seed']}) | 차이 |", "| --- | ---: | ---: | ---: |"]
    md += [f"| {x[0]} | {x[1]} | {x[2]} | {x[3]} |" for x in rows]
    md += ["", "## 평가 기간별", "", "| 시작일 | 채택 순이익 | 다시 학습 순이익 | 채택 수신거부 | 다시 학습 수신거부 |", "| --- | ---: | ---: | ---: | ---: |"]
    md += [f"| {q['start']} | {won(q['base_profit'])} | {won(q['profit'])} | {q['base_opt_outs']}명 | {q['opt_outs']}명 |" for q in r["quarters"]]
    md += ["", "## 판정 (학습 전에 정한 기준)", "", "| 기준 | 결과 | 근거 |", "| --- | --- | --- |"]
    md += [f"| {c['name']} | {'통과' if c['passed'] else '실패'} | {c['why']} |" for c in r["checks"]]
    over, noise = b["profit"] - r["v1_profit"], abs(b["profit"] - a["profit"])   # v1과의 차이 vs 시드만 바꿔도 생기는 차이
    md += ["", f"**종합: {'통과 — 시드를 바꿔도 채택 결론이 유지됨' if r['passed'] else '실패 — 시드에 따라 결론이 달라짐, 채택을 다시 검토'}**", "",
           f"참고 (판정 기준 아님): 다시 학습한 순이익과 초기 보상 v1({won(r['v1_profit'])})의 차이는 {over:+,.0f}원."
           + (f" 이 차이는 시드만 바꿔도 생기는 차이({noise:,.0f}원)보다 작아 'v1보다 돈을 더 번다'고 단정할 수 없다. "
              "확실한 개선은 비슷한 순이익에서 수신거부를 크게 줄인 것이다." if abs(over) < noise else "")]
    (REPORTS / f"seed_check_{v}.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def recheck(version: str = R.VERSION, seed: int = RECHECK_SEED, timesteps: int | None = None) -> dict:
    steps = timesteps or T.TOTAL_TIMESTEPS
    out = SEED_CHECK / f"{version}_seed{seed}"
    base = replay(f"ppo_{version}")
    current, v1 = replay("current")["summary"], replay("ppo_v1")["summary"]
    t = T.train(version, steps, False, seed=seed, out=out)
    df = S.run(S.policy_rule(version, out / "model.zip"))
    a, b = brief(base["summary"]), brief(S.summarize(df))
    gap = (b["profit"] - a["profit"]) / a["profit"]
    per = df.groupby("episode").agg(profit=("profit", "sum"), opt_outs=("opted_out", "sum"))
    quarters = [{"start": e["start"], "base_profit": e["profit"], "base_opt_outs": e["opt_outs"],
                 "profit": int(per.at[e["episode"], "profit"]), "opt_outs": int(per.at[e["episode"], "opt_outs"])} for e in base["episodes"]]
    checks = [
        {"name": f"허용 기준(수신거부 {C.MAX_OPT_OUT_CUSTOMERS}명 이하)을 네 판 모두 지킴", "passed": b["success"],
         "why": "판별 수신거부 " + ", ".join(f"{q['opt_outs']}명" for q in quarters)},
        {"name": "현재 운영보다 순이익이 높음", "passed": b["profit"] > current["profit"],
         "why": f"{b['profit']:,.0f}원 vs {current['profit']:,.0f}원 ({b['profit'] / current['profit']:.2f}배)"},
        {"name": f"채택 모델과 순이익 차이가 ±{PROFIT_TOLERANCE:.0%} 안", "passed": abs(gap) <= PROFIT_TOLERANCE,
         "why": f"{b['profit']:,.0f}원 vs {a['profit']:,.0f}원 ({gap:+.1%})"},
    ]
    r = {"version": version, "base_seed": C.TRAIN_SEED, "seed": seed, "steps": steps, "seconds": round(t["seconds"]),
         "base": a, "recheck": b, "profit_gap": round(gap, 4), "current_profit": round(float(current["profit"]), 2),
         "v1_profit": round(float(v1["profit"]), 2), "quarters": quarters, "checks": checks,
         "passed": all(c["passed"] for c in checks),
         "curve": overlay(CURVES / f"learning_curve_{version}.csv", out / f"learning_curve_{version}.csv")}
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_report(r)
    (SEED_CHECK / f"seed_check_{version}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    S.export_viewer_bundle()
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=RECHECK_SEED)
    ap.add_argument("--timesteps", type=int)
    a = ap.parse_args()
    r = recheck(seed=a.seed, timesteps=a.timesteps)
    print((REPORTS / f"seed_check_{r['version']}.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
