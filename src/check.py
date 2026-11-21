"""고객 성향 검사 — data/customer_traits.csv 가 docs/project_spec.md 3절과 맞는지 본다.

검사: 100명이 모두 있음 · 값이 범위 안 · 근거 빈칸 없음 · 근거가 나이·성별뿐이 아님 · 칸별 평균이 목표 평균 ±10% 안.
평균이 벗어난 칸만 확신도가 낮은 고객부터 값을 기준표 가운데 값으로 옮겨 맞추고, 원래 값을 adjusted 칸에 남긴다.
실행: uv run python src/check.py
"""
import re
import sys

import pandas as pd

from config import NUM_CUSTOMERS
from paths import DATA

PATH = DATA / "customer_traits.csv"
# 칸: (최소, 최대, 목표 평균) — docs/project_spec.md 3절
TRAITS = {
    "report_propensity": (0.02, 0.20, 0.08),
    "reward_sensitivity": (0.0, 1.0, 0.5),
    "fatigue_sensitivity": (0.5, 1.5, 1.0),
}
MEAN_TOLERANCE = 0.10          # 목표 평균 ±10%  # 가정
CONFIDENCE_ORDER = {"낮음": 0, "보통": 1, "높음": 2}
AGE_SEX_ONLY = re.compile(r"^[\s\d세대남녀자성별나이·,'‘’\"]*$")


def adjust_mean(df: pd.DataFrame, col: str, target: float) -> int:
    """확신도 낮은 고객부터 값을 목표 평균으로 옮겨, 평균이 허용 범위 안에 들 때까지. 고친 인원 수를 돌려준다."""
    order = df.assign(_c=df["confidence"].map(CONFIDENCE_ORDER)).sort_values(["_c", "customer_id"]).index
    changed = 0
    for i in order:
        if abs(df[col].mean() - target) <= target * MEAN_TOLERANCE:
            break
        too_high = df[col].mean() > target
        if (too_high and df.at[i, col] > target) or (not too_high and df.at[i, col] < target):
            note = f"{col}={df.at[i, col]}"
            df.at[i, "adjusted"] = f"{df.at[i, 'adjusted']}; {note}".strip("; ") if df.at[i, "adjusted"] else note
            df.at[i, col] = target
            changed += 1
    return changed


def main() -> int:
    df = pd.read_csv(PATH, encoding="utf-8-sig", keep_default_na=False)
    problems = []
    if sorted(df["customer_id"]) != list(range(1, NUM_CUSTOMERS + 1)):
        problems.append(f"고객 번호가 1~{NUM_CUSTOMERS}이 아님 (현재 {len(df)}명)")
    for col, (lo, hi, _) in TRAITS.items():
        bad = df[(df[col] < lo) | (df[col] > hi)]["customer_id"].tolist()
        if bad:
            problems.append(f"{col} 범위({lo}~{hi}) 밖: {bad}")
    empty = df[df["evidence"].str.strip() == ""]["customer_id"].tolist()
    if empty:
        problems.append(f"근거 빈칸: {empty}")
    age_sex = df[df["evidence"].str.match(AGE_SEX_ONLY)]["customer_id"].tolist()
    if age_sex:
        problems.append(f"근거가 나이·성별뿐: {age_sex} → 이 고객만 다시 추정 필요")
    if problems:
        print("검사 실패\n- " + "\n- ".join(problems))
        return 1

    lines = ["| 칸 | 목표 평균 | 실제 평균 | 차이 | 결과 |", "| --- | --- | --- | --- | --- |"]
    changed_total = 0
    for col, (_, _, target) in TRAITS.items():
        before = df[col].mean()
        changed = adjust_mean(df, col, target) if abs(before - target) > target * MEAN_TOLERANCE else 0
        changed_total += changed
        after = df[col].mean()
        result = f"평균 맞춤 ({changed}명 수정)" if changed else "통과"
        lines.append(f"| {col} | {target} | {after:.4f} | {(after - target) / target:+.1%} | {result} |")
    if changed_total:
        df.to_csv(PATH, index=False, encoding="utf-8-sig")
    conf = df["confidence"].value_counts().to_dict()
    print(f"검사 통과: {len(df)}명 · 범위 OK · 근거 빈칸 0 · 나이·성별뿐인 근거 0")
    print(f"확신도: 높음 {conf.get('높음', 0)} / 보통 {conf.get('보통', 0)} / 낮음 {conf.get('낮음', 0)}  (허용 범위: 목표 평균 ±{MEAN_TOLERANCE:.0%})")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
