"""실행 메뉴 — run.bat 을 더블클릭하면 이 메뉴가 뜬다.

실행할 때마다 결과를 덮어쓰지 않고 쌓는다.
- results/runs/<월일_시각_이름>/ : 그 실행의 요약·로그와 그때의 env.py(가정값) · config.py · reward.py 사본
- results/run_history.csv       : 실행마다 한 줄 (날짜 · 실행 종류 · 성공 척도 · 메모)
가장 최근 결과는 지금처럼 results/logs/, docs/reports/sanity_check.md 등에도 남는다.

직접 실행: uv run python src/run.py [번호] [--memo "메모"]
"""
import argparse
import csv
import shutil
from datetime import datetime
from pathlib import Path

import reward as R
from paths import CURVES, REPORTS, RESULTS

RUNS = RESULTS / "runs"
HISTORY = RESULTS / "run_history.csv"
SNAPSHOT_FILES = ["src/env.py", "src/config.py", "src/reward.py"]

MENU = {
    "1": ("sanity", "가상 데이터 생성과 환경 타당성 검증"),
    "2": ("train_trial", "학습 — 시험 학습 (1분 안쪽, 오류 확인)"),
    "3": ("train", f"학습 — 본 학습 (보상 {R.VERSION}, 400만 스텝 약 8~10분)"),
    "4": ("eval", f"학습된 정책 평가와 진단 (보상 {R.VERSION} → summary · diagnosis_{R.VERSION}.md/png)"),
    "5": ("seed_check", f"채택 버전 다시 확인 (보상 {R.VERSION}, 학습 시드만 바꿔 약 8~10분 · 기존 모델은 그대로)"),
}


def won(x: float) -> str:
    return f"{x:,.0f}원"


def do_sanity(run_dir: Path) -> str:
    import simulate
    summaries, ok = simulate.sanity_check()
    shutil.copy(REPORTS / "sanity_check.md", run_dir)
    shutil.copytree(simulate.LOGS, run_dir / "logs")
    cur, mx = summaries["current"][1], summaries["max_reward"][1]
    return (f"검증 {'통과' if ok else '실패'} · 현재 운영 {won(cur['profit'])}/수신거부 {cur['opt_outs']:.0f}명 · "
            f"매일 최대 {won(mx['profit'])}/{mx['opt_outs']:.0f}명")


def do_train(run_dir: Path, trial: bool) -> str:
    import train
    steps = train.TRIAL_TIMESTEPS if trial else train.TOTAL_TIMESTEPS
    r = train.train(R.VERSION, steps, trial)
    msg = f"{r['steps']:,}스텝 {r['seconds']:.0f}초"
    if not trial:
        for f in [f"learning_curve_{R.VERSION}.png", f"learning_curve_{R.VERSION}.csv"]:
            shutil.copy(CURVES / f, run_dir)
        msg += f" · 평균 보상 {r['first_return']:.2f} → {r['last_return']:.2f}점"
    return msg


def do_eval(run_dir: Path) -> str:
    import simulate
    import diagnose
    summaries = simulate.evaluate(R.VERSION)
    md, png = diagnose.diagnose(R.VERSION)   # 3회차 1-1: 다섯 질문 진단표 · 진단 그림
    for f in [REPORTS / f"summary_{R.VERSION}.md", md, png]:
        shutil.copy(f, run_dir)
    shutil.copytree(simulate.LOGS, run_dir / "logs")
    s = summaries[f"ppo_{R.VERSION}"][1]
    return (f"{R.VERSION} 정책 {won(s['profit'])}/수신거부 {s['opt_outs']:.1f}명 · "
            f"허용 기준 {'지킴' if s['success'] else '넘음'}")


def do_seed_check(run_dir: Path) -> str:
    import seed_check
    r = seed_check.recheck()
    shutil.copy(REPORTS / f"seed_check_{R.VERSION}.md", run_dir)
    s = r["recheck"]
    return (f"{R.VERSION} 시드 {r['seed']} 정책 {won(s['profit'])}/수신거부 {s['opt_outs']:.1f}명 · "
            f"허용 기준 {'지킴' if s['success'] else '넘음'} · 다시 확인 {'통과' if r['passed'] else '실패'}")


def run(choice: str, memo: str) -> None:
    name, label = MENU[choice]
    stamp = datetime.now()
    run_dir = RUNS / f"{stamp:%m%d_%H%M%S}_{name}"
    run_dir.mkdir(parents=True)
    for f in SNAPSHOT_FILES:
        shutil.copy(f, run_dir)
    print(f"\n▶ {label} 실행 중... (기록 폴더: {run_dir})")
    if name == "sanity":
        result = do_sanity(run_dir)
    elif name in ("train_trial", "train"):
        result = do_train(run_dir, trial=(name == "train_trial"))
    elif name == "seed_check":
        result = do_seed_check(run_dir)
    else:
        result = do_eval(run_dir)
    (run_dir / "result.txt").write_text(result + ("\n메모: " + memo if memo else "") + "\n", encoding="utf-8")
    new = not HISTORY.exists()
    with HISTORY.open("a", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        if new:
            w.writerow(["date", "kind", "reward_version", "success_metric", "memo", "run_dir"])
        w.writerow([f"{stamp:%Y-%m-%d %H:%M:%S}", name, R.VERSION, result, memo, run_dir.as_posix()])
    import export_timeline
    export_timeline.export()   # 화면 10단계 변경 흐름을 새로 만듦
    print(f"✔ 완료: {result}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("choice", nargs="?")
    ap.add_argument("--memo", default="")
    a = ap.parse_args()
    choice, memo = a.choice, a.memo
    if choice is None:
        print("=== 론팟 마포 상권 강화학습 실행 메뉴 ===")
        for k, (_, label) in MENU.items():
            print(f"  {k}. {label}")
        print("  0. 종료")
        choice = input("번호를 고르세요: ").strip()
        if choice not in MENU:
            return
        memo = input("메모 (엔터 = 없음): ").strip()
    run(choice, memo)


if __name__ == "__main__":
    main()
