"""변경 흐름 — 명세 6절 변경 기록과 실행 기록을 화면용 파일 하나로 묶는다.

읽는 것: docs/project_spec.md 6절, results/run_history.csv
만드는 것: public/data/timeline.js (화면 10단계 · 0단계 '최근 바뀐 것')

언제 만들어지나: 실행 메뉴가 끝날 때마다 자동, GitHub에 올려 웹에 배포할 때마다 자동.
명세만 고쳤을 때 내 PC 화면에 바로 보려면: uv run python src/export_timeline.py
파이썬 기본 기능만 써서 웹 배포 서버에서도 라이브러리 설치 없이 돈다.
"""
import csv
import json
import re

from paths import PUBLIC_DATA, RESULTS, SPEC

OUT = PUBLIC_DATA / "timeline.js"
HISTORY = RESULTS / "run_history.csv"
TAG_HEAD = 40                    # 분류할 때 볼 '바뀐 것'의 앞부분 글자 수  # 가정
# '바뀐 것' 앞부분에 이 말이 있으면 그 분류 (위에서부터 확인)
TAGS = [("정리·배포", ("폴더", "프로젝트 이름", "배포", "실행 파일")),
        ("화면", ("화면",)),
        ("보상·학습", ("보상 v", "학습", "진단", "정책 해석", "다시 확인"))]
DEFAULT_TAG = "설계·환경"
KINDS = {"sanity": "환경 점검", "train_trial": "시험 학습", "train": "본 학습", "eval": "평가·진단", "seed_check": "다시 확인"}
RESULT = re.compile(r"(v\d+)(?: 시드 (\d+))? 정책 ([\d,]+)원/수신거부 ([\d.]+)명")


def tag_of(text: str) -> str:
    head = re.sub(r"^\([^)]*\)\s*", "", text)[:TAG_HEAD]
    return next((name for name, words in TAGS if any(w in head for w in words)), DEFAULT_TAG)


def changes() -> list[dict]:
    """명세 6절 표의 줄을 그대로 옮긴다 (날짜 · 바꾼 것 · 이유 · 결과)."""
    section = SPEC.read_text(encoding="utf-8").split("## 6 변경 기록", 1)[1]
    out = []
    for line in section.splitlines():
        if not re.match(r"^\| \d{4}-\d{2}-\d{2} \|", line):
            continue
        date, what, why, result = [c.strip() for c in line.strip().strip("|").split("|")][:4]
        stage = re.match(r"^\(([^)]*)\)", what)
        title = re.split(r" — | · ", re.sub(r"^\([^)]*\)\s*", "", what), maxsplit=1)[0]
        if title.count("(") > title.count(")"):      # 괄호 안에서 잘렸으면 괄호 앞까지만
            title = title[:title.rfind("(")].strip()
        out.append({"date": date, "stage": stage.group(1) if stage else "", "title": re.sub(r"[`*]", "", title),
                    "tag": tag_of(what), "what": what, "why": why, "result": result})
    return out


def runs() -> tuple[list[dict], list[dict]]:
    """실행 기록 전체와, 그중 평가 결과만 뽑은 성적 흐름 (같은 결과를 다시 평가한 줄은 하나로)."""
    if not HISTORY.exists():
        return [], []
    with HISTORY.open(encoding="utf-8-sig", newline="") as fp:
        rows = list(csv.DictReader(fp))
    steps, flow, log = {}, [], []
    for r in rows:
        kind, ver, text, memo = r["kind"], r["reward_version"], r["success_metric"], r["memo"]
        log.append({"date": r["date"], "kind": KINDS.get(kind, kind), "version": ver, "result": text, "memo": memo})
        trained = re.match(r"([\d,]+)스텝", text)
        if kind == "train" and trained:
            steps[ver] = int(trained.group(1).replace(",", ""))
        found = RESULT.search(text)
        if kind not in ("eval", "seed_check") or not found:
            continue
        man = re.search(r"(\d+)만 스텝", memo)
        item = {"date": r["date"][:10], "version": found.group(1), "kind": kind, "seed": int(found.group(2)) if found.group(2) else None,
                "profit": int(found.group(3).replace(",", "")), "opt_outs": float(found.group(4)), "ok": "지킴" in text,
                "steps": steps.get(ver) or (int(man.group(1)) * 10_000 if man else None), "memo": memo}
        same = flow and all(flow[-1][k] == item[k] for k in ("version", "kind", "profit", "opt_outs"))
        if not same:
            flow.append(item)
    return flow, log


def export() -> dict:
    flow, log = runs()
    data = {"changes": changes(), "flow": flow, "runs": log}
    OUT.write_text("window.TIMELINE_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    return data


if __name__ == "__main__":
    d = export()
    print(f"변경 기록 {len(d['changes'])}줄 · 성적 흐름 {len(d['flow'])}개 · 실행 기록 {len(d['runs'])}줄 → {OUT}")
