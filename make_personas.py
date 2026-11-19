"""가상 고객 100명 추출 — Nemotron-Personas-Korea에서 1 매장 정의의 위치·주 고객 조건으로 거른다.

전체(약 2GB)를 내려받지 않는다.
1) duckdb로 원격 parquet에서 거르기용 칸만 읽어 후보를 만든다.
2) 시드를 고정해 100명을 뽑는다.
3) 뽑힌 100명의 줄만 골라 소개글 칸을 읽는다.

실행: uv run python make_personas.py  →  data/personas.csv
출처: NVIDIA, Nemotron-Personas-Korea (CC BY 4.0)
"""
import json
import urllib.request
from pathlib import Path

import duckdb
import pandas as pd

from config import AGE_MAX, AGE_MIN, NUM_CUSTOMERS, PERSONA_DISTRICT, PERSONA_FAMILY_TYPES, PERSONA_SEED

DATASET = "nvidia/Nemotron-Personas-Korea"
HF_API = "https://datasets-server.huggingface.co"
TEXT_COLUMNS = [
    "persona", "professional_persona", "family_persona",
    "culinary_persona", "sports_persona", "hobbies_and_interests",
]
OUT = Path("data/personas.csv")


def load_candidates(con, urls) -> pd.DataFrame:
    family = ", ".join(f"'{f}'" for f in PERSONA_FAMILY_TYPES)
    parts = []
    for file_idx, url in enumerate(urls):
        parts.append(con.execute(f"""
            SELECT uuid, {file_idx} AS file_idx, file_row_number AS row_in_file
            FROM read_parquet('{url}', file_row_number=true)
            WHERE district = '{PERSONA_DISTRICT}'
              AND age BETWEEN {AGE_MIN} AND {AGE_MAX}
              AND family_type IN ({family})
              AND (occupation <> '무직' OR age <= 29)
        """).df())
    return pd.concat(parts, ignore_index=True).sort_values(["file_idx", "row_in_file"], ignore_index=True)


def main() -> None:
    files = json.load(urllib.request.urlopen(f"{HF_API}/parquet?dataset={DATASET}", timeout=60))["parquet_files"]
    urls = [f["url"] for f in files]
    con = duckdb.connect()
    cand = load_candidates(con, urls)
    picked = cand.sample(n=NUM_CUSTOMERS, random_state=PERSONA_SEED).sort_values(["file_idx", "row_in_file"])
    cols = "uuid, sex, age, occupation, family_type, housing_type, marital_status, " + ", ".join(TEXT_COLUMNS)
    parts = []
    for file_idx, grp in picked.groupby("file_idx"):
        rows = ", ".join(str(int(r)) for r in grp["row_in_file"])
        parts.append(con.execute(f"""
            SELECT {cols} FROM read_parquet('{urls[file_idx]}', file_row_number=true)
            WHERE file_row_number IN ({rows}) ORDER BY file_row_number
        """).df())
    out = pd.concat(parts, ignore_index=True)
    assert list(out["uuid"]) == list(picked["uuid"]), "뽑은 사람과 읽은 사람이 다름"
    out.insert(0, "customer_id", range(1, len(out) + 1))
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"후보 {len(cand)}명 중 {len(out)}명 저장 → {OUT}")


if __name__ == "__main__":
    main()
