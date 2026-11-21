"""외부 요인 표 만들기 — 날짜마다 한 줄 (2016-01-01 ~ 2025-12-31).

- 날씨: Open-Meteo 과거 날씨(Historical Weather API) 실제 기록, 마포구청 좌표
- 공휴일: holidays 라이브러리의 한국 공휴일(대체공휴일 포함)
- 요일·주말·월말: 날짜에서 계산

실행: uv run python src/make_external_factors.py  →  data/external_factors.csv
출처: Open-Meteo Historical Weather API (CC BY 4.0), holidays (python-holidays)
"""
import json
import urllib.parse
import urllib.request

import holidays
import pandas as pd

from config import DATA_END, DATA_START, MONTH_END_START_DAY, STORE_LAT, STORE_LON
from paths import DATA

OUT = DATA / "external_factors.csv"


def load_weather() -> pd.DataFrame:
    q = urllib.parse.urlencode({
        "latitude": STORE_LAT, "longitude": STORE_LON,
        "start_date": DATA_START, "end_date": DATA_END,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum",
        "timezone": "Asia/Seoul",
    })
    d = json.load(urllib.request.urlopen(f"https://archive-api.open-meteo.com/v1/archive?{q}", timeout=120))["daily"]
    return pd.DataFrame({
        "date": pd.to_datetime(d["time"]),
        "temp_max": d["temperature_2m_max"],
        "temp_min": d["temperature_2m_min"],
        "precipitation_mm": d["precipitation_sum"],
        "snowfall_cm": d["snowfall_sum"],
    })


def main() -> None:
    df = load_weather()
    kr = holidays.KR(years=range(df["date"].dt.year.min(), df["date"].dt.year.max() + 1))
    df["day_of_week"] = df["date"].dt.day_of_week           # 0=월 ~ 6=일
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"] = df["date"].dt.date.map(lambda d: int(d in kr))
    df["holiday_name"] = df["date"].dt.date.map(lambda d: kr.get(d, ""))
    df["is_month_end"] = (df["date"].dt.day >= MONTH_END_START_DAY).astype(int)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    missing = df[["temp_max", "temp_min", "precipitation_mm"]].isna().sum().sum()
    print(f"{len(df)}일 저장 → {OUT} (날씨 빈칸 {missing}개)")


if __name__ == "__main__":
    main()
