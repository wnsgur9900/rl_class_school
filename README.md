# 론팟(LaunPot) 마포 상권 — 푸시 알림 결정 강화학습

서울미디어대학원대학교 「심층강화학습기술의 이해와 응용」 프로젝트.
위치 기반 코인세탁방 기기 현황 공유 앱에서 **유저마다 매일 18:00에 어떤 푸시를 보낼지**(안 보냄 · 일반 알림 · 300P 리워드 미션)를
강화학습(PPO)으로 정한다. 실제 매장 데이터 없이 공개 데이터(실제 날씨·공휴일, 가상 페르소나)와 가정값으로 만든 가상 매장에서 학습·검증한다.

- 설계 문서: [docs/project_spec.md](docs/project_spec.md) (매장 정의 · 반응 규칙 · 고객 성향 · 설계 카드 · 보상 · 변경 기록)
- 발표용 정리: [docs/presentation_notes.md](docs/presentation_notes.md)
- 결과 글: [docs/reports/](docs/reports/) (환경 점검 · 버전별 평가 · 진단 · 정책 해석)
- 성공 척도: 90일 누적 순이익(제보 데이터 가치 − 리워드 비용, 고객 100명 합) 극대화 · 허용 기준: 수신거부 10명 이하

## 설치 (다른 PC에서)

1. [uv](https://docs.astral.sh/uv/) 설치
2. 이 폴더에서 `uv sync` — Python 3.12와 라이브러리(gymnasium, stable-baselines3, torch 등)를 설치

## 실행

`run.bat`을 더블클릭하면 번호 메뉴가 뜬다 (또는 `uv run python src/run.py`).

| 번호 | 하는 일 | 결과 |
| --- | --- | --- |
| 1 | 가상 데이터 생성과 환경 타당성 검증 (시험 규칙 4개) | `docs/reports/sanity_check.md`, `results/logs/`, `public/data/` |
| 2 | 학습 — 시험 학습 (1분 안쪽) | 화면에 스텝당 시간 |
| 3 | 학습 — 본 학습 (400만 스텝, 약 8~10분) | `results/models/<버전>/`, `results/curves/` |
| 4 | 학습된 정책 평가와 진단 | `docs/reports/summary_<버전>.md`, `docs/reports/diagnosis_<버전>.md`, `results/diagnosis/`, `public/data/` |

실행할 때마다 `results/runs/<날짜_시각_이름>/`에 그때의 결과와 `src/env.py`·`src/config.py`·`src/reward.py` 사본이 쌓이고,
`results/run_history.csv`에 한 줄씩 기록된다.

입력 데이터를 다시 만들 때만: `uv run python src/make_personas.py`, `uv run python src/make_external_factors.py`, 검사는 `uv run python src/check.py`.

## 화면으로 보기

- 웹: **https://wnsgur9900.github.io/rl_class_school/** (어디서나 접속, `main`에 올리면 자동으로 새로 바뀜 — `.github/workflows/pages.yml`)
- 내 PC: `public/index.html`을 더블클릭하면 브라우저에서 열린다 (인터넷·서버 필요 없음).
0~10단계를 차례로 넘기며 문제 정의 → 가상 고객 → 반응 규칙(직접 해 보기) → 학습 전 점검 → AI 학습 → 결과 재생 → 진단·보상 고치기(v1 → v2 → v3) → AI가 배운 규칙 → 파일 지도 → 그림으로 보기(저녁 6시 마포구 장면, 전략·계절·상황 선택)를 본다.
화면 데이터 `public/data/viewer_data.js`는 실행 메뉴 1번·4번이 끝날 때 자동으로 새로 만들어진다.

## 폴더 구성

| 경로 | 내용 |
| --- | --- |
| `run.bat` | 더블클릭하면 뜨는 실행 메뉴 |
| `docs/` | `project_spec.md`(기준 명세) · `presentation_notes.md`(발표 정리) · `screen_design.md`(강의 제공 화면 기준) |
| `docs/reports/` | 실행이 만드는 결과 글: 환경 점검 · 버전별 평가 · 진단 · 정책 해석 |
| `src/` | 파이썬 코드: `run.py`(메뉴) · `config.py`(문제 정의 값) · `env.py`(가상 환경) · `reward.py`(지금 보상) · `train.py`(학습) · `simulate.py`(평가) · `diagnose.py`(진단) · `check.py`(성향 검사) · `make_*.py`(입력 데이터) · `paths.py`(폴더 위치) |
| `src/reward_history/` | 앞 보상 버전 보관 (`reward_v1.py`, `reward_v2.py`) |
| `data/` | `personas.csv`(가상 고객 100명) · `customer_traits.csv`(성향 수치) · `external_factors.csv`(날짜별 날씨·공휴일) |
| `results/` | `models/`(학습된 AI) · `logs/`(날짜별 기록) · `curves/`(학습 곡선) · `diagnosis/`(진단 그림·숫자) · `experience/`(경험 표본) · `screenshots/` · `runs/`(내 PC 실행 기록) · `run_history.csv` |
| `public/` | 단계별 둘러보기 화면 `index.html`과 재생용 데이터 `public/data/` |

## 데이터 출처

- 가상 고객: NVIDIA, [Nemotron-Personas-Korea](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea) (CC BY 4.0) — 실제 인물 아님
- 날씨: [Open-Meteo Historical Weather API](https://open-meteo.com/) (CC BY 4.0), 서울 마포구청 좌표, 2016~2025년 일별
- 공휴일: [python-holidays](https://github.com/vacanza/holidays) 한국 공휴일 (대체공휴일 포함)
- 반응 규칙의 숫자는 모두 가정값 (docs/project_spec.md 2절) — 결과 해석은 설계의 타당성 확인에 한정
