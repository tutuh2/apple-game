# フルーツボックス RL Auto-Player

## 프로젝트 목표

일본 사이트 ゲーム菜園의 **フルーツボックス (Apple Game)** 을 자동으로 플레이해 고득점을 기록하는 강화학습 AI 개발.

게임 룰:
- 10행 × 17열 격자, 각 칸에 1~9 사과 (총 170개)
- 드래그로 직사각형 영역 선택 → 영역 내 사과 합이 **정확히 10**이면 제거, 점수 +(제거된 사과 수)
- 제한 시간 120초, 사과는 한 번 제거되면 복구 불가

## 환경

- **하드웨어**: Mac mini M2 / 16GB RAM / 8코어 (4P+4E)
- **GPU**: 없음, PyTorch MPS 백엔드 활용 예정
- **Python**: 3.10.14 (pyenv)
- **사용자**: RL 초보, 학습 과정 이해 우선

## 진행 방식

**Step 1 ✅ 완료** (2026-05-19): 시뮬레이터 + 베이스라인 휴리스틱
**Step 2 🔧 구축 완료** (2026-05-19): Gym wrapper + MaskablePPO 학습 파이프라인. 본 학습(200k step)은 사용자가 실행.
**Step 3**: Playwright로 실제 브라우저 자동 플레이 (선택)

## Step 1 결과 (100판 평균, 최대 가능 170점)

| 정책 | 평균 | 중앙값 | min | max |
|---|---|---|---|---|
| random            | 103.43 | 103.0 |  61 | 138 |
| greedy_largest    |  97.24 |  98.0 |  67 | 124 |
| **greedy_smallest** | **113.50** | 114.0 |  62 | 144 |

**핵심 인사이트:**
- greedy_largest(큰 영역 우선)가 random보다 **나쁨** — 보드를 빨리 비워서 후속 옵션이 사라짐
- greedy_smallest(2개씩 작게)가 **베이스라인 최강** — 보드 모양 보존
- 이게 정확히 **RL이 빛날 영역**: 단순 규칙으로 안 풀리는 장기 시야 문제

## Step 2 목표

| 점수대 | 평가 |
|---|---|
| < 113 | 실패 (greedy_smallest도 못 이김) |
| 113~135 | 미흡 (휴리스틱은 이겼지만 의미 작음) |
| **140~160** | **타겟** (사람 평균 ~ 사람 상급) |
| 160+ | 우수 (사람 만점급) |

## Step 2 산출물 (2026-05-19 구축 완료)

| 파일 | 역할 |
|---|---|
| `env/action_space.py` | 직사각형 ↔ flat int 인덱스(0..8414), action mask 생성 |
| `env/fruit_box_gym.py` | `FruitBoxEnv` Gymnasium wrapper (obs (1,10,17) float32, Discrete(8415)) |
| `agent/train.py` | MaskablePPO 학습 진입점, device auto-detect |
| `scripts/eval_ppo.py` | 저장 모델 N판 평가 + 휴리스틱 비교 |
| `tests/test_action_space.py` | encode/decode round-trip, mask 정확성 |
| `tests/test_fruit_box_gym.py` | env reset/step/mask/termination |

**Sanity check (10k step, 100판 평가):** 평균 103.81 — 본 학습(200k+) 필요.

## 본 학습 실행

```bash
# 200k 스텝, M2 MPS 자동 감지 (작은 MLP는 CPU가 더 빠를 수도)
python3 -m agent.train --steps 200000 --seed 0 --device auto

# 평가 (100판 휴리스틱과 같은 시드 범위)
python3 scripts/eval_ppo.py models/ppo_fruitbox_200000.zip --episodes 100
```

**목표:** 평균 113점(greedy_smallest) 초과 → 140~160점이 타겟. 못 넘으면 하이퍼파라미터 튜닝 (learning_rate, n_steps, ent_coef) 또는 CNN policy로 전환.

## 코딩 원칙

- **단순함 우선**: 추측성 추상화 금지, 한 번만 쓰는 코드에 abstraction 만들지 말 것
- **외과적 변경**: 인접 코드 임의 리팩토링 금지, 요청된 부분만 수정
- **TDD 권장**: 시뮬레이터 같이 핵심 룰 구현부는 pytest로 검증
- **설명 우선**: 사용자가 RL 초보이므로 코드 짤 때마다 "왜 이렇게 짰는지" 짧게 설명
- **커밋은 사용자 요청 시에만**

## 디렉터리 구조

```
apple/
├── env/
│   ├── __init__.py
│   ├── fruit_box.py             ✅ 시뮬레이터 코어 (FruitBox, Action)
│   ├── action_space.py          ✅ encode/decode + compute_action_mask (Step 2)
│   └── fruit_box_gym.py         ✅ Gymnasium Env wrapper (Step 2)
├── agent/
│   ├── __init__.py
│   ├── heuristics.py            ✅ random / greedy_largest / greedy_smallest
│   └── train.py                 ✅ MaskablePPO 학습 진입점 (Step 2)
├── scripts/
│   ├── __init__.py
│   ├── play_console.py          ✅ 사람이 직접 플레이
│   ├── benchmark_heuristics.py  ✅ 100판 벤치마크
│   └── eval_ppo.py              ✅ 학습 모델 평가 (Step 2)
├── tests/
│   ├── __init__.py
│   ├── test_fruit_box.py        ✅ 시뮬레이터 26개 테스트
│   ├── test_action_space.py     ✅ 인코딩/마스크 11개 테스트 (Step 2)
│   └── test_fruit_box_gym.py    ✅ Gym wrapper 9개 테스트 (Step 2)
├── models/                       (학습 후 생성)
├── requirements.txt             ✅ numpy, pytest, gymnasium, sb3-contrib, torch
└── CLAUDE.md                    ✅ 이 파일
```

## 자주 쓰는 명령

```bash
pytest                                                          # 모든 테스트 (46개)
python3 scripts/play_console.py --seed 42                       # 사람이 직접 플레이
python3 scripts/benchmark_heuristics.py                         # 휴리스틱 100판 벤치마크
python3 -m agent.train --steps 200000 --device auto             # 본 학습
python3 scripts/eval_ppo.py models/ppo_fruitbox_200000.zip      # 학습 모델 평가
```

## RL 개념 빠른 참조

| 용어 | 이 게임에서의 의미 |
|---|---|
| Environment | 게임 시뮬레이터 |
| State | 현재 보드 (10×17 격자) |
| Action | 선택할 직사각형 (r1, c1, r2, c2) |
| Reward | 제거된 사과 개수 (합≠10이면 0) |
| Policy | 상태 → 액션 결정 함수 (RL로 학습 대상) |

## 변경 이력

| 날짜 | 변경 내용 | 사유 |
|---|---|---|
| 2026-05-19 | 초기 경량 하네스 (CLAUDE.md만) | 단일 트랙 학습 프로젝트, 풀 하네스 과한 투자 판단 |
| 2026-05-19 | Step 1 완료 + 벤치마크 결과 기록 | 시뮬레이터·테스트·휴리스틱 3종 완성, RL 목표선 113점 확정 |
| 2026-05-19 | Step 2 파이프라인 구축 | Gym wrapper + MaskablePPO 학습/평가 스크립트, 46개 테스트 통과. 본 학습은 사용자가 실행 |
