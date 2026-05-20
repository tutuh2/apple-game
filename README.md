# 🍎 Apple Game RL

일본 ゲーム菜園의 **フルーツボックス**를 강화학습으로 풀어보는 개인 학습 프로젝트.

> 사과 격자에서 합이 10인 직사각형을 골라 제거하는 게임. 인간 평균 ~140점, 최대 170점.
> 이걸 RL로 풀어내려고 했더니 생각보다 어렵다. 그 시행착오를 그대로 기록한다.

---

## 지금 상태 (2026-05-19)

```
휴리스틱
  random              평균 103.43  ← 무작위
  greedy_largest      평균  97.24  ← 큰 영역 먼저
  greedy_smallest     평균 113.50  ← 작은 영역 먼저 ★
  lookahead_greedy    평균 109.30  ← 1-step 미래 보기 (느리고 약함)

RL (5번 시도, 모두 113 못 깸)
  v1 PPO MLP            ~103   ← 70%에서 학습 중단
  v2 PPO CNN             102.90 ← 200k 완주, 실패
  v4 PointerNet          110.21 ← K-pointer 정책
  v4 + MCTS (N=200)      111.20 ← AlphaZero 스타일
  v5 PointerNet+lookahead   —   ← 16k에서 학습 불능

검색 알고리즘
  uniform MCTS (rollout 유/무)  105.20  ← PUCT q 스케일 함정
  Beam search W=1000             99     ← 그리디 함정, random보다 약함
```

**최종 결론**: 어떤 도구로도 113.50을 의미 있게 못 넘김. greedy_smallest가
사실상 게임 천장. **Step 3은 GUI 시연으로 완료** (`scripts/play_gui.py`) —
실제 사이트는 봇 차단 + canvas 렌더링으로 보류, 같은 정책을 향후 적용 가능.

자세한 시행착오는 [docs/PROGRESS.md](docs/PROGRESS.md) 참고.

---

## 환경

- macOS, Mac mini M2, 16GB RAM
- Python 3.10.14
- GPU 없음 (학습은 CPU/MPS)

## 설치

```bash
pip install -r requirements.txt
```

주요 의존성: `numpy`, `gymnasium`, `stable-baselines3`, `sb3-contrib`, `torch`, `pytest`.

## 명령어

```bash
# 테스트 (97개)
pytest

# 사람이 직접 플레이 (콘솔)
python3 scripts/play_console.py --seed 42

# AI 자동 플레이 시연 (pygame GUI, greedy_smallest)
python3 scripts/play_gui.py --seed 0 --delay 200
python3 scripts/play_gui.py --no-auto --seed 42       # 사람이 마우스로

# 실제 사이트 자동 플레이 (tkinter GUI + 화면 캡처 + 마우스 드래그)
python3 scripts/auto_player.py

# 휴리스틱 100판 벤치마크 (random/greedy_largest/greedy_smallest/lookahead_greedy)
python3 scripts/benchmark_heuristics.py

# RL 학습 (CPU, 200k step, 약 25분)
python3 -m agent.train --steps 200000 --device cpu --seed 0

# 학습 모델 평가
python3 scripts/eval_ppo.py models/ppo_fruitbox_cnn_shaped_200000.zip --episodes 100

# PointerNet (v4) 학습/평가
python3 -m agent.train_pointer --steps 200000 --device cpu --seed 0
python3 scripts/eval_pointer.py models/pointer_v1_shaped_200000.pt --episodes 100

# MCTS 평가 (PointerNet prior)
python3 scripts/eval_mcts.py models/pointer_v1_shaped_200000.pt --simulations 200

# Uniform MCTS (모델 없이, 선택적 rollout)
python3 scripts/eval_mcts_uniform.py --episodes 10 --simulations 200 --rollout

# Beam search 평가
python3 scripts/eval_beam.py --episodes 5 --beam-width 50
```

---

## 어떤 게임인가

- 10행 × 17열 격자, 각 칸에 1~9 사과 (총 170개)
- 직사각형 드래그 → 영역 내 사과 합이 **정확히 10** 이면 제거, 점수 += 제거된 사과 수
- 사과는 한 번 제거되면 복구 안 됨. 더 이상 둘 수 없으면 끝.
- 최대 가능 점수: **170**

**전략 관점에서 흥미로운 점**: 눈앞의 큰 점수보다 **보드 모양을 어떻게 보존하느냐**가 총점을 좌우. 그래서 단순 그리디는 무작위보다도 나쁠 수 있다 (`greedy_largest` 97점 < random 103점).

---

## 프로젝트 진행 흐름

### Step 1 ✅ — 시뮬레이터 + 휴리스틱

게임 규칙 그대로 구현하고 규칙 기반 정책 3종으로 베이스라인 측정.

| 정책 | 평균 (100판) |
|---|---|
| random | 103.43 |
| greedy_largest | 97.24 |
| **greedy_smallest** | **113.50** |

`greedy_smallest`(작은 영역 우선)가 가장 강함 → RL이 넘어야 할 선 = **113점**.

### Step 2 ✅ — RL 학습 시도 (5번 실패)

MaskablePPO/PointerNet/MCTS/Beam search 등 다양한 도구로 시도. 결과:

| 시도 | 모델 | Reward | 결과 |
|---|---|---|---|
| v1 | MlpPolicy | 제거된 사과 수 | 103 정체 → 중단 |
| v2 | + Custom CNN | 제거된 사과 수 | 102.90 (실패) |
| v3 | + Custom CNN | shaped (도메인 지식) | 진행 중 상태에서 정체 |
| v4 | PointerNet (K-pointer) | shaped | 110.21 (후보 특성 빈약) |
| v4+MCTS | v4 + PUCT N=200 | — | 111.20 |
| BC | sb3 CnnPolicy masked CE | — | 1 epoch smoke 108.90, 중단 |
| v5 | PointerNet + lookahead feature | shaped | 16k에서 학습 불능 |
| MCTS+rollout | uniform + greedy_smallest rollout | — | 105.20 (PUCT 버그) |
| Beam search | W=10~1000, 누적 score 정렬 | — | 97~99 (그리디 함정) |

핵심 어려움:
- **Action space 8,415개** — mask로 invalid 빼도 valid끼리 분간 어려움
- **장기 신호 희석** — 지금 수가 50수 뒤 점수에 어떻게 기여하는지 신호가 약함
- **value 학습 ≠ policy 학습** — `explained_variance` 0.9+인데도 점수 안 오름
- **MCTS q 스케일 함정** — q=누적점수 단위가 너무 커서 PUCT exploration term이 죽음
- **빔서치 그리디 함정** — 누적 score 정렬은 큰 영역 먼저 먹는 가지를 살림

### Step 3 ✅ — GUI 자동 플레이 시연 (완료)

**`scripts/play_gui.py`** — pygame 기반 GUI로 보드를 사이트와 비슷한 비주얼로
띄우고 greedy_smallest 정책이 자동으로 매 수를 두는 모습을 시각화.

```bash
python3 scripts/play_gui.py --seed 0 --delay 200   # AUTO 모드 (기본)
python3 scripts/play_gui.py --no-auto --seed 42    # 사람이 마우스 드래그
```

### Step 4 ✅ — 실제 사이트 자동 플레이 (완료, 2026-05-20)

**`scripts/auto_player.py`** — tkinter GUI로 [DETECT] → [START] 두 클릭으로
실제 ゲーム菜園 사이트(https://www.gamesaien.com/game/fruit_box_a/)에서
greedy_smallest 정책으로 자동 플레이.

```bash
python3 scripts/auto_player.py
# 1) 브라우저에서 사이트 Play 누르기
# 2) [DETECT] — 격자/템플릿/점수 영역 자동 검출
# 3) [START] — 자동 플레이 시작
# 비상정지: STOP / Esc / Cmd+. / 마우스 좌상단 코너
```

핵심 모듈:
- `agent/board_detector.py` — HSV로 사과 자동 찾아 격자 calibration 생성
- `agent/board_recognizer.py` — 셀별 정규화 마스크 + SSD 매칭으로 1~9 인식
  - `models/site_templates.npz` (11 캡처 × 170셀 학습, 숫자별 ~200 샘플)
  - 학습 시 `trained_cell_size=66px`. 브라우저 줌으로 셀이 더 작으면 자동 업스케일
- `agent/score_reader.py` — 사이트 우상단 그린 점수 OCR (pytesseract, 다중 PSM 시도)
- `scripts/auto_player.py` — 통합 GUI
  - 매 step 캡처 → 인식 → 액션 결정 → 드래그 → 다음 step 캡처가 검증
  - 점수 OCR로 사이트 거부 자동 감지, 실패 액션 cooldown
  - 모든 로그를 `/tmp/auto_player_debug/auto_player.log`에 자동 저장

실측 결과: 사이트 점수와 추정 점수 **거의 0 차이로 동기화** (95~117점, 정책 천장).

#### 왜 사이트 자동화가 까다로웠나

- 봇 감지 → Playwright 자동화는 403 차단. 화면 캡처 + 마우스 자동화로 우회.
- 보드가 **canvas 렌더링** — DOM 못 읽음. 사과 색 마스크 + 셀별 숫자 템플릿 매칭으로 해결.
- 브라우저 줌에 따라 셀 크기 달라짐 — `trained_cell_size` 메타로 자동 업스케일.
- 사이트 거부 감지가 어려웠음 — 인식만으로는 가짜 성공 누적 →
  **점수 OCR이 단조 증가하는지**로 확실히 판정.
- 첫 클릭은 브라우저 포커스 가져오는 데 소모 → **워밍업 클릭** 추가로 해결.

---

## 디렉터리 구조

<!-- AUTO:TREE -->
```
apple/
├── .claude/
├── agent/
│   ├── __init__.py
│   ├── beam_search.py
│   ├── board_detector.py
│   ├── board_recognizer.py
│   ├── heuristics.py
│   ├── mcts.py
│   ├── pointer_mcts.py
│   ├── pointer_net.py
│   ├── ppo_pointer.py
│   ├── score_reader.py
│   ├── train.py
│   ├── train_bc.py
│   └── train_pointer.py
├── data/
│   ├── site_captures/
│   │   ├── capture_01_initial.png
│   │   ├── capture_02_smaller.png
│   │   ├── 스크린샷 2026-05-20 오후 12.25.29.png
│   │   ├── 스크린샷 2026-05-20 오후 12.25.38.png
│   │   ├── 스크린샷 2026-05-20 오후 12.25.43.png
│   │   ├── 스크린샷 2026-05-20 오후 12.25.48.png
│   │   ├── 스크린샷 2026-05-20 오후 12.25.52.png
│   │   ├── 스크린샷 2026-05-20 오후 12.25.57.png
│   │   ├── 스크린샷 2026-05-20 오후 12.26.02.png
│   │   ├── 스크린샷 2026-05-20 오후 12.26.06.png
│   │   ├── 스크린샷 2026-05-20 오후 12.26.10.png
│   │   └── 스크린샷 2026-05-20 오후 12.26.15.png
│   └── imitation_greedy_smallest_1000.npz
├── docs/
│   ├── PROGRESS.md
│   └── SESSION_NOTES.md
├── env/
│   ├── __init__.py
│   ├── action_space.py
│   ├── fruit_box.py
│   └── fruit_box_gym.py
├── scripts/
│   ├── __init__.py
│   ├── _log_eval.py
│   ├── _update_meta.py
│   ├── auto_player.py
│   ├── benchmark_heuristics.py
│   ├── calibrate_screen.py
│   ├── eval_bc.py
│   ├── eval_beam.py
│   ├── eval_mcts.py
│   ├── eval_mcts_uniform.py
│   ├── eval_pointer.py
│   ├── eval_ppo.py
│   ├── generate_imitation_data.py
│   ├── play_console.py
│   ├── play_gui.py
│   └── play_site.py
├── tests/
│   ├── __init__.py
│   ├── test_action_space.py
│   ├── test_beam_search.py
│   ├── test_board_detector.py
│   ├── test_board_recognizer.py
│   ├── test_fruit_box.py
│   ├── test_fruit_box_gym.py
│   ├── test_imitation_data.py
│   ├── test_mcts.py
│   └── test_pointer_net.py
├── calibration.json
├── CLAUDE.md
├── README.md
└── requirements.txt
```
<!-- /AUTO:TREE -->

## 상태

<!-- AUTO:STATS -->
- 테스트: **97개** (`pytest`)
- 학습 산출물: **7개** in `models/`

  | 파일 |
  |---|
  | `pointer_v1_shaped_200000.pt` |
  | `pointer_v1_shaped_50000.pt` |
  | `ppo_fruitbox_10000.zip` |
  | `ppo_fruitbox_cnn_10000.zip` |
  | `ppo_fruitbox_cnn_200000.zip` |
  | `ppo_fruitbox_cnn_shaped_10000.zip` |
  | `site_templates.npz` |
<!-- /AUTO:STATS -->

---

## 배우고 있는 것

이 프로젝트는 "RL이 작동한다"를 보여주려는 게 아니라, **RL을 실제 문제에 적용할 때 마주치는 어려움**을 직접 겪어보고, 결국 안 되면 휴리스틱으로 돌아가는 과정을 정직하게 기록하는 것이 목적.

직접 부딪힌 것들:

- PPO 학습 로그 읽는 법 (`ep_rew_mean`, `explained_variance`, `entropy_loss`, `clip_fraction`)
- Value head와 Policy head가 따로 학습된다는 사실 (explained_variance 0.9+여도 정책 정체)
- Action space가 커지면 mask로도 한계 — 8,415개 액션 중 valid 끼리 분간 어려움
- Reward shaping이 약이 될 수도, 독이 될 수도 있음
- PointerNet으로 action space 다이어트했지만 후보 특성 빈약하면 효과 없음
- 강한 prior 신호 추가 → 정책 결정론화 → PPO advantage 0 → 학습 불능
- 단순 1-step lookahead가 의외로 약함 (greedy_smallest의 미묘한 효과를 못 잡음)
- MCTS도 leaf value 없으면 deep search 의미 없음 + q 스케일 정규화 안 하면 PUCT 죽음
- 빔서치도 정렬 키가 단순 누적 score면 그리디 함정 (random보다 약함)
- **deterministic + perfect info + 단일 플레이어 게임에서는 RL이 본질적으로 어색**
- **단순한 휴리스틱이 의외로 강하다는 것 + 그 천장은 단단하다는 것**

5개 알고리즘 + 검색 2종까지 시도하고 결국 휴리스틱으로 돌아온 과정 자체가 학습.
자세한 진단은 [docs/PROGRESS.md](docs/PROGRESS.md).

---

## 다른 컴퓨터에서 이어서 작업하기

```bash
git clone https://github.com/tutuh2/apple-game.git
cd apple-game

# Python 3.10+ 권장
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 사이트 자동 플레이용 OCR 바이너리
brew install tesseract              # macOS
# sudo apt install tesseract-ocr    # Linux

# 테스트로 환경 OK 확인 (97개 통과)
pytest -q
```

큰 RL 체크포인트(`models/*.pt`, `*.zip`)는 git에서 제외됨 — 필요하면 다시 학습:

```bash
python3 -m agent.train --steps 200000 --device cpu --seed 0
```

사이트 자동 플레이는 `models/site_templates.npz`(템플릿)만 있으면 바로 실행:

```bash
python3 scripts/auto_player.py
```

`docs/SESSION_NOTES.md`에 마지막 세션의 상태 + 다음 작업 메모가 있으니
그것부터 읽으면 됨.

---

## 라이선스 / 사용

개인 학습용 프로젝트. 자유롭게 참고하셔도 됩니다.
