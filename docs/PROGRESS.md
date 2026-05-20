# Apple Game RL — 진행 기록

> 일본 ゲーム菜園의 **フルーツボックス**를 RL로 푸는 개인 학습 프로젝트.
> 시도하고 실패하고 다시 시도한 과정을 그대로 기록한다.

마지막 업데이트: 2026-05-20 (Step 3 GUI 시연 완료, rejection sampling 시뮬레이터)

---

## 게임 룰 요약

- 10행 × 17열 격자, 각 칸에 1~9 사과 (총 170개)
- 드래그로 직사각형 영역 선택 → **영역 내 사과 합이 정확히 10**이면 제거, 점수 += 제거된 사과 수
- 제한 시간 120초, 사과는 한 번 제거되면 복구 불가
- 최대 가능 점수: 170점

---

## Step 1 — 시뮬레이터 + 휴리스틱 베이스라인

### 산출물

- `env/fruit_box.py` — 게임 시뮬레이터 (룰만, RL 인터페이스 없음)
- `agent/heuristics.py` — 규칙 기반 정책 3종
- `scripts/play_console.py` — 사람이 직접 플레이 가능
- `scripts/benchmark_heuristics.py` — N판 자동 비교

### 휴리스틱 100판 벤치마크 결과

| 정책 | 평균 | 중앙값 | min | max |
|---|---|---|---|---|
| random | 103.43 | 103.0 | 61 | 138 |
| greedy_largest (큰 영역 우선) | 97.24 | 98.0 | 67 | 124 |
| **greedy_smallest (작은 영역 우선)** | **113.50** | 114.0 | 62 | 144 |

### 핵심 인사이트

- **greedy_largest가 random보다 나쁨**. 큰 영역을 먼저 먹으면 보드를 빨리 비워 후속 옵션이 사라짐.
- **greedy_smallest가 가장 강함**. 보드 모양을 보존하면서 차근차근 풀어내는 게 더 좋은 전략.
- 단순 규칙으로는 113점이 천장으로 보임. 장기 시야가 필요한 문제 → RL이 빛날 영역.

---

## Step 2 — RL 학습

### Step 2 목표

| 점수대 | 평가 |
|---|---|
| < 113 | 실패 (greedy_smallest도 못 이김) |
| 113~125 | 약간 우위 (휴리스틱은 이김, 숙련 사용자 평균) |
| **130~150** | **타겟** (잘하는 사람 운 좋을 때 수준) |
| 150+ | 우수 (사람 거의 안 나오는 영역) |

### 공통 설계

- **알고리즘**: MaskablePPO (sb3-contrib). action mask로 invalid 액션 학습 시간 낭비 차단.
- **Observation**: (1, 10, 17) float32, 보드 값 / 9.0.
- **Action space**: Discrete(8415). 직사각형 (r1,c1,r2,c2) → flat int.
  - 행 쌍 55개 × 열 쌍 153개 = 8,415
- **Reward (raw)**: 제거된 사과 수.
- **Termination**: 합=10 직사각형이 없을 때.

### 시도 v1 — MlpPolicy (raw reward)

**설정**:
- policy: `MlpPolicy` (기본 64-64 MLP)
- learning_rate: 3e-4, n_steps: 2048, ent_coef: 0.01
- 200,000 timesteps

**결과**: ❌ 실패. **ep_rew_mean 102~103에서 정체**.

학습 로그 추이:

| 진행률 | ep_rew_mean | explained_variance | entropy_loss |
|---|---|---|---|
| 8k (4%) | 103 | 0.004 | -3.33 |
| 55k (28%) | 103 | 0.884 | -2.95 |
| 94k (47%) | 103 | 0.892 | -2.63 |
| 139k (70%) | 103 | 0.892 | -2.63 |

**진단**: value head는 잘 학습됨 (0.004 → 0.89). 하지만 정책이 **170개 평평한 숫자**로 보드를 보기 때문에 인접 사과 패턴을 못 잡음. policy가 100점 천장을 못 깸.

→ 70% 시점에 학습 중단. CNN으로 전환 결정.

### 시도 v2 — CnnPolicy (raw reward)

**변경 사항**:
- 10×17 보드 전용 작은 CNN feature extractor 추가 (`SmallBoardCNN`)
  - Conv(1→32, k=3, padding=1) → ReLU → Conv(32→64, k=3, padding=1) → ReLU → Flatten → Linear(→128)
- learning_rate: 3e-4 → **1e-4** (CNN 안정성)
- n_steps: 2048 → **4096**

**결과**: ❌ 또 실패. **ep_rew_mean 102.90** (200k 완주).

| 진행률 | ep_rew_mean | explained_variance |
|---|---|---|
| 12k (6%) | 105 | 7e-05 |
| 200k (100%) | 102 | **0.95** |
| eval 20판 (raw) | **102.90 ± 11.95** | — |

**진단**: value 학습은 v1보다 더 잘 됨 (0.892 → 0.95, value_loss 11.2 → 2.18). 그런데 **policy 점수는 동일**. 즉 모델 표현력 문제가 아니라 **reward 신호 자체의 문제**.

→ 200k 완주 후 reward shaping으로 전환.

### 시도 v3 — CnnPolicy + Reward Shaping ⏳ 진행 중

**가설**: PPO가 valid 액션 사이에서 어떤 게 "장기적으로 좋은지" 신호가 너무 약함. greedy_smallest가 random보다 +10점 좋았던 도메인 지식을 reward에 명시적으로 주입한다.

**Reward shaping 두 가지**:
1. **Small-region bonus**: 직사각형 크기 ≤ 4칸이면 reward × 1.5
   - greedy_smallest 전략을 학습 신호로 변환
2. **Terminal bonus**: 게임 종료 시 최종 score × 0.05 추가
   - 더 길게 두는 정책 선호

**중요**: 평가는 항상 raw 모드(게임 실제 점수)로 측정. shaped는 학습 신호 전용.

**진행 중 로그** (200k 학습, shaped):

| 진행률 | ep_rew_mean (shaped) | ep_len_mean | explained_variance |
|---|---|---|---|
| 14% (28k) | 141 | 43.0 | 1e-06 |
| 37% (73k) | 143 | 43.6 | 0.787 |
| ... | ... | ... | ... |

**현재 우려**: shaped와 raw는 값 범위가 달라 v2(102)와 v3(141)을 직접 비교 불가. ep_rew_mean이 141→143으로 거의 안 움직이는 게 v1/v2의 정체 패턴과 비슷해 보임. 50% 시점에서 추세 재판단 예정.

→ **다음 확인 포인트**: 100% 평가 시 raw mean 113 초과 여부.

### 시도 BC — Imitation Learning 트랙

**가설**: v1~v3 PPO가 102 천장에서 시작해 빠져나오지 못함. greedy_smallest(113.50)를
behavior cloning으로 복제하면 정책을 113 수준에서 시작시켜 PPO fine-tune의
발판으로 쓸 수 있을까?

**산출물**:
- `scripts/generate_imitation_data.py` — greedy_smallest로 1000판 두며 (obs, action, mask) .npz 저장
- `tests/test_imitation_data.py` — 데이터 무결성 6개 테스트 (action ⊂ mask, 재현성 등)
- `data/imitation_greedy_smallest_1000.npz` — 51,108 샘플, 1000판 평균 112.96
- `agent/train_bc.py` — sb3 MaskablePPO.policy를 그대로 가져와 masked CE로 학습
- `scripts/eval_bc.py` — 학습 zip 100판 평가

**Smoke 결과 (1 epoch, 10판)**: BC 평균 108.90. 학습 시작은 잘 됐지만…

**전환**: 사용자 결단 — "greedy_smallest 베이스를 흉내내는 게 본질 목표 아님". BC는
fine-tune 시작점만 옮길 뿐이고, PPO fine-tune이 113을 깬다는 보장이 없어 BC 트랙
중단. 더 본질적인 옵션 (action space 재설계 / reward 재설계) 검토로 이동.

### 시도 v5 — PointerNet candidate feature 풍부화

**가설**: v4 PointerNet(110)이 그친 이유는 후보 특성이 빈약했기 때문일 수 있다.
- v4 candidate feature 8개 중 7개가 좌표/크기, 1개는 상수(sum_in/10 = 1.0)
- "이걸 두면 보드 모양이 어떻게 변할지"를 모델이 추론할 정보가 부족

**변경**:
- `CANDIDATE_FEATURE_DIM`: 8 → 12
- 추가 4개: unique_apples/9, max_apple/9, min_apple/9, **lookahead_K/200**, delta_K/100
- 핵심은 lookahead — "이 후보를 두면 다음 step에 valid 후보가 몇 개 남는가" 직접 측정

**결과**:
- **Sanity (2048 step, 5판)**: 평균 116.60 — 거의 학습 안 된 모델인데 113 위. lookahead feature 자체가 강한 신호로 보였음.
- **본 학습 (200k step 시도)**: ❌ 16k step까지 돌렸지만 학습 진전 없음.
  - policy_loss ≈ 0, approx_kl ≈ 0, clip_fraction = 0 (PPO가 사실상 정책 업데이트 안 함)
  - entropy 3.39에서 안 움직임 (탐험 없음, 결정적 정책 박힘)
  - ep_rew_mean 146 → 139 오히려 감소

**진단**: lookahead feature가 너무 강해서 초기 모델조차 거의 결정적 정책을 출력 →
액션 분포가 평탄하지 않아 advantage ≈ 0 → 정책 그라디언트 사라짐. PPO가 학습할
신호가 사라진 상태.

→ 16k에서 중단. lookahead feature가 진짜 좋은 신호인지 휴리스틱으로 검증하는
단계로 이동.

### 시도 lookahead_greedy 휴리스틱 — v5 가설 검증

**가설**: v5의 lookahead feature가 PPO 학습은 못 시켰지만, **신호 자체가 강하면**
PPO 없이 단순 휴리스틱으로 113을 깰 수 있어야 한다.

**구현**: 매 step에서 후보 중 "다음 step의 valid 후보 수가 최대"인 것 선택.
동률은 랜덤. `agent/heuristics.py:lookahead_greedy_policy`.

**100판 결과**:

| 정책 | 평균 | 중앙값 | min | max | 시간 |
|---|---|---|---|---|---|
| random | 103.43 | 103.0 | 61 | 138 | 7s |
| greedy_largest | 97.24 | 98.0 | 67 | 124 | 6s |
| **greedy_smallest** | **113.50** | 114.0 | 62 | 144 | 8s |
| lookahead_greedy | **109.30** | 111.0 | 80 | 136 | **965s (16분)** |

**충격적 결과**: lookahead_greedy 109.30 < greedy_smallest 113.50 (−4.20).

**의미**:
1. 1-step lookahead 자체가 약한 신호 — greedy_smallest의 "작은 영역 선호"가 잡는
   미묘한 효과를 못 따라감 (예: 작은 영역 선호는 동시에 큰 사과 보존 효과가 있는데
   lookahead는 그 부분을 놓침)
2. v5 PPO가 학습 못한 진짜 이유 — 학습 신호로 강제 주입한 feature가 사실
   greedy_smallest보다 약함. 정책을 거기로 끌어당겨봤자 113 밑으로 갈 뿐.
3. **이 게임에서 113을 넘으려면 다단(multi-step) 미래를 봐야 한다**는 결론.

### 시도 uniform MCTS — 다단 lookahead 가능성 검증

**가설**: 1-step lookahead가 부족하다면 MCTS의 다단 검색이 113을 깰 수 있는가?
PointerNet prior 없이 (입력 dim 변경으로 v4 .pt와 incompatible) uniform prior로
시뮬레이션만 늘려서 측정.

**산출물**: `scripts/eval_mcts_uniform.py` (model 없이 prior=1/K, value=0).

**N 그리드 결과 (5판 sanity, seed 0..4)**:

| N | 5판 평균 | 시간 |
|---|---|---|
| 200 | 105.20 | 116s |
| 1000 | **105.20** | 168s |

**결정적 발견**: N=200 → N=1000 (5배) 늘려도 **점수 동일**. 시드별 점수도 109, 118,
87, 111, 101로 한 점도 안 바뀜. MCTS가 N=200에서 이미 같은 결정에 수렴됨.

**진단**:
- K(후보 수)가 30~100인데 N=200을 K개에 나누면 자식당 2~7번 — 통계 부족
- 더 큰 문제: **value 신호가 0**. 우리 MCTS는 expand만 하고 leaf value 0으로 backup.
  즉시 reward만 누적될 뿐 깊은 게임 끝 점수가 안 보임 → 깊은 검색이 효과 없음.
- 진짜 MCTS는 leaf에서 **rollout policy**로 게임 끝까지 시뮬레이션해서 value를
  얻어야 함 (AlphaZero 이전 고전 MCTS 방식). 우리 코드엔 없음.

→ MCTS 구조를 더 깊게 만들지(rollout 추가), 아예 다른 접근(빔 서치 등)으로 갈지
결정해야 함.

### 시도 MCTS + rollout — 진짜 고전 MCTS

**가설**: leaf value를 greedy_smallest 게임-끝까지의 점수로 backup하면
113이 V 신호로 들어와 113+를 깰 수 있다.

**산출물**:
- `agent/mcts.py`: `_simulate`에 `rollout_fn` 인자 추가, leaf에서 호출
- `scripts/eval_mcts_uniform.py`: `--rollout` 플래그
- `tests/test_mcts.py`: rollout 단위 테스트 3개

**결과 (5판 N=200, rollout=greedy_smallest)**: 평균 **105.20**.
**rollout 없는 MCTS(105.20)와 한 점도 안 바뀜.** 시드별 점수
109, 118, 87, 111, 101 완전히 동일. 시간만 116초 → 245초로 늘어났을 뿐.

**진단**: `_simulate` 디버깅 결과 — **root에서 200번 시뮬레이션이 단 하나의
후보(인덱스 0)에만 모두 visit됨**. K=58인 보드인데 visit 분포가
[200, 0, 0, ..., 0]. PUCT가 사실상 첫 자식만 미친 듯이 선택.

**근본 원인**: PUCT 공식 `q + c_puct * prior * sqrt(N_total) / (1 + N_child)`에서
- q는 누적 점수 (0~170 단위), 첫 visit 후 100 근처
- prior=1/K=1/58≈0.017, c_puct=1.4 → exploration term ≈ 0.012
- q ≫ U → q 큰 자식만 계속 선택, 다른 자식은 visit=0 영구
- → MCTS가 형식만 돌고 실질은 그리디. uniform/N=1000/rollout 모두 같은 결과.

→ c_puct을 200으로 키워보니 반대 극단(분기 폭발로 18분에 첫 판도 못 끝냄).
→ 정공법은 q를 [0,1]로 정규화하는 것. 하지만 더 본질적 한계 확인 위해 빔서치로.

### 시도 Beam Search — 정공법 검색

**가설**: deterministic + perfect info 게임이라면 RL/MCTS가 안 풀리는 게 알고리즘
한계인지 게임 자체 한계인지 빔서치로 직접 측정한다.

**산출물**:
- `agent/beam_search.py`: 폭 W, 시간 제한 옵션, 중복 보드 제거
- `scripts/eval_beam.py`: 평가 스크립트
- `tests/test_beam_search.py`: 6개 단위 테스트

**seed=0 한 판 W 그리드**:

| W | score | 시간 |
|---|---|---|
| 10 | 98 | 0.77s |
| 50 | 97 | 2.71s |
| 200 | 97 | 10.67s |
| 500 | 99 | 27.36s |
| 1000 | 99 | 52.72s |

**충격적 결과**: W를 100배 늘려도 점수 거의 변동 없음 (97~99). 같은 seed에서
greedy_smallest는 109, random은 100~110. **빔서치가 random보다 못함.**

**진단**: 빔서치의 **그리디 함정**. 정렬 키가 "현재 누적 점수"라 큰 영역 먼저
먹는 가지가 빔에 살아남고, 그 가지가 후속 옵션이 빠르게 마름. greedy_largest가
greedy_smallest보다 약했던 것과 같은 이유.

→ 정렬에 미래 보존 휴리스틱 추가가 정답이지만, 5개 알고리즘 + 빔서치까지
모두 113 근처 천장에 부딪힌 시점. 도구 미흡이 아니라 **현재의 모든 휴리스틱/
학습 신호가 부족**하다는 강한 증거.

---

## 🏁 최종 결론 (2026-05-19)

5개 알고리즘 + 빔서치 + 휴리스틱 검증 후:

| 정책 | 100판 평균 |
|---|---|
| random | 103.43 |
| greedy_largest | 97.24 |
| **greedy_smallest** | **113.50** |
| lookahead_greedy | 109.30 |
| v1/v2 PPO | ~103 |
| v4 PointerNet | 110.21 |
| v4 + MCTS | 111.20 |
| v5 PointerNet | 학습 불능 |
| uniform MCTS (rollout 유/무) | 105.20 |
| Beam search (W=10~1000) | 97~99 |

**결론**: 이 게임에서 **greedy_smallest 113.50이 사실상 천장**. 어떤 도구로도
의미 있는 개선 못 함. 시도 가치 있는 옵션(빔서치 정렬 휴리스틱, q 정규화 MCTS,
n-step lookahead, RL+MCTS 데이터 BC 시나리오)이 더 있지만, 각각 비용이
크고 +5점도 보장 못 하는 상황.

**Step 3은 greedy_smallest로 진행**. Playwright 자동화에 greedy_smallest를
바로 끼움 — 평균 113점. 실제 사람 점수와 비교하면 일반 사용자(~100점 안팎)는
이기지만 잘하는 사람(130점대)에는 못 미치는 수준. 그래도 무엇보다 **즉시
결정 가능 (1수 < 10ms)** 해서 실전 적용 가능.

### 왜 이게 좋은 마무리인가

- **정직한 결과**: "어떤 도구로도 의미 있는 개선이 안 되더라"는 것 자체가 학습
- **재사용 가치**: 시뮬레이터/벤치 인프라/4가지 RL 시도 코드는 다른 게임/문제에 응용 가능
- **실전성**: 113점이 인상적이진 않지만 실제로 작동하는 자동 플레이어
- **학습 가치**: PPO/PointerNet/MCTS/빔서치를 모두 직접 구현하고 한계를 직접 확인

### 향후 시도 가능한 방향 (필요 시)

- **q-정규화 MCTS**: q를 [0,1]로, 합리적 c_puct. 가능성 중.
- **빔서치 정렬 휴리스틱**: 정렬 키에 미래 보존 추가. 가능성 중.
- **이 게임을 떠난다**: RL이 본질적으로 맞는 도메인(상대 있음, stochastic)으로 이동.

---

## 시도 비교 요약표

| 버전 | Policy | Reward | LR | n_steps | 200k eval (raw) | 결론 |
|---|---|---|---|---|---|---|
| v1 | MlpPolicy | raw | 3e-4 | 2048 | ~103 (70%에서 중단) | MLP 한계 |
| v2 | Cnn + SmallBoardCNN | raw | 1e-4 | 4096 | **102.90** | reward 신호 약함 |
| v3 | Cnn + SmallBoardCNN | shaped | 1e-4 | 4096 | TBD | 진행 중 |
| v4 | PointerNet (K-pointer) | shaped | 1e-4 | 2048 | 110.21 | 후보 특성 빈약 의심 |
| v4+MCTS | v4 + PUCT N=200 | — | — | — | 111.20 | MCTS 효과 미미 |
| BC | sb3 CnnPolicy + masked CE | — | 3e-4 | — | 108.90 (1 epoch smoke) | 사용자가 트랙 중단 |
| v5 | PointerNet, 12-feature (lookahead 포함) | shaped | 1e-4 | 2048 | — (16k에서 중단) | policy_loss≈0, 학습 안 됨 |
| MCTS+rollout | uniform prior + greedy_smallest rollout | — | — | — | 105.20 (5판) | PUCT 자식 선택 버그로 무용 |
| Beam search | W=10~1000, 누적 점수 정렬 | — | — | — | 97~99 (seed=0) | 그리디 함정, random보다 약함 |

휴리스틱 + 100판 결과:

| 기준선 | 100판 평균 |
|---|---|
| random | 103.43 |
| greedy_largest | 97.24 |
| **greedy_smallest** | **113.50** ← 사실상 게임 천장 |
| lookahead_greedy | 109.30 (16분, 비싸고 약함) |
| v1 / v2 PPO | ~103 (실패) |
| v4 PointerNet | 110.21 |
| v4 + MCTS (N=200) | 111.20 |
| uniform MCTS (N=200/1000/rollout) | 105.20 (5판, 다 동점) |
| Beam search W=1000 | 99 (seed=0, 그리디 함정) |

---

## 학습 로그 읽는 법 (자습 메모)

| 지표 | 의미 |
|---|---|
| `ep_rew_mean` | 한 판 평균 누적 보상. raw 모드면 = 게임 점수, shaped면 보너스 포함 |
| `ep_len_mean` | 한 판 평균 수 (더 똑똑해지면 늘어남) |
| `explained_variance` | value head가 미래 보상을 얼마나 잘 예측하나 (1에 가까울수록 좋음) |
| `entropy_loss` | 정책의 무작위성 (큰 음수 = 탐험, 0 가까움 = 결정적) |
| `value_loss` | value head 예측 오차 (낮을수록 좋음) |
| `clip_fraction` | PPO clipping 비율 (0.3 이상이면 업데이트가 너무 급격) |
| `iterations × n_steps` | = `total_timesteps` (전체 진행률 가늠) |
| `n_updates` | 가중치 갱신 누적 횟수 (iterations × n_epochs) |

**좋은 학습의 신호**:
- ep_rew_mean이 우상향
- explained_variance가 1에 가까워짐
- entropy_loss가 천천히 0으로 (너무 빠르면 갇힘)
- value_loss와 clip_fraction이 안정

**나쁜 학습의 신호**:
- ep_rew_mean 단조 하락
- entropy_loss가 갑자기 0에 붙음 → policy collapse
- explained_variance가 다시 0으로 떨어짐
- clip_fraction이 0.5 넘어감

---

## 배운 점

1. **RL은 마법이 아님**. 단순 휴리스틱(greedy_smallest 113점)을 넘기는 게 생각보다 어렵다.
2. **Action space 8,415개는 PPO에게 큰 부담**. action mask로 invalid는 빼지만 valid 끼리의 미세 차이를 학습하기 어렵다.
3. **Credit assignment 문제**. 지금 둔 한 수가 50수 뒤 점수에 어떻게 기여하는지 신호가 gamma=0.99로도 희석된다.
4. **Value 학습 ≠ Policy 학습**. v1/v2 모두 value는 잘 배웠는데(explained_variance 0.9+) 정책은 못 깸. value가 좋아진다고 점수가 따라 오르는 게 아님.
5. **MLP vs CNN**. 10×17 같은 작은 보드에선 차이가 dramatic하지 않을 수 있다. v2(CNN)가 v1(MLP) 대비 거의 안 좋아진 게 그 사례.
6. **강한 prior 신호 = PPO 죽음**. v5에서 lookahead feature가 모델 출력을 거의 결정적으로 만들자 advantage가 사라져 학습 신호가 0이 됨. PPO는 정책에 충분한 무작위성이 필요.
7. **1-step lookahead로는 113을 못 깬다**. greedy_smallest의 "작은 영역 선호"가 lookahead가 못 보는 미묘한 효과(예: 큰 사과 보존)를 함께 잡고 있음. 이 게임은 다단 미래(multi-step)를 봐야 함.
8. **MCTS도 value 신호 없이는 깊은 검색이 무용**. expand+즉시 reward만으로 backup하면 N을 5배 늘려도 같은 결정. 진짜 MCTS는 leaf rollout이 필요.
9. **MCTS의 q 스케일 함정**. q=누적 점수(0~170 단위)인데 c_puct=1.4면 exploration term이 q에 압도당해 첫 자식만 영구 선택. 우리 코드의 root visits 분포가 [200, 0, 0, ...]. AlphaZero 표준은 q를 [-1, 1] / [0, 1] 정규화 필요.
10. **빔서치도 그리디 함정**. "현재 누적 점수" 정렬은 greedy_largest 같은 결과. W=1000도 random보다 약함. 정렬 키에 미래 가치 휴리스틱이 필요한데, 그 휴리스틱이 충분히 강하면 검색 없이도 잘 됨 → 결국 휴리스틱 게임.
11. **이 게임이 RL에 안 맞을 수 있음**. deterministic + perfect info + 단일 플레이어 → self-play가 의미 없음. 고전 검색이 적절한 도구이지만 그것도 강한 휴리스틱 없이는 한계.

---

## 다음 단계 — Step 3 진행

**결정**: greedy_smallest로 Step 3 (Playwright 브라우저 자동화) 진입.

이유:
- 5개 RL 시도 + MCTS(2가지) + 빔서치까지 모두 113 근처 천장
- greedy_smallest 113.50이 사실상 게임 천장으로 보임
- 1수 결정 < 10ms로 실전 적용 가능 (게임 제한 120초)
- "RL이 안 됐는데 결국 휴리스틱으로 돌아간다"는 자체가 정직한 결론

**Step 3 작업**:
1. `scripts/play_browser.py` — Playwright로 ゲーム菜園 자동 플레이
2. greedy_smallest 정책을 인메모리 보드 상태에 매 수 적용
3. 브라우저 좌표 ↔ 보드 좌표 매핑

### 향후 시도 가능 (관심 생기면)

| 옵션 | 가능성 | 비용 |
|---|---|---|
| q-정규화 MCTS (q/170, c_puct=1.4) | 중 | 30분 |
| 빔서치 정렬에 미래 가치 휴리스틱 | 중 | 30분 |
| n-step lookahead 휴리스틱 (K^n 폭증) | 낮음 | 1시간+ |
| MCTS+rollout으로 고득점 데이터 → BC | 낮음 | 반나절+ |
| 다른 RL 도메인으로 이동 | — | — |

### 결과가 105~113 (미흡)
- 학습 더 길게 (500k+)
- ent_coef 튜닝 (0.01 → 0.03)
- net_arch 키우기 ([256, 256] head)

### 결과가 105 이하 (실패)
- **Imitation Learning**으로 전환:
  1. greedy_smallest로 데이터 1,000~5,000판 생성
  2. behavior cloning으로 정책 사전학습 (113점에서 시작)
  3. 그 위에 PPO fine-tuning
- 또는 Action space 자체 재설계 (예: 합=10 직사각형만 후보로 좁히기)

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

## 현재 상태

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

## 자주 쓰는 명령

```bash
# 테스트 전체 (51개)
pytest

# 사람이 직접 플레이
python3 scripts/play_console.py --seed 42

# 휴리스틱 100판 벤치마크
python3 scripts/benchmark_heuristics.py

# 본 학습 (shaped reward, CPU, 200k step, 약 25분)
python3 -m agent.train --steps 200000 --device cpu --seed 0

# 평가 (raw 게임 점수로 측정)
python3 scripts/eval_ppo.py models/ppo_fruitbox_cnn_shaped_200000.zip --episodes 100
```

---

## 변경 이력

| 날짜 | 변경 | 결과 |
|---|---|---|
| 2026-05-19 | Step 1: 시뮬레이터 + 휴리스틱 3종 | greedy_smallest 113.50점 |
| 2026-05-19 | Step 2 v1: MlpPolicy + raw reward, 200k | 70% 시점 ep_rew_mean 103 정체, 중단 |
| 2026-05-19 | Step 2 v2: CnnPolicy + raw reward, 200k | 완주, eval 102.90 (실패) |
| 2026-05-19 | Step 2 v3: CnnPolicy + shaped reward, 200k | 진행 중 |
| 2026-05-19 (오후) | BC 트랙: greedy_smallest 1000판 데이터 + sb3 CnnPolicy masked CE | 1 epoch smoke 108.90, 사용자 결단으로 중단 |
| 2026-05-19 (오후) | v5 PointerNet: candidate feature 8→12 (lookahead 포함) | 16k step에서 학습 진전 없음(policy_loss≈0), 중단 |
| 2026-05-19 (오후) | lookahead_greedy 휴리스틱 100판 | 109.30 (greedy_smallest 113.50보다 약함, 16분 소요) |
| 2026-05-19 (오후) | uniform MCTS N=200/1000 5판 sanity | 둘 다 105.20 동점 — value=0이 결정적 한계 |
| 2026-05-19 (저녁) | MCTS+rollout(greedy_smallest) 5판 | 105.20 (rollout 없는 거와 한 점도 안 바뀜) |
| 2026-05-19 (저녁) | MCTS root visits 진단 | [200, 0, ..., 0] — PUCT q 스케일로 첫 자식만 영구 선택 |
| 2026-05-19 (저녁) | Beam search seed=0, W=10..1000 | 97~99 (random보다도 약함, 그리디 함정) |
| 2026-05-19 (저녁) | **최종 결정**: greedy_smallest로 Step 3 진행 | RL/검색 5개 도구 모두 천장 113 근처, 휴리스틱 채택 |
| 2026-05-20 | 보드 분포 검증 (oshizi.com 출처) | 진짜 게임은 보드 합이 항상 10의 배수 (rejection sampling) |
| 2026-05-20 | FruitBox.reset에 rejection sampling 추가 + 100판 재측정 | 휴리스틱 점수 거의 동일 (random 103.59, greedy_smallest 112.98) — 분포 차이는 결과에 영향 없음 |
| 2026-05-20 | 실제 사이트(ゲーム菜園) Playwright 자동화 시도 | 봇 차단으로 403, IP 차단까지 발생 → 사이트 자동화 보류 |
| 2026-05-20 | **Step 3 완료**: pygame GUI 시연 (scripts/play_gui.py) | greedy_smallest 자동 플레이 시각화, 같은 정책을 향후 실제 사이트에도 적용 가능 |
| 2026-05-20 | board_detector + board_recognizer 추가 | 화면 캡처에서 사과 격자 자동 검출 + 1~9 템플릿 매칭 |
| 2026-05-20 | OCR 폐기 후 11장 캡처로 templates 재학습 | `models/site_templates.npz` (숫자별 ~200 샘플), 다른 줌 환경엔 자동 업스케일 |
| 2026-05-20 | scripts/auto_player.py (tkinter 통합 GUI) | DETECT→START 두 클릭, 매 step 캡처+인식+드래그+검증, 워밍업 클릭, Esc/Cmd+. 즉시 중단 |
| 2026-05-20 | agent/score_reader.py — 사이트 점수 OCR | 우상단 그린 점수 다중 PSM 시도, 단조 증가로 사이트 거부 자동 감지 |
| 2026-05-20 | **Step 4 완료**: 실제 사이트 자동 플레이 | 사이트 실제 vs 추정 점수 0 차이 동기화. 점수 95~117 (정책 천장 = greedy_smallest 한계) |
