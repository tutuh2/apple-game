# Session Notes — 사이트 자동 플레이 (Step 3 확장)

> 2026-05-20 세션 종료 시점 작성.
> Step 3 완료 후 "실제 사이트에서도 자동 플레이"를 위해 추가 작업 중이었음.
> 토큰 한도로 세션 초기화 필요 — 다음 세션에서 여기서 이어감.

---

## 목표

`scripts/play_gui.py` (로컬 GUI 시연)는 완성. 이제 **실제 ゲーム菜園 사이트**
(https://www.gamesaien.com/game/fruit_box_a/) 에서 `greedy_smallest`가 자동
플레이하도록 만드는 것.

핵심 요구:
- 사용자 손이 거의 안 가야 함 (마우스 드래그 calibration 같은 거 X)
- 앱 켜고 [DETECT] → [START] 두 클릭으로 끝

---

## 지금까지 만든 것

### 1. `agent/board_detector.py` ✅ 동작
- 화면 캡처에서 빨간 사과 자동 검출 → 격자 좌표 자동 추출
- HSV 마스크 + connectedComponents + 격자 정렬 검증
- **6개 테스트 통과** (`tests/test_board_detector.py`)
- 실제 사이트 캡처에서 confidence=1.00로 사과 170개 모두 검출 성공

### 2. `agent/board_recognizer.py` ✅ 기본 동작
- `Calibration` 데이터클래스 + JSON 로드
- `extract_cells()`, `_has_apple()`, `_white_digit_mask()`
- `build_templates()`, `recognize_board()` — 템플릿 기반 매칭
- `auto_recognize_first_board()` — 클러스터링 자동 라벨링 (실패)
- `ocr_recognize_first_board()` — Tesseract OCR (50~128/170 실패) ❌
- **5개 테스트 통과** (합성 이미지 round-trip)

### 3. `scripts/auto_player.py` ✅ GUI 띄움
- tkinter 통합 앱 (LabelFrame 회피 — macOS Tk 8.5 호환)
- [DETECT] / [START] / [STOP] 버튼
- 점수/스텝 라벨, 로그 영역, stdout 병행 출력
- 백그라운드 스레드로 매수 캡처-인식-드래그 루프
- pyautogui FAILSAFE (마우스 좌상단 코너로 → abort)

### 4. `scripts/calibrate_screen.py` (마우스 수동 보정) — **폐기 예정**
- 사용자가 격자 좌상단/우하단 마우스로 가리키는 방식
- 사용성 나빠서 board_detector로 대체

### 5. `scripts/play_site.py` (터미널 기반) — **폐기 예정**
- 터미널 카운트다운 + 옛 calibration 의존
- auto_player.py가 대체

### 패키지 설치됨
- `playwright` + chromium (사이트 봇 차단으로 보류)
- `mss`, `pyautogui`, `opencv-python`, `pillow`
- `tesseract` (brew) + `pytesseract`
- `pygame` (로컬 GUI 트랙)

---

## 막힌 지점 — OCR 인식 실패

DETECT 흐름:
1. ✅ 화면 캡처
2. ✅ 격자 자동 검출 (사과 170개, confidence 1.00)
3. ❌ **첫 보드 인식 — Tesseract OCR이 셀 50~128개 못 읽음**

### 실패 원인 진단

| 시도 | 결과 |
|---|---|
| 작은 캡처 (1920×1080), 32px 셀, OCR | 50/170 실패 |
| 큰 캡처 (1186×690, 66px 셀), 32px resize, OCR | 128/170 실패 |
| 큰 캡처, **원본 셀 크기** + OCR | 88/170 실패 |
| 더 엄격한 흰 픽셀 마스크 (s<40, v>200) | 74/170 (더 나빠짐) |

근본 원인: **`_white_digit_mask`가 사과 줄기/하이라이트까지 같이 잡아 OCR 노이즈**.
샘플 `/tmp/fail_0_4.png`에서 "6" 글자 주변에 사과 윤곽선이 같이 들어가 있음.

PSM 모드 비교 (단일 셀):
- PSM 6: '4' ✓
- PSM 7: '4' ✓
- PSM 8: '13' ✗
- PSM 10: '4' ✓ (현재 사용)
- PSM 13: '13' ✗

PSM 10이 가장 좋지만 그래도 30~75% 실패. **OCR로는 안 되겠다는 결론**.

---

## 다음 세션에서 할 일

**사용자 결정**: OCR 폐기. 사용자 제공 스크린샷에서 1~9 셀을 시각 확인해
영구 템플릿 만들기.

### 사용자 제공 자료
- `/Users/junyoung/Desktop/sdp/apple/data/site_captures/capture_01_initial.png`
- 1186×690, Play 직후 170 사과 완전한 보드
- `agent.board_detector.detect_grid()`로 격자 검출됨:
  - TL=(64, 48), cell=66.0×66.0, confidence=1.00

### 다음 세션 시작 작업

#### Step A: 사용자 스크린샷에서 보드 정답 시각 추출

마지막 세션에서 부분 진행:
- `/tmp/rows_first5.png` (행 0~4) 큰 이미지 생성 완료
- `/tmp/rows_last5.png` (행 5~9) 큰 이미지 생성 완료

행 0~2까지는 시각 읽기 완료 (검증 필요):
```
행 0: 8 5 2 3 7 3 9 1 8 4 1 6 8 5 9 5 2
행 1: 4 6 8 3 6 7 5 4 2 5 8 7 1 1 3 6 5
행 2: 9 4 5 4 1 5 7 9 4 6 5 3 2 2 4 9 3
```

행 3~9는 미완. 다음 세션에서:
1. `/tmp/rows_first5.png`, `/tmp/rows_last5.png` 다시 보고 10×17 정답 보드 완성
2. **보드 합이 10의 배수**인지 검증 (rejection sampling 사이트 보장)
3. 정답 보드를 코드에 박거나 별도 파일로 저장

**`/tmp/` 캐시가 사라졌다면** 사용자 스크린샷에서 다시 생성 가능 (스크립트 본문은
이 노트 끝 부록 참조).

#### Step B: 영구 템플릿 생성

```python
from agent.board_recognizer import build_templates, save_templates
templates = build_templates(img, calibration, ground_truth_board)
save_templates(templates, Path("models/site_templates.npz"))
```

#### Step C: auto_player.py에서 OCR 경로 제거 + 사전 저장 템플릿 사용

`_do_detect()`를 단순화:
1. 격자 검출 → calibration
2. `Path("models/site_templates.npz")` 있으면 그걸로 시작
3. 없으면 친절한 에러 메시지 (`scripts/build_templates.py` 실행 안내)
4. OCR / 클러스터링 자동 라벨링 코드 다 제거

#### Step D: 실제 사이트에서 1판 자동 플레이 검증

1. 사용자가 사이트 띄우고 Play 누름
2. `python3 scripts/auto_player.py`
3. [DETECT] — 격자 좌표 자동, 기존 templates 로드
4. [START] — greedy_smallest 정책 → 드래그 시작
5. 점수 113 근처면 성공

#### Step E: 다음 보드(새 시드)에서도 동작 검증

사이트는 매 게임 새 랜덤 보드 — templates는 폰트가 같으면 재사용 가능.
한 번 만든 templates가 다른 보드에도 통하는지 확인.

---

## 알려진 주의사항

1. **TK Deprecation 경고**: macOS 시스템 Tk 8.5 — `TK_SILENCE_DEPRECATION=1`로 억제.
   `tkinter.ttk` 위젯이 안 그려지는 버그 있어 일반 `tk` 위젯 사용.
2. **mss vs pyautogui 좌표계**: Retina 디스플레이라면 mss는 물리 픽셀,
   pyautogui는 논리 픽셀. `auto_player.py:get_retina_scale()`로 검출.
   사용자 환경은 외부 모니터 1920×1080 (Retina 아님, scale=1.00).
3. **사이트 봇 차단**: Playwright로 직접 자동화 시도 시 IP 차단됨 (VPN 우회 가능).
   화면 캡처 + 마우스 자동화 방식은 차단 안 당함.
4. **pyautogui FAILSAFE**: 자동 플레이 중 비상 정지하려면 마우스를 화면
   좌상단 코너로 빠르게 이동 → 즉시 abort.

---

## 파일 변경 사항 요약 (이번 세션)

추가:
- `agent/board_detector.py` (자동 격자 검출)
- `agent/board_recognizer.py` (이전 세션에서 만들었지만 OCR 함수 추가)
- `scripts/auto_player.py` (tkinter 통합 GUI)
- `scripts/calibrate_screen.py` (폐기 예정)
- `scripts/play_site.py` (폐기 예정)
- `tests/test_board_detector.py` (6개)
- `tests/test_board_recognizer.py` (5개)

수정:
- `env/fruit_box.py` — `reset`에 rejection sampling
- `.claude/settings.json` — hooks 스키마 nested로
- `docs/PROGRESS.md`, `README.md` — Step 3 완료 표시
- `scripts/play_gui.py` — Try Again 버튼 + GameState 추출

테스트: **97개 전체 통과** (마지막 확인 시점).

---

## 2026-05-20 후속 세션 결과 (Step A–C 완료)

### A: 정답 보드 추출 — **완료**
사용자 스크린샷에서 시각 확인으로 10×17 전 셀 라벨링.
- 합 **890** (mod10=0, 정상 범위 600~1000 안)
- 이전 노트의 행 0~2 부분 라벨링은 부정확했음 → enlarged per-row 이미지로 재읽기

### B: 영구 템플릿 생성 — **완료**
`models/site_templates.npz` (3,307 bytes) 생성.
- 9개 숫자, (32, 32) float32 mean mask
- self-check: `recognize_board()`로 동일 이미지 → 170/170 GT 복원

### C: auto_player.py 단순화 — **완료**
`_do_detect()` 흐름:
1. `models/site_templates.npz` 존재 확인 (없으면 친절한 에러 + abort)
2. 캡처 → `detect_grid()` → calibration
3. `load_templates()` → `recognize_board()` 1회 (현재 보드 상태 로그)
4. [START] 활성화

제거: `auto_recognize_first_board`, `ocr_recognize_first_board`, 클러스터링 폴백,
디버그 캡처/샘플 셀 저장.

테스트: **97개 전체 통과** (회귀 없음).

### D: 실제 사이트 1판 자동 플레이 — **미실행 (사용자 검증 필요)**
다음 사용자 단계:
```bash
# 1. 사이트 띄우고 Play 누름 → 격자 표시
# 2. 자동 플레이 GUI 실행
python3 scripts/auto_player.py
# 3. [DETECT] → 격자 검출 + 보드 인식 로그 확인
# 4. [START] → 5초 후 드래그 시작. greedy_smallest 정책
# 5. 점수 113 근처면 성공
```

### E: 다른 시드 보드에서 템플릿 재사용 — **미실행**
같은 폰트면 통해야 함. 사용자가 한 판 더 돌려보고 인식 정확도 확인.

---

## 다음 세션 첫 명령 권장

```bash
# 1. 현재 상태 확인
cd /Users/junyoung/Desktop/sdp/apple
python3 -m pytest 2>&1 | tail -5

# 2. 이 노트 + PROGRESS 다시 읽고 컨텍스트 복원
cat docs/SESSION_NOTES.md
```

작업 시작점: **사용자 스크린샷에서 정답 보드 완성** (Step A).

---

## 2026-05-20 후속 — Step 4 완료 (사이트 자동 플레이)

### 완료된 마일스톤
1. **정답 보드 라벨링** — capture_01에서 사람 라벨 10×17, 합 890
2. **`models/site_templates.npz` 생성** — 11장 캡처 × 170셀, 숫자별 ~200 샘플
   - `trained_cell_size=66.0` 메타 포함 → 다른 줌에서도 자동 업스케일
3. **`scripts/auto_player.py` 정비**:
   - 격자 자동 검출, 점수 영역 자동 검출, 워밍업 클릭
   - 매 step 캡처+인식+드래그+검증, 점수 OCR로 사이트 거부 감지
   - Esc/Cmd+./STOP 즉시 중단, `/tmp/auto_player_debug/auto_player.log` 자동 기록
4. **`agent/score_reader.py` 추가** — 우상단 그린 점수 OCR (다중 PSM)
5. **테스트 97개 모두 통과** 유지

### 실측 결과
- 사이트 실제 vs 추정 점수 **0 차이 동기화** (95~117점)
- 점수가 낮은 건 정책 천장(greedy_smallest) 때문 → RL/MCTS는 별개 트랙

### 알려진 잔여 이슈
- 시뮬 game.board에 0(빈 칸)이 끼어들면 `[8 0 0 0 0 2]` 같은 사각형이 합 10으로 보임
  → 인식 한두 셀 오인 → 사이트 거부. action_key의 zero_count 우선순위로 완화 적용
- 점수 OCR이 가끔 두 자리 ↔ 한 자리 잘림 (예: site_score=11 vs 실제 115) → 단조 검증으로 흡수

### 다음 세션 후보
1. 정책 개선: greedy_smallest → MCTS/beam search (200+ 노리기)
2. 인식 더 강건화: 셀 마스크 다중 임계 평균화
3. 드래그 더 빠르게: drag_dur 더 줄여 시간 한도 안에 더 많은 step

### git 상태
- `.gitignore` 추가, `models/*.zip,*.pt` 제외 (단 `site_templates.npz` 포함)
- `data/site_captures/`, `*.log`, `/tmp/`, `calibration.json` 제외

---

## 부록: 셀 자르기 스크립트 (참고)

`/tmp/rows_first5.png`, `/tmp/rows_last5.png` 다시 생성:

```python
import sys
sys.path.insert(0, '/Users/junyoung/Desktop/sdp/apple')
import numpy as np
from PIL import Image
from agent.board_detector import detect_grid

img = np.array(Image.open(
    '/Users/junyoung/Desktop/sdp/apple/data/site_captures/capture_01_initial.png'
).convert('RGB'))
result = detect_grid(img)
cal = result.calibration
half = int(min(cal.cell_width, cal.cell_height) // 2) - 2
upscale = 5
gap = 10
sw = sh = 2 * half * upscale

for half_range, label in [((0, 5), 'first5'), ((5, 10), 'last5')]:
    r0, r1 = half_range
    n_rows = r1 - r0
    total_w = (sw + gap) * 17 + gap
    total_h = (sh + gap) * n_rows + gap
    combined = np.full((total_h, total_w, 3), 240, dtype=np.uint8)
    for r in range(r0, r1):
        for c in range(17):
            cx, cy = cal.cell_center(r, c)
            crop = img[max(0,cy-half):cy+half, max(0,cx-half):cx+half]
            big = np.array(Image.fromarray(crop).resize((sw, sh), Image.NEAREST))
            x = gap + c * (sw + gap)
            y = gap + (r - r0) * (sh + gap)
            combined[y:y+big.shape[0], x:x+big.shape[1]] = big
    Image.fromarray(combined).save(f'/tmp/rows_{label}.png')
```
