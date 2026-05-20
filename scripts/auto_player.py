"""사과게임 자동 플레이 통합 GUI — 한 창에서 모든 작업.

사용 흐름:
  1. 브라우저로 사이트 띄우고 Play 눌러 격자 보이게 함
  2. 이 앱 실행
  3. [DETECT] 버튼 클릭 — 화면 캡처해서 격자 자동 검출 + 템플릿 학습
  4. [START] 버튼 클릭 — 자동 플레이 시작
  5. [STOP] 클릭 또는 마우스를 화면 좌상단 코너로 → 즉시 중단

사용법:
    python3 scripts/auto_player.py
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

# macOS 시스템 Tk 8.5의 deprecation 경고 억제 (동작에는 영향 없음)
os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import mss  # noqa: E402
import pyautogui  # noqa: E402

from agent.board_detector import detect_grid  # noqa: E402
from agent.board_recognizer import (  # noqa: E402
    Calibration,
    load_templates_meta,
    recognize_board,
)
from agent.score_reader import (  # noqa: E402
    ScoreRegion,
    find_score_region,
    read_score,
)
from typing import Optional  # noqa: E402
import cv2  # noqa: E402
from agent.heuristics import (  # noqa: E402
    find_valid_actions,
    greedy_smallest_policy,
)
from env.fruit_box import Action, FruitBox  # noqa: E402


pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0

TEMPLATES_PATH = Path("models/site_templates.npz")
DEBUG_DIR = Path("/tmp/auto_player_debug")


def _dump_capture(img: np.ndarray, name: str) -> Path:
    """캡처 RGB 이미지를 PNG로 저장하고 경로 반환."""
    from PIL import Image as PILImage

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    out = DEBUG_DIR / name
    PILImage.fromarray(img).save(out)
    return out


def _dump_grid_overlay(
    img: np.ndarray,
    calibration: "Calibration",
    name: str,
    board: np.ndarray | None = None,
) -> Path:
    """캡처 이미지 위에 격자 셀 중심 + (옵션) 인식 숫자를 표시해 저장.

    board가 주어지면 각 셀 중심 위쪽에 인식한 숫자를 노란색으로 그림.
    셀 중심점은 항상 빨간 점으로 표시.
    """
    import cv2 as _cv2

    overlay = img.copy()[:, :, ::-1].copy()  # RGB→BGR for cv2
    half_w = int(calibration.cell_width // 2)
    half_h = int(calibration.cell_height // 2)
    for r in range(calibration.rows):
        for c in range(calibration.cols):
            cx, cy = calibration.cell_center(r, c)
            x1, y1 = cx - half_w, cy - half_h
            x2, y2 = cx + half_w, cy + half_h
            _cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
            _cv2.circle(overlay, (cx, cy), 3, (0, 0, 255), -1)
            if board is not None:
                d = int(board[r, c])
                if d > 0:
                    _cv2.putText(
                        overlay,
                        str(d),
                        (cx - 10, cy - 8),
                        _cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        _cv2.LINE_AA,
                    )
    out = DEBUG_DIR / name
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    _cv2.imwrite(str(out), overlay)
    return out


def _board_to_text(board: np.ndarray) -> str:
    lines = []
    for r in range(board.shape[0]):
        lines.append(" ".join(f"{int(v)}" if v > 0 else "." for v in board[r]))
    return "\n".join(lines)


def capture_screen() -> np.ndarray:
    """전체 화면 캡처 → RGB ndarray (mss 픽셀, Retina면 물리 픽셀)."""
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[0])
        img = np.array(shot)[:, :, :3]
        return img[:, :, ::-1].copy()


def get_retina_scale() -> float:
    """mss 캡처 크기 vs pyautogui 화면 크기로 Retina 배율 계산.

    macOS Retina면 보통 2.0, 일반 디스플레이면 1.0.
    """
    with mss.mss() as sct:
        monitor = sct.monitors[0]
        physical_w = monitor["width"]
    logical_w = pyautogui.size()[0]
    return physical_w / logical_w if logical_w > 0 else 1.0


def drag(
    start_xy: tuple[int, int],
    end_xy: tuple[int, int],
    duration: float,
    stop_event: threading.Event | None = None,
) -> bool:
    """이동 → 안정화 → 누름 → 안정화 → 끌어서 → 뗌.

    stop_event가 set되면 mouseUp 후 즉시 False 반환. 정상 완료 시 True.
    어떤 경로로든 mouseUp 보장 (마우스가 끌리는 상태로 남지 않게).
    """
    def _should_stop() -> bool:
        return stop_event is not None and stop_event.is_set()

    try:
        pyautogui.moveTo(start_xy[0], start_xy[1], duration=0.03)
        if _should_stop():
            return False
        time.sleep(0.03)
        if _should_stop():
            return False
        pyautogui.mouseDown()
        time.sleep(0.03)
        if _should_stop():
            return False
        pyautogui.moveTo(end_xy[0], end_xy[1], duration=duration)
        time.sleep(0.02)
        return True
    finally:
        # mouseDown 후 어디서 멈추든 mouseUp 반드시 실행
        try:
            pyautogui.mouseUp()
        except Exception:
            pass


def action_to_pixels(
    action: Action, calibration: Calibration, inset: int = 4
) -> tuple[tuple[int, int], tuple[int, int]]:
    """액션 사각형의 셀 중심 기준 시작/끝 좌표. inset만큼 안쪽으로 들임.

    너무 가장자리면 사이트가 인접 셀 포함해버리고, 너무 안쪽이면 영역 미인식.
    cell=66px 환경 기준 inset=4가 안정.
    """
    x1, y1 = calibration.cell_center(action.r1, action.c1)
    x2, y2 = calibration.cell_center(action.r2, action.c2)
    return (x1 - inset, y1 - inset), (x2 + inset, y2 + inset)


class AutoPlayerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Apple Game Auto Player")
        self.root.geometry("520x460")

        self.calibration: Calibration | None = None
        self.templates: dict[int, np.ndarray] | None = None
        self.trained_cell_size: float | None = None
        # 인식 직전 입력 이미지에 적용할 배율(>1이면 업스케일).
        # 드래그 좌표는 원본/scale로 환산.
        self.recognize_scale: float = 1.0
        self.score_region: ScoreRegion | None = None
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.log_q: queue.Queue[str] = queue.Queue()

        # 로그를 파일에도 기록 (claude가 직접 읽도록)
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_file_path = DEBUG_DIR / "auto_player.log"
        session_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self.log_file_path.open("a", encoding="utf-8") as f:
                f.write(f"\n=== session start {session_ts} ===\n")
        except Exception:
            pass

        # 키보드 단축키: Esc / Cmd+. / Ctrl+C → STOP
        self.root.bind("<Escape>", lambda e: self._on_stop())
        self.root.bind("<Command-period>", lambda e: self._on_stop())
        self.root.bind("<Control-c>", lambda e: self._on_stop())

        self._build_ui()
        self._poll_log_queue()

    def _build_ui(self) -> None:
        # macOS Tk 8.5의 LabelFrame 렌더링 문제 회피 — 일반 Frame + 라벨 헤더만 사용.
        BG = "#f0f0f0"
        SECTION_BG = "#ffffff"
        self.root.configure(bg=BG)

        # 1. 격자 검출 섹션
        sec1 = tk.Frame(self.root, bg=SECTION_BG, bd=1, relief="solid")
        sec1.pack(fill="x", padx=12, pady=(12, 6))
        tk.Label(
            sec1,
            text="1. 격자 검출",
            font=("Arial", 12, "bold"),
            bg=SECTION_BG,
            fg="#222",
            anchor="w",
        ).pack(fill="x", padx=10, pady=(8, 4))
        row1 = tk.Frame(sec1, bg=SECTION_BG)
        row1.pack(fill="x", padx=10, pady=(0, 10))
        self.detect_btn = tk.Button(
            row1,
            text="DETECT (자동 검출)",
            command=self._on_detect,
            font=("Arial", 13, "bold"),
            bg="#4a90e2",
            fg="white",
            activebackground="#357abd",
            highlightbackground="#4a90e2",
            relief="flat",
            padx=14,
            pady=8,
        )
        self.detect_btn.pack(side="left")
        self.detect_status = tk.Label(
            row1,
            text="아직 검출 안 됨",
            fg="#666",
            bg=SECTION_BG,
            font=("Arial", 12),
        )
        self.detect_status.pack(side="left", padx=12)

        # 2. 자동 플레이 섹션
        sec2 = tk.Frame(self.root, bg=SECTION_BG, bd=1, relief="solid")
        sec2.pack(fill="x", padx=12, pady=6)
        tk.Label(
            sec2,
            text="2. 자동 플레이",
            font=("Arial", 12, "bold"),
            bg=SECTION_BG,
            fg="#222",
            anchor="w",
        ).pack(fill="x", padx=10, pady=(8, 4))
        row2 = tk.Frame(sec2, bg=SECTION_BG)
        row2.pack(fill="x", padx=10, pady=(0, 10))
        self.start_btn = tk.Button(
            row2,
            text="▶ START",
            command=self._on_start,
            state="disabled",
            font=("Arial", 13, "bold"),
            bg="#28a745",
            fg="white",
            activebackground="#218838",
            disabledforeground="#999",
            highlightbackground="#28a745",
            relief="flat",
            padx=14,
            pady=8,
        )
        self.start_btn.pack(side="left")
        self.stop_btn = tk.Button(
            row2,
            text="■ STOP",
            command=self._on_stop,
            state="disabled",
            font=("Arial", 13, "bold"),
            bg="#dc3545",
            fg="white",
            activebackground="#c82333",
            disabledforeground="#999",
            highlightbackground="#dc3545",
            relief="flat",
            padx=14,
            pady=8,
        )
        self.stop_btn.pack(side="left", padx=6)
        self.play_status = tk.Label(
            row2, text="대기 중", fg="#666", bg=SECTION_BG, font=("Arial", 12)
        )
        self.play_status.pack(side="left", padx=12)

        # 점수
        sec3 = tk.Frame(self.root, bg=SECTION_BG, bd=1, relief="solid")
        sec3.pack(fill="x", padx=12, pady=6)
        row3 = tk.Frame(sec3, bg=SECTION_BG)
        row3.pack(fill="x", padx=10, pady=10)
        tk.Label(
            row3,
            text="Score:",
            font=("Arial", 14, "bold"),
            bg=SECTION_BG,
            fg="#222",
        ).pack(side="left")
        self.score_label = tk.Label(
            row3,
            text="0",
            font=("Arial", 22, "bold"),
            fg="#c84",
            bg=SECTION_BG,
        )
        self.score_label.pack(side="left", padx=(6, 24))
        tk.Label(
            row3,
            text="Step:",
            font=("Arial", 14),
            bg=SECTION_BG,
            fg="#222",
        ).pack(side="left")
        self.step_label = tk.Label(
            row3, text="0", font=("Arial", 16), bg=SECTION_BG, fg="#222"
        )
        self.step_label.pack(side="left", padx=6)

        # 로그 섹션
        sec4 = tk.Frame(self.root, bg=SECTION_BG, bd=1, relief="solid")
        sec4.pack(fill="both", expand=True, padx=12, pady=6)
        tk.Label(
            sec4,
            text="Log",
            font=("Arial", 12, "bold"),
            bg=SECTION_BG,
            fg="#222",
            anchor="w",
        ).pack(fill="x", padx=10, pady=(8, 4))
        self.log_text = tk.Text(
            sec4,
            height=12,
            font=("Menlo", 11),
            bg="#fafafa",
            fg="#222",
            bd=0,
            highlightthickness=0,
        )
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log_text.configure(state="disabled")

        help_text = (
            "사용법: 사이트에서 Play 누른 뒤 → DETECT → START\n"
            "비상정지: STOP 버튼 / Esc / Cmd+. / 마우스를 좌상단 코너로"
        )
        tk.Label(
            self.root,
            text=help_text,
            fg="#888",
            bg=BG,
            font=("Arial", 10),
        ).pack(side="bottom", pady=6)

    def _log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"{ts}  {msg}"
        self.log_q.put(line)
        # GUI 로그가 macOS Tk 8.5에서 안 그려지는 경우 대비, stdout에도.
        print(line, flush=True)
        # 파일에도 append (분석용)
        try:
            with self.log_file_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _poll_log_queue(self) -> None:
        try:
            while True:
                line = self.log_q.get_nowait()
                self.log_text.configure(state="normal")
                self.log_text.insert("end", line + "\n")
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def _on_detect(self) -> None:
        self.detect_btn.configure(state="disabled")
        self.detect_status.configure(
            text="3초 후 캡처... 브라우저를 활성 상태로 두세요"
        )
        self._log("[DETECT] 3초 후 캡처합니다 — 브라우저 활성화")
        self.root.after(3000, self._do_detect)

    def _do_detect(self) -> None:
        try:
            if not TEMPLATES_PATH.exists():
                self._log(
                    f"[DETECT] 템플릿 파일이 없습니다: {TEMPLATES_PATH}\n"
                    "        먼저 정답 보드 한 장으로 site_templates.npz를 생성하세요."
                )
                self.detect_status.configure(
                    text="템플릿 없음", foreground="#c33"
                )
                self.detect_btn.configure(state="normal")
                return

            self._log("[DETECT] 캡처 중…")
            img = capture_screen()
            scale = get_retina_scale()
            logical_w, logical_h = pyautogui.size()
            self._log(
                f"[DETECT] mss 캡처={img.shape[1]}×{img.shape[0]} px, "
                f"pyautogui 화면={logical_w}×{logical_h}, Retina scale={scale:.2f}"
            )
            cap_path = _dump_capture(img, "detect_capture.png")
            self._log(f"[DETECT] 캡처 저장: {cap_path}")

            # 1차 격자 검출 (원본 캡처)
            raw_result = detect_grid(img)
            if raw_result is None:
                self._log(
                    "[DETECT] 실패 — 격자를 찾지 못함. "
                    "사이트가 Play 상태인지 확인하세요."
                )
                self.detect_status.configure(text="검출 실패", foreground="#c33")
                self.detect_btn.configure(state="normal")
                return

            meta = load_templates_meta(TEMPLATES_PATH)
            self.templates = meta["templates"]
            self.trained_cell_size = meta["trained_cell_size"]
            self._log(
                f"[DETECT] 템플릿 로드: {TEMPLATES_PATH} "
                f"({len(self.templates)}개 숫자, "
                f"trained_cell_size={self.trained_cell_size})"
            )

            # 학습 시 셀 크기와 현재 셀 크기 비교 → 자동 업스케일 결정
            current_cell = raw_result.calibration.cell_width
            if (
                self.trained_cell_size is not None
                and current_cell > 0
                and self.trained_cell_size / current_cell >= 1.3
            ):
                self.recognize_scale = self.trained_cell_size / current_cell
                self._log(
                    f"[DETECT] 셀 크기 {current_cell:.1f}px < 학습 시 "
                    f"{self.trained_cell_size:.1f}px → 인식용으로 "
                    f"{self.recognize_scale:.2f}× 업스케일"
                )
                upscaled = cv2.resize(
                    img,
                    None,
                    fx=self.recognize_scale,
                    fy=self.recognize_scale,
                    interpolation=cv2.INTER_CUBIC,
                )
                result = detect_grid(upscaled)
                if result is None:
                    self._log("[DETECT] 업스케일 후 격자 재검출 실패 → 원본 사용")
                    self.recognize_scale = 1.0
                    self.calibration = raw_result.calibration
                    img_for_recog = img
                else:
                    self.calibration = result.calibration
                    img_for_recog = upscaled
                    self._log(
                        f"[DETECT] 업스케일 후 cell="
                        f"{result.calibration.cell_width:.1f}px"
                    )
            else:
                self.recognize_scale = 1.0
                self.calibration = raw_result.calibration
                img_for_recog = img
                result = raw_result

            self._log(
                f"[DETECT] 격자 검출 OK — 사과 {raw_result.n_apples_detected}개, "
                f"confidence={raw_result.confidence:.2f}"
            )
            self._log(
                f"[DETECT] 격자: TL=({self.calibration.top_left_x},"
                f"{self.calibration.top_left_y}) "
                f"cell={self.calibration.cell_width:.1f}×"
                f"{self.calibration.cell_height:.1f} "
                f"(recognize_scale={self.recognize_scale:.2f})"
            )

            grid_path = _dump_grid_overlay(
                img_for_recog, self.calibration, "detect_grid_overlay.png"
            )
            self._log(f"[DETECT] 격자 오버레이 저장: {grid_path}")

            board = recognize_board(
                img_for_recog, self.calibration, self.templates
            )
            total = int(board.sum())
            n_apples = int((board > 0).sum())
            self._log(
                f"[DETECT] 보드 인식 — 사과 {n_apples}개, 합 {total} "
                f"(mod10={total % 10})"
            )
            board_txt = _board_to_text(board)
            (DEBUG_DIR / "detect_board.txt").write_text(
                board_txt, encoding="utf-8"
            )
            self._log(
                "[DETECT] 인식된 보드 (저장: detect_board.txt):\n" + board_txt
            )
            recog_path = _dump_grid_overlay(
                img_for_recog,
                self.calibration,
                "detect_recognized_overlay.png",
                board=board,
            )
            self._log(
                f"[DETECT] 인식 결과 시각화 저장: {recog_path} "
                "(셀 위에 노란 숫자 = 인식한 값)"
            )

            # 점수 영역 자동 검출 + 첫 OCR 시도
            self.score_region = find_score_region(
                img_for_recog, self.calibration
            )
            if self.score_region is not None:
                roi = self.score_region.crop(img_for_recog)
                score0 = read_score(roi)
                self._log(
                    f"[DETECT] 점수 영역=({self.score_region.x1},"
                    f"{self.score_region.y1})-"
                    f"({self.score_region.x2},{self.score_region.y2}) "
                    f"OCR={score0}"
                )
                try:
                    cv2.imwrite(
                        str(DEBUG_DIR / "score_region.png"),
                        cv2.cvtColor(roi, cv2.COLOR_RGB2BGR),
                    )
                except Exception:
                    pass
            else:
                self._log(
                    "[DETECT] 점수 영역 자동 검출 실패 — 공간 검증만 사용"
                )
            if total % 10 != 0:
                self._log(
                    "[DETECT] ⚠ 보드 합이 10의 배수가 아님 → 인식 오류 있음. "
                    "detect_recognized_overlay.png에서 사이트 보드와 비교해보세요"
                )

            self.detect_status.configure(
                text=f"검출 완료 (conf {result.confidence:.2f})",
                foreground="#080",
            )
            self.detect_btn.configure(state="normal")
            self.start_btn.configure(state="normal")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._log(f"[DETECT] 예외: {e}")
            self.detect_status.configure(text="에러", foreground="#c33")
            self.detect_btn.configure(state="normal")

    def _on_start(self) -> None:
        if self.calibration is None or self.templates is None:
            self._log("[START] 검출 먼저 하세요.")
            return
        if self.worker and self.worker.is_alive():
            self._log("[START] 이미 진행 중")
            return
        self.stop_event.clear()
        self.start_btn.configure(state="disabled")
        self.detect_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.play_status.configure(text="플레이 중…", foreground="#080")
        self._log("[START] 자동 플레이 시작 — 5초 후. 브라우저 활성화하세요.")
        self.root.after(5000, self._start_worker)

    def _start_worker(self) -> None:
        self.worker = threading.Thread(target=self._play_loop, daemon=True)
        self.worker.start()

    def _on_stop(self) -> None:
        # idle 상태(worker 없음)에서 Esc 눌러도 안전
        if self.worker is None or not self.worker.is_alive():
            return
        self.stop_event.set()
        self._log("[STOP] 중단 요청. worker 정리 대기…")
        try:
            self.stop_btn.configure(state="disabled")
        except Exception:
            pass

    def _capture_recog_image(self) -> np.ndarray:
        """현재 화면 캡처 (recognize_scale 적용된 RGB 이미지)."""
        img = capture_screen()
        if self.recognize_scale != 1.0:
            img = cv2.resize(
                img, None,
                fx=self.recognize_scale, fy=self.recognize_scale,
                interpolation=cv2.INTER_CUBIC,
            )
        return img

    def _capture_and_recognize(self) -> np.ndarray:
        """현재 화면 캡처 + 인식 → 보드 (10, 17)."""
        img = self._capture_recog_image()
        return recognize_board(img, self.calibration, self.templates)

    def _read_score_now(self) -> Optional[int]:
        """현재 화면에서 점수 영역을 읽어 OCR. 매번 영역을 재검출해
        점수 자릿수 증가에 따른 영역 확장을 자동 반영.
        """
        if self.calibration is None:
            return None
        img = self._capture_recog_image()
        try:
            # 매 step 영역 재검출 (점수 자릿수 변하면 영역도 변함)
            region = find_score_region(img, self.calibration)
            if region is None:
                # 검출 실패 시 캐시된 영역 사용
                if self.score_region is None:
                    return None
                region = self.score_region
            else:
                self.score_region = region
            roi = region.crop(img)
            return read_score(roi)
        except Exception:
            return None

    @staticmethod
    def _action_succeeded(board_after: np.ndarray, action: Action) -> bool:
        """드래그 후 액션 사각형 안 셀들이 모두 0이면 사이트가 받아들인 것."""
        rect = board_after[
            action.r1 : action.r2 + 1, action.c1 : action.c2 + 1
        ]
        return bool((rect == 0).all())

    def _verify_cleared(self, action: Action, board_after: np.ndarray) -> bool:
        """엄격한 검증: 인식보드 + 원본 픽셀 + 셀별 사과 개수 모두 통과해야 성공.

        가짜 성공을 줄이기 위해 다층 검사:
          1. 인식된 board_after에서 사각형이 모두 0인지
          2. 사각형 픽셀 영역의 빨간 비율이 1.5% 미만인지
          3. 사각형 안 각 셀에서 빨간 컴포넌트 ≥ 50px 짜리가 없는지
             (사과 한 알의 핵심 영역. 흐림/anti-aliasing은 < 50px)
        """
        if not self._action_succeeded(board_after, action):
            return False
        import mss as _mss
        with _mss.mss() as sct:
            shot = sct.grab(sct.monitors[0])
            img = np.array(shot)[:, :, :3][:, :, ::-1].copy()
        if self.recognize_scale != 1.0:
            img = cv2.resize(
                img, None,
                fx=self.recognize_scale, fy=self.recognize_scale,
                interpolation=cv2.INTER_CUBIC,
            )
        assert self.calibration is not None
        cw = self.calibration.cell_width
        ch = self.calibration.cell_height
        half = int(min(cw, ch) // 2)

        # 1) 전체 사각형 영역 빨간 비율
        x1, y1 = self.calibration.cell_center(action.r1, action.c1)
        x2, y2 = self.calibration.cell_center(action.r2, action.c2)
        rx1, ry1 = max(0, x1 - half), max(0, y1 - half)
        rx2 = min(img.shape[1], x2 + half)
        ry2 = min(img.shape[0], y2 + half)
        if rx2 <= rx1 or ry2 <= ry1:
            return True
        roi = img[ry1:ry2, rx1:rx2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        red_mask = ((h < 15) | (h > 165)) & (s > 100) & (v > 80)
        red_ratio = float(red_mask.sum()) / max(red_mask.size, 1)
        if red_ratio > 0.015:
            return False

        # 2) 셀별 큰 빨간 컴포넌트(=사과 본체) 존재 여부
        # 사과 한 알의 빨간 컴포넌트는 대략 30~60% 셀 면적. 임계 50px (cell=66 기준).
        # cell_size 무관하게 셀 면적의 8% 이상 빨간 컴포넌트면 사과 있다고 판단.
        cell_area = max(int(cw * ch), 1)
        comp_threshold = max(50, cell_area * 8 // 100)

        n_labels, _labels, stats, _cent = cv2.connectedComponentsWithStats(
            red_mask.astype(np.uint8) * 255, connectivity=8
        )
        for i in range(1, n_labels):
            area = int(stats[i][4])
            if area >= comp_threshold:
                return False
        return True

    def _play_loop(self) -> None:
        """worker 스레드. UI는 self.root.after()로만 갱신.

        흐름 (비동기 진행, 빠른 루프):
          1. 캡처+인식 → game.board 동기화 (사이트 ground-truth)
          2. 직전 액션이 있으면: 그 사각형이 비었는지 검사
             - 성공: score += rect_size, 실패 카운터 리셋
             - 실패: 그 액션 자리 잠금(같은 사각형 잠시 회피)
          3. 정책으로 다음 액션 선택, 드래그 (대기 없이 다음 루프로)
          4. step++
        """
        assert self.calibration is not None and self.templates is not None
        game = FruitBox()
        game.board = np.zeros_like(game.board)
        game.score = 0
        game._initialized = True

        score = 0
        step = 0
        max_steps = 500
        drag_dur = 0.12       # 사이트가 드래그 인식하기에 충분히 빠르되 안정적
        post_drag_wait = 0.18  # 사이트 애니메이션 + DOM 업데이트 시간
        consec_failures = 0
        max_consec_failures = 10  # 초반 인식 한 칸이라도 빗나가면 연쇄 실패 가능 — 여유
        recent_fail_actions: set[tuple[int, int, int, int]] = set()
        fail_cool_steps = 5
        fail_cool: dict[tuple[int, int, int, int], int] = {}

        retina_scale = get_retina_scale()
        recog_scale = self.recognize_scale
        coord_div = recog_scale * retina_scale
        self._log(
            f"[PLAY] 시작. drag 좌표 = calibration / "
            f"(recog_scale={recog_scale:.2f} × retina={retina_scale:.2f}) "
            f"= /{coord_div:.2f}"
        )
        self._log(
            f"[PLAY] drag_dur={drag_dur}s (비동기 진행, 다음 step 캡처가 검증)"
        )

        # 워밍업: 브라우저 활성화용 안전 클릭 (격자 위쪽 빈 영역)
        # 좌표는 격자 top_left보다 위쪽(타이틀바 아래 + 보드 위) 또는 cell_size/2 위
        try:
            warmup_x_phys = (
                self.calibration.top_left_x
                + self.calibration.cell_width * 8  # 격자 중앙 가로
            )
            warmup_y_phys = max(
                self.calibration.top_left_y - self.calibration.cell_height * 1.5,
                10.0,
            )
            wx = int(round(warmup_x_phys / coord_div))
            wy = int(round(warmup_y_phys / coord_div))
            self._log(f"[PLAY] 워밍업 클릭 → ({wx}, {wy}) 브라우저 활성화")
            pyautogui.moveTo(wx, wy, duration=0.03)
            time.sleep(0.05)
            pyautogui.click()
            time.sleep(0.20)  # 브라우저가 포커스 받는 시간
        except Exception as e:
            self._log(f"[PLAY] 워밍업 실패 (무시): {e}")

        prev_action: Action | None = None
        # 게임 시작 시 사이트 점수는 반드시 0이라고 가정.
        # OCR이 잘못 큰 값을 읽으면 무시하고 0으로 초기화 → 노이즈 차단.
        last_known_score: Optional[int] = 0
        if self.score_region is not None:
            raw = self._read_score_now()
            if raw is not None and 0 <= raw <= 5:
                last_known_score = raw
            self._log(
                f"[PLAY] 시작 점수 OCR raw={raw} → 초기값={last_known_score}"
            )

        try:
            while not self.stop_event.is_set() and step < max_steps:
                # 직전 액션이 있을 때만 사이트 처리 시간 추가 대기
                # (드래그 시작 → 사과 소실 애니메이션 ~300ms)
                if prev_action is not None:
                    if self.stop_event.wait(0.18):
                        break
                board = self._capture_and_recognize()

                # 직전 액션 검증
                # 1순위: 점수 OCR — 진실의 근원, 단조 증가만 가능
                # 2순위: 공간 검사(사각형 비었나) — OCR 실패 시 fallback
                if prev_action is not None:
                    cleared = False
                    site_score: Optional[int] = None
                    if self.score_region is not None:
                        site_score = self._read_score_now()
                        if site_score is not None:
                            self._log(
                                f"[OCR] site_score={site_score} "
                                f"(last_known={last_known_score})"
                            )
                        # 한 액션 최대 보상은 사각형 크기 = 10*17=170 이지만
                        # 합 10 제약 때문에 실질 최대는 ~10. 안전하게 30 이하 증가만 인정.
                        if (
                            site_score is not None
                            and last_known_score is not None
                            and site_score > last_known_score
                            and site_score - last_known_score <= 30
                        ):
                            cleared = True
                        elif (
                            site_score is not None
                            and last_known_score is not None
                            and site_score - last_known_score > 30
                        ):
                            # OCR이 갑자기 큰 값 → 노이즈, 무시
                            self._log(
                                f"[OCR-IGNORE] score jump {last_known_score}→{site_score}"
                            )
                            site_score = None

                    # OCR이 단조 증가로 확인되면 cleared 확정
                    # OCR이 작동하지만 미증가/감소면 → 거의 확실히 FAIL
                    # OCR이 None이면 → 공간 검사로 fallback
                    if not cleared and site_score is None:
                        # OCR 자체가 안 됨 → 공간 검사로 fallback
                        if self.stop_event.wait(0.10):
                            break
                        board2 = self._capture_and_recognize()
                        if self._verify_cleared(prev_action, board2):
                            cleared = True
                            board = board2
                            # 한 번 더 OCR
                            s2 = self._read_score_now()
                            if (
                                s2 is not None
                                and last_known_score is not None
                                and s2 > last_known_score
                                and s2 - last_known_score <= 30
                            ):
                                site_score = s2

                    if cleared:
                        # OCR이 단조 증가로 신뢰되면 → 사이트 점수 절대값으로 동기화
                        # OCR 없으면 → 사각형 크기로 누적 (fallback)
                        if (
                            site_score is not None
                            and last_known_score is not None
                            and site_score > last_known_score
                            and site_score - last_known_score <= 30
                        ):
                            score = site_score
                            last_known_score = site_score
                        else:
                            rect_size = (
                                (prev_action.r2 - prev_action.r1 + 1)
                                * (prev_action.c2 - prev_action.c1 + 1)
                            )
                            score += rect_size
                        consec_failures = 0
                        self.root.after(0, self._update_score, score, step)
                    else:
                        consec_failures += 1
                        key = (
                            prev_action.r1, prev_action.c1,
                            prev_action.r2, prev_action.c2,
                        )
                        fail_cool[key] = fail_cool_steps
                        recent_fail_actions.add(key)
                        # 사이트 거부 후 보드의 그 사각형이 어떻게 보이는지
                        after_cells = board[
                            prev_action.r1 : prev_action.r2 + 1,
                            prev_action.c1 : prev_action.c2 + 1,
                        ]
                        after_rows = [
                            " ".join(str(int(v)) for v in row)
                            for row in after_cells
                        ]
                        after_str = " | ".join(after_rows)
                        self._log(
                            f"[FAIL] r=({prev_action.r1},{prev_action.c1})-"
                            f"({prev_action.r2},{prev_action.c2}) "
                            f"사이트 거부 (연속 {consec_failures}) "
                            f"after_cells=[{after_str}]"
                        )
                        if consec_failures >= max_consec_failures:
                            self._log(
                                f"[PLAY] 연속 실패 {max_consec_failures}회 → 종료"
                            )
                            break

                # 시뮬은 항상 가장 최근 인식 결과로 강제 덮어쓰기
                game.board = board
                apples_now = int((board > 0).sum())
                if apples_now == 0:
                    self._log("[PLAY] 사과 0 → 클리어!")
                    break

                actions = find_valid_actions(game)
                if not actions:
                    self._log("[PLAY] valid 액션 없음 → 종료")
                    break

                # 정렬 우선순위:
                # 1) 사각형 안 0(빈 칸) 개수 적은 것 — 인식 오류로 의심되는 액션 회피
                # 2) 사각형 크기 작은 것 — greedy_smallest 보드 모양 보존
                # 3) 위치 (안정적 ordering)
                def action_key(a: Action) -> tuple[int, int, int, int]:
                    cells = board[a.r1 : a.r2 + 1, a.c1 : a.c2 + 1]
                    zero_count = int((cells == 0).sum())
                    size = (a.r2 - a.r1 + 1) * (a.c2 - a.c1 + 1)
                    return (zero_count, size, a.r1, a.c1)

                actions_sorted = sorted(actions, key=action_key)

                # 최근 실패한 액션은 일정 step 동안 회피
                action = None
                for cand in actions_sorted:
                    key = (cand.r1, cand.c1, cand.r2, cand.c2)
                    if key not in fail_cool:
                        action = cand
                        break
                if action is None:
                    # 모든 후보가 cooldown 중이면 첫 번째 그냥 시도
                    action = actions_sorted[0]

                start_xy_phys, end_xy_phys = action_to_pixels(
                    action, self.calibration
                )
                start_xy = (
                    int(round(start_xy_phys[0] / coord_div)),
                    int(round(start_xy_phys[1] / coord_div)),
                )
                end_xy = (
                    int(round(end_xy_phys[0] / coord_div)),
                    int(round(end_xy_phys[1] / coord_div)),
                )

                step += 1
                # 사각형 안 셀별 숫자 추출 + 합 (인식 보드 기준)
                rect_cells = board[
                    action.r1 : action.r2 + 1,
                    action.c1 : action.c2 + 1,
                ]
                rect_rows = [
                    " ".join(str(int(v)) for v in row) for row in rect_cells
                ]
                rect_str = " | ".join(rect_rows)
                rect_sum = int(rect_cells.sum())
                self._log(
                    f"[step {step:3d}] try "
                    f"r=({action.r1},{action.c1})-"
                    f"({action.r2},{action.c2}) "
                    f"cells=[{rect_str}] sum={rect_sum} "
                    f"→ score so far={score}"
                )

                completed = drag(
                    start_xy, end_xy, drag_dur, stop_event=self.stop_event
                )
                if not completed:
                    self._log("[STOP] 드래그 중단됨")
                    break
                prev_action = action

                # cooldown 카운트 감소
                expired = [k for k, v in fail_cool.items() if v <= 1]
                for k in expired:
                    fail_cool.pop(k, None)
                fail_cool = {k: v - 1 for k, v in fail_cool.items()}

                if self.stop_event.wait(post_drag_wait):
                    break

            self._log(f"[PLAY] 종료. 추정 점수={score}")
        except pyautogui.FailSafeException:
            self._log("[PLAY] FAILSAFE — 마우스가 코너로 이동, 중단")
        except Exception as e:
            self._log(f"[PLAY] 예외: {e}")
        finally:
            self.root.after(0, self._on_worker_done)

    def _update_score(self, score: int, step: int) -> None:
        self.score_label.configure(text=str(score))
        self.step_label.configure(text=str(step))

    def _on_worker_done(self) -> None:
        self.start_btn.configure(state="normal")
        self.detect_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.play_status.configure(text="대기 중", foreground="#666")


def main() -> int:
    root = tk.Tk()
    AutoPlayerApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
