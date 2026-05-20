"""scripts.generate_imitation_data로 만든 데이터의 무결성 검증.

작은 episodes로 .npz를 만들고 다음을 확인한다:
  - shape/dtype 일관성
  - 모든 저장된 action이 해당 step의 mask에서 valid (True)
  - action 인덱스 범위 (0 <= a < NUM_ACTIONS)
  - episode_id 길이가 obs와 같고, scores 길이가 episodes와 같음
  - obs 값 범위 0..1 (board/9 정규화)
  - 같은 seed로 다시 만들면 완전히 동일 (재현성)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from env.action_space import NUM_ACTIONS
from env.fruit_box import BOARD_COLS, BOARD_ROWS
from scripts.generate_imitation_data import generate


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict:
    out_dir = tmp_path_factory.mktemp("imitation_data")
    path = generate(episodes=5, seed=42, output_dir=out_dir)
    with np.load(path) as npz:
        data = {k: npz[k].copy() for k in npz.files}
    return data


def test_shapes_and_dtypes(small_dataset: dict) -> None:
    n = small_dataset["obs"].shape[0]
    assert n > 0
    assert small_dataset["obs"].shape == (n, 1, BOARD_ROWS, BOARD_COLS)
    assert small_dataset["obs"].dtype == np.float32

    assert small_dataset["action"].shape == (n,)
    assert small_dataset["action"].dtype == np.int64

    assert small_dataset["mask"].shape == (n, NUM_ACTIONS)
    assert small_dataset["mask"].dtype == bool

    assert small_dataset["episode_id"].shape == (n,)
    assert small_dataset["scores"].shape == (5,)


def test_action_in_range(small_dataset: dict) -> None:
    actions = small_dataset["action"]
    assert actions.min() >= 0
    assert actions.max() < NUM_ACTIONS


def test_actions_are_valid_under_mask(small_dataset: dict) -> None:
    """모든 저장된 action은 그 step의 mask에서 True여야 한다 — BC의 전제."""
    actions = small_dataset["action"]
    masks = small_dataset["mask"]
    for i, a in enumerate(actions):
        assert masks[i, int(a)], f"sample {i}: action {int(a)} not valid in mask"


def test_obs_value_range(small_dataset: dict) -> None:
    obs = small_dataset["obs"]
    # 보드는 1..9 또는 0(제거됨) → /9 후 [0, 1].
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0


def test_episode_id_consistency(small_dataset: dict) -> None:
    """episode_id는 0..episodes-1을 단조 비감소로 사용."""
    ep_ids = small_dataset["episode_id"]
    assert ep_ids.min() == 0
    assert ep_ids.max() <= 4
    assert np.all(np.diff(ep_ids) >= 0)


def test_reproducible(tmp_path: Path) -> None:
    """같은 seed면 obs/action/mask가 완전히 동일해야 함."""
    p1 = generate(episodes=3, seed=7, output_dir=tmp_path / "a")
    p2 = generate(episodes=3, seed=7, output_dir=tmp_path / "b")
    with np.load(p1) as d1, np.load(p2) as d2:
        assert np.array_equal(d1["obs"], d2["obs"])
        assert np.array_equal(d1["action"], d2["action"])
        assert np.array_equal(d1["mask"], d2["mask"])
        assert np.array_equal(d1["scores"], d2["scores"])
