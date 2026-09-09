import numpy as np
import pytest

from hand2_microwave.env import Hand2MicrowaveEnv
from hand2_microwave.rewards import Stage, advance_stage, shaped_reward


def test_stages_only_advance():
    assert advance_stage(Stage.HANDLE, 0.2, 0.1, 0.0, 1.0) == Stage.OPEN_DOOR
    assert advance_stage(Stage.HANDLE, 0.2, 0.2, 0.0, 1.0) == Stage.HANDLE
    assert advance_stage(Stage.OPEN_DOOR, 0.8, 1.0, 0.0, 1.0) == Stage.GRASP_FOOD
    assert advance_stage(Stage.PLACE_FOOD, 0.0, 1.0, 0.0, 1.0) == Stage.PLACE_FOOD


def test_reward_prefers_near_target_during_place():
    near = shaped_reward(Stage.PLACE_FOOD, 1, 1, 1, 0.01, 0, np.zeros(17))
    far = shaped_reward(Stage.PLACE_FOOD, 1, 1, 1, 1.0, 0, np.zeros(17))
    assert near > far


def test_environment_reset_is_seeded_and_observation_is_valid():
    env = Hand2MicrowaveEnv()
    first, info = env.reset(seed=123)
    second, _ = env.reset(seed=123)

    assert info == {"stage": "HANDLE", "success": False}
    assert env.observation_space.contains(first)
    np.testing.assert_allclose(first, second)
    env.close()


def test_environment_enforces_action_shape_and_episode_horizon():
    env = Hand2MicrowaveEnv(max_episode_steps=1)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="expected action shape"):
        env.step(np.zeros(1, dtype=np.float32))

    observation, _, terminated, truncated, info = env.step(np.zeros(17, dtype=np.float32))
    assert env.observation_space.contains(observation)
    assert not terminated
    assert truncated
    assert "food_to_target" in info
    env.close()
