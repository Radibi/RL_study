import numpy as np

from hand2_microwave.rewards import Stage, advance_stage, shaped_reward


def test_stages_only_advance():
    assert advance_stage(Stage.HANDLE, 0.2, 0.0, 1.0) == Stage.OPEN_DOOR
    assert advance_stage(Stage.OPEN_DOOR, 0.8, 0.0, 1.0) == Stage.GRASP_FOOD
    assert advance_stage(Stage.PLACE_FOOD, 0.0, 0.0, 1.0) == Stage.PLACE_FOOD


def test_reward_prefers_near_target_during_place():
    near = shaped_reward(Stage.PLACE_FOOD, 1, 1, 1, 0.01, 0, np.zeros(17))
    far = shaped_reward(Stage.PLACE_FOOD, 1, 1, 1, 1.0, 0, np.zeros(17))
    assert near > far
