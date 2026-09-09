"""Task-stage and reward helpers kept independent from the simulator."""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class Stage(IntEnum):
    HANDLE = 0
    OPEN_DOOR = 1
    GRASP_FOOD = 2
    PLACE_FOOD = 3


def advance_stage(
    stage: Stage,
    door_angle: float,
    handle_distance: float,
    tactile_max: float,
    food_distance: float,
) -> Stage:
    """Advance monotonically to prevent reward hacking by closing the door."""
    if stage == Stage.HANDLE and door_angle > 0.12 and handle_distance < 0.12:
        return Stage.OPEN_DOOR
    if stage == Stage.OPEN_DOOR and door_angle > 0.72:
        return Stage.GRASP_FOOD
    if stage == Stage.GRASP_FOOD and tactile_max > 0.2 and food_distance < 0.09:
        return Stage.PLACE_FOOD
    return stage


def shaped_reward(
    stage: Stage,
    hand_to_handle: float,
    door_angle: float,
    hand_to_food: float,
    food_to_target: float,
    tactile_max: float,
    action: np.ndarray,
) -> float:
    """Dense, bounded objective with a small control penalty."""
    reward = -0.002 * float(np.dot(action, action))
    if stage == Stage.HANDLE:
        reward += 1.0 - np.tanh(5.0 * hand_to_handle)
    elif stage == Stage.OPEN_DOOR:
        reward += 2.0 * np.clip(door_angle / 0.9, 0.0, 1.0)
    elif stage == Stage.GRASP_FOOD:
        reward += 1.0 - np.tanh(5.0 * hand_to_food) + 0.4 * min(tactile_max, 1.0)
    else:
        reward += 2.0 * (1.0 - np.tanh(6.0 * food_to_target)) + 0.2 * min(tactile_max, 1.0)
    return float(reward)
