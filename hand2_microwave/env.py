"""Gymnasium environment for a 7-DoF arm and Hand2-style tactile hand."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from mujoco import viewer

from hand2_microwave.rewards import Stage, advance_stage, shaped_reward


ASSET_PATH = Path(__file__).parent / "assets" / "scene.xml"
ARM_JOINTS = [f"arm_{index}" for index in range(1, 8)]
FINGER_JOINTS = [f"{finger}_{joint}" for finger in ("thumb", "index", "middle", "ring", "little") for joint in ("mcp", "pip")]
ACTUATORS = ARM_JOINTS + FINGER_JOINTS
TACTILE_SENSORS = [f"tactile_{finger}" for finger in ("thumb", "index", "middle", "ring", "little")]


class Hand2MicrowaveEnv(gym.Env[np.ndarray, np.ndarray]):
    """Open a microwave and place a drumstick inside using position actuators."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 10,
        max_episode_steps: int = 500,
    ) -> None:
        if frame_skip < 1:
            raise ValueError("frame_skip must be at least one")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be at least one")
        self.model = mujoco.MjModel.from_xml_path(str(ASSET_PATH))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self._renderer: mujoco.Renderer | None = None
        self._viewer = None
        self._actuator_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATORS])
        self._joint_qpos = np.array([self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in ACTUATORS])
        self._door_qpos = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "door_hinge")]
        self._food_qpos = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "drumstick_free")]
        self._tactile_ids = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name) for name in TACTILE_SENSORS])
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(ACTUATORS),), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(51,), dtype=np.float32)
        self.stage = Stage.HANDLE

    def _site(self, name: str) -> np.ndarray:
        return self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)].copy()

    def _observation(self) -> np.ndarray:
        qpos = self.data.qpos[self._joint_qpos]
        qvel = self.data.qvel[self.model.jnt_dofadr[[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ACTUATORS]]]
        food = self.data.qpos[self._food_qpos : self._food_qpos + 3]
        palm = self._site("palm_site")
        target = self._site("cavity_target")
        tactile = np.clip(self.data.sensordata[self._tactile_ids] / 20.0, 0.0, 1.0)
        door_angle = self.data.qpos[self._door_qpos]
        return np.concatenate(
            (
                qpos,
                qvel,
                food,
                food - palm,
                food - target,
                [np.sin(door_angle), np.cos(door_angle)],
                tactile,
                [float(self.stage) / 3.0],
            )
        ).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._joint_qpos] = np.array([0, -0.45, 0, -1.4, 0, 1.0, 0.3] + [0.08, 0.12] * 5)
        self.data.qpos[self._food_qpos : self._food_qpos + 3] = np.array([0.34, self.np_random.uniform(-0.08, 0.08), 0.16])
        self.data.qpos[self._food_qpos + 3 : self._food_qpos + 7] = np.array([1, 0, 0, 0])
        self.stage = Stage.HANDLE
        self._elapsed_steps = 0
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[self._actuator_ids] = self.data.qpos[self._joint_qpos]
        return self._observation(), {"stage": self.stage.name, "success": False}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape:
            raise ValueError(f"expected action shape {self.action_space.shape}, got {action.shape}")
        action = np.clip(action, -1.0, 1.0)
        target = self.data.qpos[self._joint_qpos] + action * 0.04
        ranges = self.model.actuator_ctrlrange[self._actuator_ids]
        self.data.ctrl[self._actuator_ids] = np.clip(target, ranges[:, 0], ranges[:, 1])
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._elapsed_steps += 1
        food = self.data.qpos[self._food_qpos : self._food_qpos + 3]
        target_site = self._site("cavity_target")
        tactile = np.clip(self.data.sensordata[self._tactile_ids] / 20.0, 0.0, 1.0)
        hand_to_handle = np.linalg.norm(self._site("palm_site") - self._site("handle_site"))
        hand_to_food = np.linalg.norm(food - self._site("palm_site"))
        self.stage = advance_stage(
            self.stage,
            self.data.qpos[self._door_qpos],
            hand_to_handle,
            float(tactile.max()),
            hand_to_food,
        )
        food_to_target = np.linalg.norm(food - target_site)
        reward = shaped_reward(self.stage, hand_to_handle, self.data.qpos[self._door_qpos], hand_to_food, food_to_target, float(tactile.max()), action)
        food_dof = self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "drumstick_free")]
        success = bool(self.data.qpos[self._door_qpos] > 0.72 and food_to_target < 0.08 and np.linalg.norm(self.data.qvel[food_dof : food_dof + 3]) < 0.3)
        if success:
            reward += 10.0
        terminated = success or bool(food[2] < 0.02)
        truncated = self._elapsed_steps >= self.max_episode_steps
        if self.render_mode == "human":
            self.render()
        return self._observation(), reward, terminated, truncated, {
            "success": success,
            "stage": self.stage.name,
            "door_angle": float(self.data.qpos[self._door_qpos]),
            "food_to_target": float(food_to_target),
        }

    def render(self):
        if self.render_mode == "rgb_array":
            self._renderer = self._renderer or mujoco.Renderer(self.model)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        if self._viewer is not None:
            self._viewer.close()
