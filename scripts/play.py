"""Visualize a trained policy."""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO

from hand2_microwave.env import Hand2MicrowaveEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    env = Hand2MicrowaveEnv(render_mode="human")
    model = PPO.load(args.model)
    observation, _ = env.reset()
    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset()


if __name__ == "__main__":
    main()
