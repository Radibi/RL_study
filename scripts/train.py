"""Train PPO on the Hand2 microwave task."""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from hand2_microwave.env import Hand2MicrowaveEnv


def make_env() -> Monitor:
    return Monitor(Hand2MicrowaveEnv())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--out", default="runs/ppo_hand2_microwave")
    args = parser.parse_args()
    env = make_vec_env(make_env, n_envs=args.n_envs)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=args.out, n_steps=2048, batch_size=256)
    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    model.save(f"{args.out}/final_model")
    env.close()


if __name__ == "__main__":
    main()
