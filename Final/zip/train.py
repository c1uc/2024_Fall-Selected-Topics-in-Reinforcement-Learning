import wandb
from racecar_gym.custom_env import CustomEnv
import gymnasium as gym
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecEnv,
    VecMonitor,
)
from stable_baselines3 import PPO, SAC
from wandb.integration.sb3 import WandbCallback
import numpy as np


run = None


def make_env(scenario: str, reset_when_collision: bool, num_envs: int):
    def _init() -> gym.Env:
        env = CustomEnv(
            scenario=scenario,
            reset_when_collision=reset_when_collision,
            render_mode="rgb_array_birds_eye",
            render_options={"width": 128, "height": 128},
        )
        env.metadata["render_fps"] = 30
        env = gym.wrappers.TransformObservation(env, lambda obs: np.moveaxis(obs, 0, -1), observation_space=gym.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8))
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.FrameStackObservation(env, 8)
        return env

    return [lambda: _init() for _ in range(num_envs)]


def make_vec_env(scenario: str, reset_when_collision: bool, num_envs: int) -> VecEnv:
    global run
    envs = make_env(scenario, reset_when_collision, num_envs)
    env = SubprocVecEnv(envs, start_method="fork")
    env = VecMonitor(env)
    return env


def train(
    agent: str,
    scenario: str,
    num_envs: int,
    reset_when_collision: bool,
    total_timesteps: int,
    use_sde: bool,
) -> None:
    global run
    env = make_vec_env(scenario, reset_when_collision, num_envs)
    if agent == "SAC":
        model = SAC(
            "CnnPolicy",
            env,
            device="cuda:2",
            tensorboard_log=f"runs/{run.name}",
            use_sde=use_sde,
        )
    elif agent == "PPO":
        model = PPO(
            "CnnPolicy",
            env,
            device="cuda:2",
            tensorboard_log=f"runs/{run.name}",
            use_sde=use_sde,
            n_steps=1024,
        )
    else:
        raise NotImplementedError
    model.learn(
        total_timesteps=total_timesteps,
        callback=WandbCallback(
            model_save_path=f"trained_models/{run.name}",
            model_save_freq=10_000
        ),
    )
    model.save(f"trained_models/{run.name}/final")


def parse_args() -> dict:
    import argparse, yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/PPO_austria_reset_sde.yml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_run_name(config: dict) -> str:
    import time

    t = time.strftime("%Y-%m-%d-%H-%M-%S")
    s = f"{config['agent']}-{config['scenario']}-{config['num_envs']}-{'reset' if config['reset_when_collision'] else 'no-reset'}-{'sde' if config['use_sde'] else 'no-sde'}"
    return f"{s}-{t}"


if __name__ == "__main__":
    args = parse_args()
    run_name = get_run_name(args)
    run = wandb.init(
        project="racecar-gym",
        config=args,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        name=run_name,
    )
    train(**args)
    run.finish()
