import argparse
import json
from collections import deque
from copy import deepcopy

import gymnasium
import numpy as np
import requests

import stable_baselines3
import cv2
from gymnasium.vector.utils import create_empty_array, concatenate
from gymnasium.wrappers.utils import create_zero_array


def connect(agent, url: str = "http://localhost:5000"):
    while True:
        # Get the observation
        response = requests.get(f"{url}")
        if json.loads(response.text).get("error"):
            print(json.loads(response.text)["error"])
            break
        obs = json.loads(response.text)["observation"]
        obs = np.array(obs).astype(np.uint8)

        # Decide an action based on the observation (Replace this with your RL agent logic)
        action_to_take = agent.act(obs)  # Replace with actual action

        # Send an action and receive new observation, reward, and done status
        response = requests.post(f"{url}", json={"action": action_to_take.tolist()})
        if json.loads(response.text).get("error"):
            print(json.loads(response.text)["error"])
            break

        result = json.loads(response.text)
        terminal = result["terminal"]

        if terminal:
            print("Episode finished.")
            return


class Agent:
    def __init__(self, model_path, stack_size=8):
        self.model = stable_baselines3.PPO.load(model_path, device="cuda")

        self.obs_space = gymnasium.spaces.Box(0, 255, (84, 84), np.uint8)

        padding_value = create_zero_array(self.obs_space)
        self.obs_queue = deque(
            [padding_value for _ in range(stack_size)], maxlen=stack_size
        )

        self.stacked_obs = create_empty_array(self.obs_space, n=stack_size)

    def act(self, obs):
        obs = np.moveaxis(obs, 0, -1)
        obs = np.sum(
            np.multiply(obs, np.array([0.2125, 0.7154, 0.0721])), axis=-1
        ).astype(np.uint8)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)

        self.obs_queue.append(obs)

        updated_obs = deepcopy(
            concatenate(self.obs_space, self.obs_queue, self.stacked_obs)
        )
        return self.model.predict(updated_obs)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:5000",
        help="The url of the server.",
    )
    args = parser.parse_args()

    agent_model_path = "./model.zip"
    rand_agent = Agent(agent_model_path)

    connect(rand_agent, url=args.url)
