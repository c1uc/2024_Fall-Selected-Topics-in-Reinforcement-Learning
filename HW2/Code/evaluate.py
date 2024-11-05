import gym
from dqn_agent_atari import AtariDQNAgent

model_path = "../log/DQN/Pacman/model_14838016_4482.pth"

algo = "DQN"
env = "Pacman"

config = {
    "algo": algo,
    "gpu": True,
    "training_steps": 2e7,
    "gamma": 0.99,
    "batch_size": 32,
    "eps_min": 0.1,
    "warmup_steps": 20000,
    "eps_decay": 1000000,
    "eval_epsilon": 0.01,
    "replay_buffer_capacity": 100000,
    "logdir": f"log/{algo}/{env}/",
    "update_freq": 4,
    "update_target_freq": 10000,
    "learning_rate": 0.0000625,
    "eval_interval": 100,
    "eval_episode": 5,
    "env_id": "ALE/MsPacman-v5" if env == "Pacman" else "ALE/Enduro-v5",
}


def evaluate_and_save():
    agent = AtariDQNAgent(config)
    agent.load_and_evaluate(model_path)

    obs, _ = agent.test_env.reset()
    done = False
    total_reward = 0
    images = []
    while not done:
        rgb = agent.test_env.render()
        images.append(rgb)

        action = agent.decide_agent_actions(
            obs, agent.eval_epsilon, agent.test_env.action_space
        )
        obs, reward, terminate, truncate, _ = agent.test_env.step(action)
        done = terminate or truncate

        total_reward += reward

    print(f"Total reward: {total_reward}")
    agent.test_env.close()

    return images


def save_img(images):
    import cv2
    import numpy as np

    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("Pacman.mp4", fourcc, 30.0, (width, height))

    for image in images:
        out.write(image)
    out.release()


if __name__ == "__main__":
    img = evaluate_and_save()
    save_img(img)
