import gym
from ppo_agent_atari import AtariPPOAgent

model_path = "./log/Enduro/model_85458944_2355.pth"

config = {
    "gpu": 2,
    "training_steps": 1e9,
    "update_sample_count": 10000,
    "discount_factor_gamma": 0.99,
    "discount_factor_lambda": 0.95,
    "clip_epsilon": 0.2,
    "max_gradient_norm": 0.5,
    "batch_size": 512,
    "logdir": "log/Enduro/",
    "update_ppo_epoch": 3,
    "learning_rate": 2.5e-4,
    "value_coefficient": 0.5,
    "entropy_coefficient": 0.01,
    "horizon": 512,
    "env_id": "ALE/Enduro-v5",
    "eval_interval": 256 * 1024,
    "eval_episode": 5,
    "num_envs": 256,
}


def evaluate_and_save():
    agent = AtariPPOAgent(config)
    agent.load_and_evaluate(model_path)

    obs, _ = agent.test_env.reset()
    done = False
    total_reward = 0
    images = []
    while not done:
        rgb = agent.test_env.render()
        images.append(rgb)

        action, _, _ = agent.decide_agent_actions(obs, evaluation=True)
        (
            obs,
            reward,
            terminate,
            truncate,
            _,
        ) = agent.test_env.step(action[0])
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
    out = cv2.VideoWriter("Enduro.mp4", fourcc, 30.0, (width, height))

    for image in images:
        out.write(image)
    out.release()


if __name__ == "__main__":
    img = evaluate_and_save()
    save_img(img)
