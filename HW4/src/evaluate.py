import gym
from td3_agent_CarRacing import CarRacingTD3Agent

model_path = "./model_748319_805.pth"
seeds=[3128, 6727, 8843, 7021, 2712]

config = {
    "gpu": 0,
    "training_steps": 1e8,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 32,
    "warmup_steps": 1000,
    "total_episode": 100000,
    "lra": 4.5e-5,
    "lrc": 4.5e-5,
    "replay_buffer_capacity": 5000,
    "logdir": f"log/evaluate/",
    "update_freq": 2,
    "eval_interval": 1,
    "eval_episode": 10,
}


def evaluate_and_save():
    agent = CarRacingTD3Agent(config)
    agent.load(model_path)

    for seed in seeds:
        obs, _ = agent.test_env.reset(seed=seed)
        done = False
        total_reward = 0
        images = []
        while not done:
            rgb = agent.test_env.render()
            images.append(rgb)

            action = agent.decide_agent_actions(obs)
            (
                obs,
                reward,
                terminate,
                truncate,
                _,
            ) = agent.test_env.step(action)
            done = terminate or truncate

            total_reward += reward

        print(f"Total reward: {total_reward}")
        save_img(images, f"CarRacingTD3_seed_{seed}_score_{total_reward}.mp4")
    agent.test_env.close()


def save_img(images, fname):
    import cv2

    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(fname, fourcc, 30.0, (width, height))

    for image in images:
        out.write(image)
    out.release()


if __name__ == "__main__":
    evaluate_and_save()
