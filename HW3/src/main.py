from ppo_agent_atari import AtariPPOAgent

import wandb

if __name__ == "__main__":

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
        "eval_episode": 3,
        "num_envs": 256,
    }

    wandb.init(project="RL_Topic_PPO", config=config)

    agent = AtariPPOAgent(config)
    agent.train()
