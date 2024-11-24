from td3_agent_CarRacing import CarRacingTD3Agent
import wandb
import time

test_type = "td3_original"
run_name = "CarRacing_TD3"
t = time.strftime("%Y%m%d-%H%M%S")

if __name__ == "__main__":
    # my hyperparameters, you can change it as you like
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
        "logdir": f"log/{run_name}/{test_type}/",
        "update_freq": 2,
        "eval_interval": 10,
        "eval_episode": 10,
    }
    wandb.init(project="CarRacing_TD3", name=f"{run_name}_{test_type}_{t}", group=test_type, config=config)
    agent = CarRacingTD3Agent(config)
    agent.train()
