from vectorized_dqn_agent_atari import AtariVectorizedDQNAgent

algo = 'Vectorized_DQN'

env = 'Pacman'
# env = 'Enduro'

if __name__ == '__main__':
    # my hyperparameters, you can change it as you like
    config = {
		"algo": algo,
		"gpu": False,
		"training_steps": 1e8,
		"gamma": 0.99,
		"batch_size": 32,
		"eps_min": 0.1,
		"warmup_steps": 20000,
		"eps_decay": 1000000,
		"eval_epsilon": 0.01,
		"replay_buffer_capacity": 100000,
		"logdir": f'log/{algo}/{env}/',
		"update_freq": 4,
		"update_target_freq": 10000,
		"learning_rate": 0.0000625,
        "eval_interval": 100,
        "eval_episode": 5,
		"env_id": 'ALE/MsPacman-v5' if env == 'Pacman' else 'ALE/Enduro-v5',
		"n_envs": 4,
	}
    agent = AtariVectorizedDQNAgent(config)
    agent.train()