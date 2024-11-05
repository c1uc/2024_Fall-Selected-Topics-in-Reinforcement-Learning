import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque

from torch.utils.tensorboard import SummaryWriter
from replay_buffer.gae_replay_buffer import GaeSampleMemory
from base_agent import PPOBaseAgent
from models.atari_model import AtariNet
import gym

import wandb


class AtariPPOAgent(PPOBaseAgent):
    def __init__(self, config):
        super(AtariPPOAgent, self).__init__(config)
        ### TODO ###
        # initialize env
        def make_env(noop_max=30):
            env = gym.make(config["env_id"], render_mode="rgb_array")
            env = gym.wrappers.AtariPreprocessing(env, frame_skip=1, noop_max=noop_max)
            env = gym.wrappers.FrameStack(env, 4)
            return env

        self.num_envs = config["num_envs"]
        self.env = gym.vector.AsyncVectorEnv([make_env for _ in range(self.num_envs)])

        ### TODO ###
        # initialize test_env
        self.test_env = make_env(noop_max=0)

        self.net = AtariNet(self.env.single_action_space.n)
        self.net.to(self.device)
        self.lr = config["learning_rate"]
        self.update_count = config["update_ppo_epoch"]
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def decide_agent_actions(self, observation, evaluation=False):
        ### TODO ###
        # add batch dimension in observation
        # get action, value, logp from net

        observation = np.asarray(observation)
        if observation.ndim == 3:
            observation = observation[np.newaxis, :]

        observation = torch.tensor(observation).to(self.device)
        if evaluation:
            with torch.no_grad():
                act, lgp, val, _ = self.net(observation, evaluation=evaluation)
        else:
            act, lgp, val, _ = self.net(observation, evaluation=evaluation)

        return (
            act.detach().cpu().numpy(),
            val.detach().cpu().numpy(),
            lgp.detach().cpu().numpy(),
        )

    def update(self):
        loss_counter = 0.0001
        total_surrogate_loss = 0
        total_v_loss = 0
        total_entropy = 0
        total_loss = 0

        batches = self.gae_replay_buffer.extract_batch(
            self.discount_factor_gamma, self.discount_factor_lambda
        )
        sample_count = len(batches["action"])
        batch_index = np.random.permutation(sample_count)

        observation_batch = {}
        for key in batches["observation"]:
            observation_batch[key] = batches["observation"][key][batch_index]
        action_batch = batches["action"][batch_index]
        return_batch = batches["return"][batch_index]
        adv_batch = batches["adv"][batch_index]
        v_batch = batches["value"][batch_index]
        logp_pi_batch = batches["logp_pi"][batch_index]

        for _ in range(self.update_count):
            for start in range(0, sample_count, self.batch_size):
                ob_train_batch = {}
                for key in observation_batch:
                    ob_train_batch[key] = observation_batch[key][
                        start : start + self.batch_size
                    ]
                ac_train_batch = action_batch[start : start + self.batch_size]
                return_train_batch = return_batch[start : start + self.batch_size]
                adv_train_batch = adv_batch[start : start + self.batch_size]
                v_train_batch = v_batch[start : start + self.batch_size]
                logp_pi_train_batch = logp_pi_batch[start : start + self.batch_size]

                ob_train_batch = torch.from_numpy(ob_train_batch["observation_2d"])
                ob_train_batch = ob_train_batch.to(self.device, dtype=torch.float32)
                ac_train_batch = torch.from_numpy(ac_train_batch)
                ac_train_batch = ac_train_batch.to(self.device, dtype=torch.long)
                adv_train_batch = torch.from_numpy(adv_train_batch)
                adv_train_batch = adv_train_batch.to(self.device, dtype=torch.float32)
                logp_pi_train_batch = torch.from_numpy(logp_pi_train_batch)
                logp_pi_train_batch = logp_pi_train_batch.to(
                    self.device, dtype=torch.float32
                )
                return_train_batch = torch.from_numpy(return_train_batch)
                return_train_batch = return_train_batch.to(
                    self.device, dtype=torch.float32
                )

                ### TODO ###
                # calculate loss and update network
                action, log_p, val, entropy = self.net(ob_train_batch, a=ac_train_batch)

                # calculate policy loss
                ratio = torch.exp(log_p - logp_pi_train_batch)
                surrogate_loss = -torch.min(
                    ratio * adv_train_batch,
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * adv_train_batch,
                ).mean()

                # calculate value loss
                value_criterion = nn.MSELoss()
                v_loss = value_criterion(val, return_train_batch)

                # calculate total loss
                entropy = entropy.mean()
                loss = (
                    surrogate_loss
                    + self.value_coefficient * v_loss
                    - self.entropy_coefficient * entropy
                )

                # update network
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
                self.optim.step()

                total_surrogate_loss += surrogate_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += entropy.item()
                total_loss += loss.item()
                loss_counter += 1

        self.writer.add_scalar(
            "PPO/Loss", total_loss / loss_counter, self.total_time_step
        )
        self.writer.add_scalar(
            "PPO/Surrogate Loss",
            total_surrogate_loss / loss_counter,
            self.total_time_step,
        )
        self.writer.add_scalar(
            "PPO/Value Loss", total_v_loss / loss_counter, self.total_time_step
        )
        self.writer.add_scalar(
            "PPO/Entropy", total_entropy / loss_counter, self.total_time_step
        )

        wandb.log(
            {
                "PPO/Loss": total_loss / loss_counter,
                "PPO/Surrogate Loss": total_surrogate_loss / loss_counter,
                "PPO/Value Loss": total_v_loss / loss_counter,
                "PPO/Entropy": total_entropy / loss_counter,
            },
            step=self.total_time_step,
        )

        print(
            f"Loss: {total_loss / loss_counter}\
			\tSurrogate Loss: {total_surrogate_loss / loss_counter}\
			\tValue Loss: {total_v_loss / loss_counter}\
			\tEntropy: {total_entropy / loss_counter}\
			"
        )

    def train(self):
        ob, _ = self.env.reset()
        episode_rewards = np.zeros(self.num_envs)
        episode_lens = np.zeros(self.num_envs)

        while self.total_time_step <= self.training_steps:
            act, val, lgp = self.decide_agent_actions(ob)
            next_ob, rew, terminates, truncates, _ = self.env.step(act)

            for i in range(self.num_envs):
                obs = {}
                obs["observation_2d"] = np.asarray(ob[i], dtype=np.float32)
                self.gae_replay_buffer.append(
                    i,
                    {
                        "observation": obs,  # shape = (4,84,84)
                        "action": np.array(act[i]),  # shape = (1,)
                        "reward": rew[i],  # shape = ()
                        "value": val[i],  # shape = ()
                        "logp_pi": lgp[i],  # shape = ()
                        "done": terminates[i],  # shape = ()
                    },
                )

                if len(self.gae_replay_buffer) >= self.update_sample_count:
                    self.update()
                    self.gae_replay_buffer.clear_buffer()
            episode_rewards += rew
            episode_lens += 1

            for i in range(self.num_envs):
                if terminates[i] or truncates[i]:
                    if i == 0:
                        self.writer.add_scalar(
                            "Train/Episode Reward",
                            episode_rewards[i],
                            self.total_time_step,
                        )
                        self.writer.add_scalar(
                            "Train/Episode Len", episode_lens[i], self.total_time_step
                        )
                        wandb.log(
                            {
                                "Train/Reward": episode_rewards[i],
                                "Train/Episode_Len": episode_lens[i],
                            },
                            step=self.total_time_step
                        )
                    print(
                        f"env[{i}]: [{len(self.gae_replay_buffer)}/{self.update_sample_count}][{self.total_time_step}/{self.training_steps}] episode reward: {episode_rewards[i]}  episode len: {episode_lens[i]}"
                    )
                    episode_rewards[i] = 0
                    episode_lens[i] = 0

            ob = next_ob
            self.total_time_step += self.num_envs

            if self.total_time_step % self.eval_interval == 0:
                # save model checkpoint
                avg_score = self.evaluate()
                self.save(
                    os.path.join(
                        self.writer.log_dir,
                        f"model_{self.total_time_step}_{int(avg_score)}.pth",
                    )
                )
                self.writer.add_scalar(
                    "Evaluate/Episode Reward", avg_score, self.total_time_step
                )
                wandb.log(
                    {
                        "Evaluate/Reward": avg_score
                    },
                    step=self.total_time_step
                )
