from abc import ABC

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from base_agent import DQNBaseAgent
from models.atari_model import AtariNetDQN
import gym
import random

from dqn_agent_atari import AtariDQNAgent


class AtariDDQNAgent(AtariDQNAgent):
    def __init__(self, config):
        super(AtariDDQNAgent, self).__init__(config)

    def update_behavior_network(self):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size, self.device
        )

        ### TODO ###
        # calculate the loss and update the behavior network
        # 1. get Q(s,a) from behavior net
        # 2. get max_a Q(s',a) from target net
        # 3. calculate Q_target = r + gamma * max_a Q(s',a)
        # 4. calculate loss between Q(s,a) and Q_target
        # 5. update behavior net

        action = action.type(torch.LongTensor).to(self.device)
        reward = reward.to(self.device)
        done = done.to(self.device)

        next_action = self.behavior_net(next_state).argmax(1).unsqueeze(1)

        q_value = self.behavior_net(state).gather(1, action)
        with torch.no_grad():
            q_next = self.target_net(next_state).gather(1, next_action)

            # if episode terminates at next_state, then q_target = reward
            q_target = reward + self.gamma * q_next * (1 - done)

        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)

        self.writer.add_scalar(f"{self.algo}/Loss", loss.item(), self.total_time_step)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
