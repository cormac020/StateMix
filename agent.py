# -*- coding: utf-8 -*-
"""
@Author: hc
@Date: 2025-03-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# torch.manual_seed(42)
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, action_scale: int):
        super(QNetwork, self).__init__()
        self.state_mix = nn.Sequential(nn.Linear(state_dim * 2, 256),
                                       nn.ReLU()
                                        )
        self.feature = nn.Sequential(nn.Linear(256, 128),
                                        nn.ReLU(),
                                        )
        # evaluate action advantages on each branch
        self.actions = [nn.Sequential(nn.Linear(128, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, action_scale)
                                      ) for _ in range(action_dim)]
        self.actions = nn.ModuleList(self.actions)
        # module to calculate state value
        self.value = nn.Sequential(nn.Linear(128, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 1)
                                   )

    def forward(self, prev_state, cur_state):
        # catenate two states, use only one mix layer, works better in BipedalWalker-v2, but worse in Humanoid-v2. But both are not good in HalfCheetah-v2 and Ant-v2
        state = torch.cat([prev_state, cur_state], dim=-1)
        state = self.state_mix(state)
        # current method is better in Humanoid-v2
        # prev_state = self.state_mix_1(prev_state)
        # cur_state = self.state_mix_2(cur_state)
        # state = prev_state + cur_state

        feature = self.feature(state)
        actions = torch.stack([head(feature) for head in self.actions])
        value = self.value(feature)
        # with baseline
        maxa = actions.mean(-1).max(0)[0].unsqueeze(-1)
        actions = actions - maxa + value

        # without baseline
        # maxa = actions.max(-1)[0].unsqueeze(-1)
        # actions = actions - maxa  + value

        return actions


class StateMix(nn.Module):
    def __init__(self, state_dim, action_dim, action_scale, learning_rate, gamma, device, batch_size=16):
        """
        agent for discrete action space
        """
        super(StateMix, self).__init__()

        self.device = device
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.q = QNetwork(state_dim, action_dim, action_scale).to(device)
        self.target_q = QNetwork(state_dim, action_dim, action_scale).to(device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam([{'params': self.q.state_mix.parameters(), 'lr': learning_rate / action_dim},
                                     {'params': self.q.feature.parameters(), 'lr': learning_rate / action_dim},
                                     {'params': self.q.value.parameters(), 'lr': learning_rate / action_dim},
                                     {'params': self.q.actions.parameters(), 'lr': learning_rate}])
        self.update_freq = 1000
        self.update_count = 0
        self.epsilon = 1.0
        self.epsilon_decay = 0.99

        self.gamma = gamma # discount factor
        self.batch_size = batch_size
        self.buffer_threshold = 3000  # start to update the agent if there are enough samples in replay buffer

    def take_action(self, prev_state, state):
        # epsilon greedy
        self.epsilon = max(0.001, self.epsilon_decay * self.epsilon)
        if self.epsilon > torch.rand(1).item():
            action = torch.randint(0, self.action_scale, (self.action_dim,)).tolist()
        else:
            prev_state = torch.tensor(prev_state).float().reshape(1, -1).to(self.device)
            state = torch.tensor(state).float().reshape(1, -1).to(self.device)
            action_value = self.q(prev_state, state)  # get the index of the max value in each row
            action = [int(x.max(1)[1]) for x in action_value]
        return action

    def update(self, replay_buffer):
        if replay_buffer.size() < self.buffer_threshold:
            return None
        prev_state, state, actions, reward, next_state, done_mask = replay_buffer.sample(self.batch_size)
        actions = torch.stack(actions).transpose(0, 1).unsqueeze(-1)
        done_mask = torch.abs(done_mask - 1)

        q_values = self.q(prev_state, state).transpose(0, 1)  # q_values for all possible actions
        q_values = q_values.gather(2, actions.long()).squeeze(-1)  # get q_values for current action

        # select best actions from Q and calculate Q-values in target Q; DDQN
        max_next_action = self.q(state, next_state).transpose(0, 1).max(-1, keepdim=True)[1]
        next_q_values = self.target_q(state, next_state).transpose(0, 1)
        max_next_q_values = next_q_values.gather(2, max_next_action.long()).squeeze(-1)  # get q_values for next action
        
        # next_q_values = self.target_q(next_state).transpose(0, 1)  # normal dqn
        # max_next_q_values = next_q_values.max(-1, keepdim=True)[0].squeeze(-1)

        q_target = (done_mask * self.gamma * max_next_q_values + reward)

        loss = F.mse_loss(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward(torch.ones_like(loss))
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.update_freq == 0:
            self.update_count = 0
            self.target_q.load_state_dict(self.q.state_dict())

        return loss
