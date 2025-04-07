# -*- coding: utf-8 -*-
"""
@Author: hc
@Date: 2025-03-05
"""

import torch
import numpy as np
import os
import argparse
import tqdm
import pandas as pd

from agent import StateMix
import gym

parser = argparse.ArgumentParser('parameters')
parser.add_argument('--not_render', '-n', action='store_true', help='not render during the evaluation')
parser.add_argument('--round', '-r', type=int, default=100, help='evaluation rounds (default: 100)')
parser.add_argument('--action_scale', '-a', type=int, default=25, help='discrete action scale, \
                    specifying network to load in ./model/ (default: 25)')
parser.add_argument('--env', '-e', type=str, default='BipedalWalker-v3', help='Environment (default: BipedalWalker-v3)')
args = parser.parse_args()
print(args)

action_scale = args.action_scale
env_name = args.env
eva_round = args.round

gym.logger.set_level(40)
env = gym.make(env_name)
# set seed to make evaluation repeatable

# record a video clip of render
# gym.logger.setLevel(gym.logger.ERROR)
# os.makedirs('./data/', exist_ok=True)
# env = gym.wrappers.Monitor(env, directory='./data/', force=True)  

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print('observation space:', env.observation_space)
print('action space:', env.action_space)
print('action space limits:', env.action_space.low, env.action_space.high)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    agent = StateMix(state_dim, action_dim, action_scale, 0, 0, device).cuda()
else:
    agent = StateMix(state_dim, action_dim, action_scale, 0, 0, device)

# if specified a model, load it
model_path = './model/' + env_name + '_' + str(action_scale) + '.pth'
if os.path.isfile(model_path):
    agent.load_state_dict(torch.load(model_path))
    print('Found model:', model_path)
real_actions = [np.linspace(env.action_space.low[i], env.action_space.high[i], action_scale)
                for i in range(action_dim)]

reward_list = []
pbar = tqdm.tqdm(range(eva_round))  # evaluate EVA_ROUND rounds
for n_epi in pbar:
    state = env.reset()
    prev_state = np.zeros_like(state)
    done = False
    acc_reward = 0.0
    while not done:
        if not args.not_render:
            env.render()
        action = agent.take_action(prev_state, state)
        next_state, reward, done, _ = env.step(np.array([real_actions[i][action[i]]
                                                                 for i in range(action_dim)]))
        acc_reward += reward
        prev_state, state = state, next_state
    reward_list.append(acc_reward)
    n_epi += 1
    pbar.set_postfix({
        'round':
            '%d' % n_epi,
        'avg_reward':
            '%.1f' % np.mean(reward_list[-(n_epi + 1):])
    })

print('Mean award in %d evaluations: %f' % (eva_round, np.mean(reward_list)))
reward_list.append(np.mean(reward_list))
dataframe = pd.DataFrame({env_name: reward_list})
dataframe.to_csv('./data/' + env_name + '_' + str(action_scale) + '_evaluation.csv', index=False, sep=',')