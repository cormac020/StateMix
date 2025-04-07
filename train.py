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
import matplotlib.pyplot as plt
import pandas as pd
import time

from utils import ReplayBuffer
from agent import StateMix

import gym

parser = argparse.ArgumentParser('parameters')
parser.add_argument('--round', '-r', type=int, default=2000, help='training rounds (default: 2000)')
parser.add_argument('--lr_rate', '-lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('--batch_size', '-b', type=int, default=64, help='batch size (default: 64)')
parser.add_argument('--gamma', '-g', type=float, default=0.99, help='discounting factor (default: 0.99)')
parser.add_argument('--action_scale', '-a', type=int, default=25, help='discrete action scale (default: 25)')
parser.add_argument('--env', '-e', type=str, default='BipedalWalker-v3', help='Environment (default: BipedalWalker-v3)')
parser.add_argument('--load', '-l', type=str, default='no', help='load network name in ./model/')
parser.add_argument('--tenacious', '-t', action='store_true', help='make agent tenacious with low reward')

parser.add_argument('--save_interval', '-s', type=int, default=1000, help='interval to save model (default: 1000)')
parser.add_argument('--print_interval', '-d', type=int, default=200, help='interval to print evaluation (default: 200)')
args = parser.parse_args()
print(args)

action_scale = args.action_scale
learning_rate = args.lr_rate
batch_size = args.batch_size
gamma = args.gamma
env_name = args.env
total_round = args.round
iter_size = args.print_interval
reward_redo = args.tenacious

os.makedirs('./model/', exist_ok=True)  # save model
os.makedirs('./data/', exist_ok=True)  # save rewards and training time

gym.logger.set_level(40)  # surpress a warning from gym
env = gym.make(env_name)
# set seed for np, env, and torch to make training repeatable
# seed = 42
# np.random.seed(seed)
# env.seed(seed)
# torch.manual_seed(seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print('observation space:', env.observation_space)
print('action space:', env.action_space)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    agent = StateMix(state_dim, action_dim, action_scale, learning_rate, gamma, device, batch_size).cuda()
else:
    agent = StateMix(state_dim, action_dim, action_scale, learning_rate, gamma, device, batch_size)

# if specified a model, load it
model_path = './model/' + env_name + '_' + args.load + '.pth'
if os.path.isfile(model_path):  # model exists
    agent.load_state_dict(torch.load(model_path))
    print('Found model:', model_path)
model_path = './model/' + env_name + '_' + str(action_scale) + '.pth'
data_path = './data/MIX-' + env_name

replay_buffer = ReplayBuffer(action_dim, device)
# divide continuous action space into discrete actions, according to ACTION_SCALE
real_actions = [np.linspace(env.action_space.low[i], env.action_space.high[i], action_scale)
                for i in range(action_dim)]

# divide TOTAL_ROUND into ITERATION iterations
iteration = int(total_round / iter_size)
reward_list, time_list = [], []  # record rewards in each episode, record time cost
n_epi = 0  # current episode count
start = time.time()  # starting time

# train begins
for it in range(iteration):
    with tqdm.tqdm(total=iter_size, desc='Iteration %d' % it) as pbar:
        for ep in range(iter_size):
            state = env.reset()
            prev_state = np.zeros_like(state)
            done = False
            acc_reward = 0  # accumulated reward in an episode
            while not done:
                action = agent.take_action(prev_state, state) # get action from agent
                next_state, reward, done, _ = env.step(np.array([real_actions[i][action[i]]
                                                                 for i in range(action_dim)]))
                acc_reward += reward

                done_mask = int(done or (reward_redo and reward <= -100))
                reward = -1 if reward_redo and reward <= -100 else reward
                
                # start to update the agent if there are enough samples
                replay_buffer.add_transition((prev_state, state, action, reward, next_state, done_mask))
                agent.update(replay_buffer)
                prev_state, state = state, next_state
            # record data in this episode
            reward_list.append(acc_reward)
            time_list.append(time.time() - start)

            n_epi += 1
            if n_epi % args.save_interval == 0:  # time to save model and data
                torch.save(agent.state_dict(), model_path)
                dataframe = pd.DataFrame({env_name: reward_list, 'time': time_list})  # save training data as csv file
                dataframe.to_csv(data_path + '.csv', index=False, sep=',')
            # update the progress bar
            pbar.set_postfix({
                'lst_r': '%.1f' % acc_reward,
                'avg_r':
                    '%.1f' % np.mean(reward_list[-(ep + 1):])
            })
            pbar.update(1)

torch.save(agent.state_dict(), model_path)
dataframe = pd.DataFrame({env_name: reward_list, 'time': time_list})  # save training data as csv file
dataframe.to_csv(data_path + '.csv', index=False, sep=',')
print('Training time in total:', time_list[-1])

episodes_list = list(range(len(reward_list)))
plt.plot(episodes_list, reward_list)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('MIX on {}'.format(env_name))
plt.savefig(data_path + '_reward.png')
# plt.show()
