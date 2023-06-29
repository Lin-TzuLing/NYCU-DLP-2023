'''DLP DQN Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time
import warnings

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))



class Net(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=(32, 16)):# default hidden_dim=32
        super().__init__()
        self.action_dim = action_dim

        # hidden representation
        self.fc1 = nn.Linear(state_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[0])

        # predict value (state)
        self.value_fc1 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.value_fc2 = nn.Linear(hidden_dim[1], 1)

        # predict action (state, action)
        self.action_fc1 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.action_fc2 = nn.Linear(hidden_dim[1], action_dim)

        self.relu = nn.ReLU()

    def forward(self, x):

        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
    
        value = self.relu(self.value_fc1(out))
        value = self.value_fc2(value)

        action = self.relu(self.action_fc1(out))
        action = self.action_fc2(action)
        
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a')
        action_mean = action.mean(dim=-1, keepdim=True)
        if action_mean.shape[0] == 1:
            action_mean = action_mean.unsqueeze(-1)
        return value + action - action_mean.expand(-1, self.action_dim).squeeze()
        


class DDQN:
    def __init__(self, args):
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())

        ## TODO ##
        self._optimizer = Adam(self._behavior_net.parameters(), lr=args.lr)

        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
         ## TODO ##
         
        self._behavior_net.eval()
        with torch.no_grad():
            input_state = torch.from_numpy(state).to(self.device)
            action_value = self._behavior_net(input_state) 
        self._behavior_net.train()

        # exploitation: input state into behavior net and choose the action with highest probability
        if random.random() > epsilon:   
            with torch.no_grad():         
                action = torch.argmax(action_value).item()
            
        # exploration: random choose an action
        else:
            action = action_space.sample()

        return action


    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, [action], [reward / 10], next_state,
                            [int(done)])

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        ## TODO DDQN ##

        # get minibatch q-value prediction for their corresponding action
        q_value = torch.gather(input=self._behavior_net(state), index=action.long(), dim=1)
        with torch.no_grad():
           # find action with max q-value with behavior net
           q_argmax = torch.max(self._behavior_net(next_state), dim=1)[1].unsqueeze(-1)
           # get q-value with action decided by q_argmax   
           q_next = torch.gather(input=self._target_net(next_state), index=q_argmax, dim=1)
           # *(1-done), if done=True=1, q_target should be reward only
           q_target = reward + gamma * torch.unsqueeze(q_next, dim=1) * (1-done) 
        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)

        # optimize
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 1)
        self._optimizer.step()

    def _update_target_network(self, tau=1e-2):
        '''soft update (DDQN) target network by copying from behavior network'''
        ## TODO ##
        # DDQN soft copy behavior network's parameters as target network's
        # self._target_net.load_state_dict(self._behavior_net.state_dict())
        for target, behavior in zip(self._target_net.parameters(), self._behavior_net.parameters()):
            target.data.copy_(tau * behavior.data + (1.0 - tau) * target.data)


    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'behavior_net': self._behavior_net.state_dict(),
                    'target_net': self._target_net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])


def train(args, env, agent, writer):
    print('Start Training')
    action_space = env.action_space
    total_steps, epsilon = 0, 1.
    ewma_reward = 0
    rewards_deque = deque(maxlen=10)
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        for t in itertools.count(start=1):
            # select action
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                action = agent.select_action(state, epsilon, action_space)
                epsilon = max(epsilon * args.eps_decay, args.eps_min)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update(total_steps)

            state = next_state
            total_reward += reward
            total_steps += 1 

            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                    .format(total_steps, episode, t, total_reward, ewma_reward,
                            epsilon))
                
                rewards_deque.append(total_reward)
                avg_rewards = np.mean(rewards_deque)
                break
        if avg_rewards > 100:
            break
    env.close()


def test(args, env, agent, writer):
    print('Start Testing')
    action_space = env.action_space
    epsilon = args.test_epsilon
    seeds = (args.seed + i for i in range(10))
    rewards = []
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()
        ## TODO ##

        # run an episode
        with torch.no_grad():
            for t in itertools.count(start=1):
                # select action
                action = agent.select_action(state, epsilon, action_space)
                # execute action
                next_state, reward, done, _ = env.step(action)

                total_reward += reward
                state = next_state

                if done:
                    writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                    print("Episode:{}, Reward:{}".format(n_episode, total_reward))
                    break
        #  append reward from an episode
        rewards.append(total_reward)

    print('Average Reward', np.mean(rewards))
    env.close()


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='../checkpoint/ddqn.pth')
    parser.add_argument('--logdir', default='../log/ddqn')
    # train
    parser.add_argument('--warmup', default=10000, type=int)
    parser.add_argument('--episode', default=1600, type=int)
    parser.add_argument('--capacity', default=int(1e5), type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=.0005, type=float)
    parser.add_argument('--eps_decay', default=.996, type=float)
    parser.add_argument('--eps_min', default=.01, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=4, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--seed', default=20200519, type=int)
    parser.add_argument('--test_epsilon', default=.001, type=float)
    args = parser.parse_args()
    # ## arguments ##
    # parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument('-d', '--device', default='cuda')
    # parser.add_argument('-m', '--model', default='../checkpoint/ddqn.pth')
    # parser.add_argument('--logdir', default='../log/ddqn')
    # # train
    # parser.add_argument('--warmup', default=10000, type=int)
    # parser.add_argument('--episode', default=4000, type=int)
    # parser.add_argument('--capacity', default=100000, type=int)
    # parser.add_argument('--batch_size', default=64, type=int)
    # parser.add_argument('--lr', default=.0005, type=float)
    # parser.add_argument('--eps_decay', default=.995, type=float)
    # parser.add_argument('--eps_min', default=.01, type=float)
    # parser.add_argument('--gamma', default=.99, type=float)
    # parser.add_argument('--freq', default=4, type=int)
    # parser.add_argument('--target_freq', default=100, type=int)
    # # test
    # parser.add_argument('--test_only', action='store_true')
    # parser.add_argument('--render', action='store_true')
    # # parser.add_argument('--seed', default=20200519, type=int)
    # parser.add_argument('--seed', default=1, type=int)
    # parser.add_argument('--test_epsilon', default=.001, type=float)
    # args = parser.parse_args()

    ## main ##
    env = gym.make('LunarLander-v2')
    agent = DDQN(args)
    writer = SummaryWriter(args.logdir)
    if not args.test_only:
        train(args, env, agent, writer)
        agent.save(args.model)
    agent.load(args.model)
    test(args, env, agent, writer)


if __name__ == '__main__':
    main()
