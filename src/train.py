from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import random
from copy import deepcopy
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

class ProjectAgent:
    def __init__(self):

        # these hyperparameters can be tuned and I tried different combinations
        # to get models with different "biases" in order to make them vote at the end
        # to avoid overfitting on the default patient

        config = {'nb_actions': env.action_space.n,
                'learning_rate': 0.001, # IMPORTANT
                'gamma': 0.98,
                'buffer_size': 1000000,
                'epsilon_min': 0.05, # IMPORTANT
                'epsilon_max': 1.,
                'epsilon_decay_period': 15000,
                'epsilon_delay_decay': 300,
                'batch_size': 1024, # IMPORTANT
                'gradient_steps': 5,
                'update_target_strategy': 'ema', # 'replace' or 'ema'
                'update_target_freq': 100, # set between 10 and 1000
                'update_target_tau': 0.001,
                'criterion': torch.nn.SmoothL1Loss(),
                'state_dim': env.observation_space.shape[0],
        }

        # self.device = "cuda" if next(self.policy.parameters()).is_cuda else "cpu"
        # for github grading use cpu
        self.device = "cpu"
        # self.scalar_dtype = next(self.policy.parameters()).dtype
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.98
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,self.device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.state_dim = config['state_dim']
        self.nb_neurons = config['nb_neurons'] if 'nb_neurons' in config.keys() else 512
        self.model = torch.nn.Sequential(nn.Linear(self.state_dim, self.nb_neurons),
                                nn.ReLU(),
                                nn.Linear(self.nb_neurons, self.nb_neurons),
                                nn.ReLU(), 
                                nn.Linear(self.nb_neurons, self.nb_neurons),
                                nn.ReLU(),
                                nn.Linear(self.nb_neurons, self.nb_neurons),
                                nn.ReLU(),
                                nn.Linear(self.nb_neurons, self.nb_neurons),
                                nn.ReLU(),
                                nn.Linear(self.nb_neurons, self.nb_actions)).to(device) 
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.steps = 0


        # Best policy obtained by making a lot of different models vote
        self.vote = True
        self.model_512_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'best_models_512')
        self.models_512 = []


    def greedy_action(self, state):
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.model(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()
        
    def greedy_action_vote(self, state):
        device = self.device
        votes = torch.zeros(self.nb_actions)
        with torch.no_grad():
            for model in self.models_512:
                Q = model(torch.Tensor(state).unsqueeze(0).to(device))
                votes[torch.argmax(Q).item()] += 1
        return torch.argmax(votes).item()
        
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.greedy_action(state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            self.steps += 1
            if done or trunc:
                if episode % 25 == 0: # every 25 training eps, do MC eval of mean reward
                    obs, _ = env.reset()
                    done = False
                    truncated = False
                    episode_reward = 0
                    while not done and not truncated:
                        action = self.greedy_action(obs)
                        obs, reward, done, truncated, _ = env.step(action)
                        episode_reward += reward
                    if episode_reward > 2e10: # if it passes the 5th threshold for the default
                        self.save(f"./best_models_512/{episode}_run6.pt")
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0

            else:
                state = next_state
        return episode_return

    def act(self, observation, use_random=False):
        if self.vote:
            return self.greedy_action_vote(observation)
        else:
            return self.greedy_action(observation)

    def save(self, path):
        torch.save(self.model.to('cpu').state_dict(), path)
        self.model.to(self.device)

    def load(self):
        if self.vote:
            for file in os.listdir(self.model_512_path):
                DQN = torch.nn.Sequential(nn.Linear(self.state_dim, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    nn.ReLU(), 
                                    nn.Linear(512, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, self.nb_actions)).to(device)
                DQN.load_state_dict(torch.load(os.path.join(self.model_512_path, file), map_location=self.device))
                self.models_512.append(DQN)
        else:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))