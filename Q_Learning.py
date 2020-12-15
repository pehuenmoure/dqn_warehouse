#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

from warehouse_env.warehouse_env import WarehouseEnv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T    

from PIL import Image

# Code Snippet edited below taken from

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_torch_screen(env, agent_id=0):
    render_array = np.asarray(env.render(agent_id=agent_id, zoom_size=3))[:,:,:3]
    screen = render_array.transpose((2, 0, 1))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0).to(device)

def get_torch_observation(env, agent):
#     screen = np.array(env._observe(agent)).transpose((2, 0, 1))
#     screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    torch_tensor = torch.from_numpy(env._observe(agent))
    return torch_tensor

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def select_optimal_action(state):
    with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        return policy_net(state).max(1)[1].view(1, 1)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    

if name == "__main__":
    # Simple world
    simple_agent =          
             [[ 1,  0,  0,  0,  0,  2, 0],
              [ 0,  0,  0,  0,  0,  0, 0],
              [ 0,  0,  0,  0,  0,  0, 0],
              [ 0,  0,  0,  0,  0,  0, 0],
              [ 0,  0,  0,  0,  0,  0, 0],
              [ 0,  0,  0,  3,  0,  0, 0]]
    simple_world =          
             [[  0,  0,  0,  0,  0,  0, 0],
              [  0,  0,  0,  1,  0,  0, 0],
              [  1,  0,  0,  0,  1,  0, 0],
              [  0,  0,  0,  1,  0,  0, 0],
              [  0,  0,  0,  0,  0,  0, 0],
              [  0,  0,  0,  0,  0,  0, 0]]

    env = WarehouseEnv(agent_map=np.array(simple_agent), 
                 obstacle_map=np.array(simple_world))
    env.assign_goal(agent=0, goal=(5, 5))
    env.assign_goal(agent=1, goal=(4, 0))
    env.assign_goal(agent=2, goal=(2, 2))

    # TRAINING HYPERPARAMETERS
    BATCH_SIZE = 100
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10

    # PEHUEN's WAREHOUSE HYPERPARAMTERS
    NUM_STEPS = 100
    NUM_AGENTS = 3

    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. Typical dimensions at this point are close to 3x40x90
    # which is the result of a clamped and down-scaled render buffer in get_screen()
    init_screen = get_torch_screen(env)
    _, _, screen_height, screen_width = init_screen.shape

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    policy_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net = DQN(screen_height, screen_width, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)


    steps_done = 0
    episode_durations = []

    num_episodes = 100000
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        for agent_id in env.agent_goal.keys():
                env.assign_goal(agent_id, env.get_new_goal_location())
        current_agent_id = 0

        last_screen = get_torch_screen(env=env, agent_id=current_agent_id)
        current_screen = get_torch_screen(env=env, agent_id=current_agent_id)
        state = current_screen # - last_screen
        for t in np.arange(NUM_STEPS):
            # Select and perform an action
            action = select_action(state)
            _, reward, done, _ = env.step(current_agent_id, action.item())
            current_agent_id = (current_agent_id + 1) % NUM_AGENTS
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_torch_screen(env=env, agent_id=current_agent_id)
            if not done:
                next_state = current_screen #- last_screen
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')