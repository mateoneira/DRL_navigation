from collections import deque
from collections import namedtuple
from itertools import accumulate
import torch
import numpy as np
import random

from .params import *

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PriorityReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, alpha,seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            alpha (float): (0,1) controls how much priority to give, if = 0, sampling is uniform
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.priorities = deque(maxlen=buffer_size)
        self.alpha = alpha
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

        #set priority as 1, this will get updated once we calculate TD
        self.priorities.append(1.0)
    
    def sample(self, beta=0.4):
        """sample using priorities"""
        priorities_alpha = np.array(self.priorities)**self.alpha
        probs = priorities_alpha / priorities_alpha.sum()

        idx = np.random.choice(len(self.memory), self.batch_size, p=probs) # index of samples

        states = torch.from_numpy(np.vstack([self.memory[i].state for i in idx])).float().to(device)
        actions = torch.from_numpy(np.vstack([self.memory[i].action for i in idx])).long().to(device)
        rewards = torch.from_numpy(np.vstack([self.memory[i].reward for i in idx])).float().to(device)
        next_states = torch.from_numpy(np.vstack([self.memory[i].next_state for i in idx])).float().to(device)
        dones = torch.from_numpy(np.vstack([self.memory[i].done for i in idx]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(np.vstack((self.batch_size * probs[idx])**(-beta))).float().to(device)

        #weights need to be normalized to stability
        #this way weights are only scaled downward
        weights /= weights.max()


        return (states, actions, rewards, next_states, dones, weights, idx)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority[0]
            break

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)