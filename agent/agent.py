import numpy as np 
import random
from .model import QNetwork, DuelingQNetwork
import torch
import torch.nn.functional as F 
import torch.optim as optim
from .memory import ReplayBuffer, PriorityReplayBuffer

from .params import *

class Agent():
	"""
	Agent using dqn to interact with environment
	"""

	def __init__(self, state_size, action_size, PER= False, doubleQN = False, dueling_network = False, seed=42):
		"""
		Initialize an Agent object

		Args:
			state_size (int): dimension of state vector
			action_size (int): dimension of action vector
			seed (int): random seed

		Returns: 
			action (int): action to take given state
		"""

		self.state_size = state_size
		self.action_size = action_size

		self.seed = random.seed(seed)
		self.PER = PER
		self.doubleQN = doubleQN

		#Q-Network
		if dueling_network:
			self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
			self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
		else:
			self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
			self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)

		#set target network to eval, so dropout is not applied
		# self.qnetwork_target.eval()

		self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

		#Memory for Prioritized Experience Replay
		if PER:
			self.memory = PriorityReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, ALPHA,seed)
		else: 
			self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE,seed)
		self.beta = BETA

		#initialize time step, to keep track of when to update target network
		self.t_step = 0

	def act(self, state, eps=0):
		"""
		Return action for given state

		Args:
			state (array_like): current state
			eps (float): epsilon, for epsilon-greedy action selection

		Returns:
			action (int): action to take given state and epsilon-greedy action selection
		"""
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		self.qnetwork_local.eval()
		with torch.no_grad():
			action_values = self.qnetwork_local(state)
		self.qnetwork_local.train()

		#epsilon greedy-selection
		if random.random() > eps:
			return np.argmax(action_values.cpu().data.numpy())
		else:
			return random.choice(np.arange(self.action_size))


	def step(self, state, action, reward, next_state, done):
		# save experience to replay memory
		self.memory.add(state, action, reward, next_state, done)

		# learn every UPDATE_EVERY time steps
		self.t_step = (self.t_step + 1) % UPDATE_EVERY
		if self.t_step == 0:
			# if enough smaples are available in memory, sample memory and learn
			if len(self.memory) > BATCH_SIZE:
				if self.PER:
					experiences = self.memory.sample(self.beta)
					#increase beta to get closer to 1, anneal linearly
					self.beta = min(1, self.beta + 5e-4)
				else:
					experiences = self.memory.sample()
				self.learn(experiences, GAMMA)


	def learn(self, experiences, gamma):
		"""
		Update mlp parameters using given batch of experience tuples

		Args:
			experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
			gamma (float): discount factor
		"""
		if self.PER:
			states, actions, rewards, next_states, dones, weights, idx = experiences
		else:
			states, actions, rewards, next_states, dones = experiences

		
		# get max predicted q values from target model
		if self.doubleQN:
			argmax_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
			q_target_next = self.qnetwork_target(next_states).gather(1, argmax_actions)

		else:
			q_target_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

		# compute target
		q_targets = rewards + (gamma * q_target_next * (1-dones))

		# get expected q values
		q_expected = self.qnetwork_local(states).gather(1,actions)

		# compute loss
		if self.PER:
			loss, deltas = self.calc_weighted_loss(q_expected, q_targets, weights)
			#update priorities in memory
			self.memory.update_priorities(idx, deltas)
		else:
			loss = F.mse_loss(q_expected, q_targets)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU) 

	def calc_weighted_loss(self, q_expected, q_targets, weights):
		loss = weights * (q_expected-q_targets)**2
		return torch.mean(loss), loss.detach().numpy() + 1e-5

	def soft_update(self, local_model, target_model, tau):
		"""Soft update model parameters.
		θ_target = τ*θ_local + (1 - τ)*θ_target

		Params
		======
		    local_model (PyTorch model): weights will be copied from
		    target_model (PyTorch model): weights will be copied tos
		    tau (float): interpolation parameter 
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

	def save_model(self, path):
		torch.save(self.qnetwork_local.state_dict(), path)

	def load_model(self, path):
		self.qnetwork_local.load_state_dict(torch.load(path))
		self.qnetwork_target.load_state_dict(torch.load(path))






