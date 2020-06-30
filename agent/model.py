import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
	"""
	Action value function approximator
	qˆ(S, A, w) ≈ qπ(S, A)

	neural network with 2 hidden layers with rectified linear units to
	estimate acton value fuction for deep q-network
	""" 

	def __init__(self, state_size, action_size, seed, fc1=64,fc2=64, dropout_prob=0.3):
		"""
		Initialize parameters and build MLP

		Args:
			state_size (int): dimensions of state vector
			action_size (int): dimension of action vector
			seed (int): random seed

		Returns: 
			tensors of expected reward for all actions given states q(a|s)
		"""

		super(QNetwork, self).__init__()

		self.seed = torch.manual_seed(seed)

		self.fc1 = nn.Linear(state_size, fc1)
		self.fc2 = nn.Linear(fc1, fc2)
		self.fc3 = nn.Linear(fc2, action_size)
		# self.dropout = nn.Dropout(p=dropout_prob)

	def forward(self, state):
		x = self.fc1(state)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)

		return self.fc3(x)

class DuelingQNetwork(nn.Module):
	"""
	Action value function approximator
	qˆ(S, A, w) ≈ qπ(S, A)

	neural network with 2 hidden layers with rectified linear units to
	estimate acton value fuction for deep q-network
	""" 

	def __init__(self, state_size, action_size, seed, fc1=64,fc2=64, dropout_prob=0.3):
		"""
		Initialize parameters and build MLP

		Args:
			state_size (int): dimensions of state vector
			action_size (int): dimension of action vector
			seed (int): random seed

		Returns: 
			tensors of expected reward for all actions given states q(a|s)
		"""

		super(DuelingQNetwork, self).__init__()

		self.seed = torch.manual_seed(seed)

		self.fc1 = nn.Linear(state_size, fc1)
		self.fc2 = nn.Linear(fc1, fc2)

		self.fc_v = nn.Linear(fc2, 1) 
		self.fc_a = nn.Linear(fc2, action_size)
		# self.dropout = nn.Dropout(p=dropout_prob)

	def forward(self, state):
		x = self.fc1(state)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)

		v = self.fc_v(x)
		a = self.fc_a(x)

		#average operator
		a_scaled = a - a.mean(dim=1, keepdim=True)

		return v + a_scaled