import torch

device = torch.device("cpu")

#hyperparameters
BUFFER_SIZE = int(1e5)	# replay buffer size
BATCH_SIZE = 128		# minibatch size
GAMMA = 0.99			# discount factor
TAU = 1e-3				# for soft update of target parameters
LR = 1e-3				# learning rate
UPDATE_EVERY = 4		# how often to update target network

ALPHA = 0.6				# hyperparameter of priority replay buffer
BETA = 0.4				# hyperparameter controls bias introduced by priority replay