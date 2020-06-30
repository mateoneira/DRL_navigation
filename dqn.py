import sys, getopt
from unityagents import UnityEnvironment
import numpy as np
from agent.agent import Agent
from collections import deque

env = UnityEnvironment(file_name="./Banana_Windows_x86_64/Banana.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

action_size = brain.vector_action_space_size
state_size = len(env_info.vector_observations[0])

short_options = "tpdn"
long_options = ["train","per", "doubleQN", "dueling"]

def train(agent, n_episodes=700, eps_start=1, eps_end=0.01, eps_decay=0.95):
	"""
	Deep Q-Learning

	Params
    ------
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon

     Returns scores (array): returns scores over n_episodes
	"""
	scores = []                        # list containing scores from each episode
	scores_window = deque(maxlen=100)  # last 100 scores
	eps = eps_start                    # initialize epsilon
	for i_episode in range(1, n_episodes+1):
		env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
		state = env_info.vector_observations[0]            # get the current state
		score = 0                                          # initialize the score

		while True:
			action = agent.act(state, eps=eps).astype(int)
			env_info = env.step(action)[brain_name]

			next_state = env_info.vector_observations[0]
			reward = env_info.rewards[0]
			done = env_info.local_done[0] 

			#update agent
			agent.step(state, action, reward, next_state, done)

			score += reward
			state = next_state

			if done:
				break
		scores_window.append(score)       # save most recent score
		scores.append(score)              # save most recent score

		#reduce epsilon
		eps = max(eps_end, eps_decay*eps)

		print('\rEpisode {}\t Score: {:.2f}\t Epsilon: {:.2f}'.format(i_episode, score, eps), end="")

		if i_episode % 100 == 0:
			print('\rEpisode {}\tAverage Score: {:.2f}\t Epsilon: {:.2f}'.format(i_episode, np.mean(scores_window), eps))

	return scores

def play(agent, n_episodes=10):
	for i_episode in range(1, n_episodes+1):
		env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
		state = env_info.vector_observations[0]            # get the current state
		score = 0                                          # initialize the score

		while True:
			action = agent.act(state, eps=0).astype(int)
			env_info = env.step(action)[brain_name]

			next_state = env_info.vector_observations[0]
			reward = env_info.rewards[0]
			done = env_info.local_done[0] 

			score += reward
			state = next_state

			if done:
				print(score)
				break

if __name__ == "__main__":
	PER = False
	doubleQN = False
	duelingQN = False
	train = False
	# try:
	opts, args = getopt.getopt(sys.argv[1:],short_options, long_options)
	for opt, val in opts:
		if opt in ("-t", "--train"):
			train=True
		elif opt in ("-p", "--pre"):
			PER = True
			print("Prioritized Experience Replay enabled")
		elif opt in ("-d", "--doubleQN"):
			doubleQN = True
			print("Double DQN enabled")
		elif opt in ("-n", "--dueling"):
			dueling_network = True
			print("Dueling Network enabled")

	if train:
		agent = Agent(state_size, action_size, PER, doubleQN, dueling_network)
		scores = train(agent)
		np.savetxt("scores.csv", scores, delimiter=",")
		agent.save_model("weights.pt")
	else:
		agent = Agent(state_size, action_size, dueling_network=True)
		agent.load_model("dueling_dqn.pt")
		play(agent)
	env.close()

