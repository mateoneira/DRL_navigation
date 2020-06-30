# Navigation Using Deep Reinforcement Learning 

Implementation of Deep Q-Learning algorithm to train an agent to navigate and collect rewards in a large, square world. 

**Report.pdf** contains model architecture details and comparative training results using:

* DQN
* Double DQN
* Prioritized Experience Replay
* Dueling DQN


### Project Details

* __Goal__: An agent must learn to collect as many yellow bananas while avoiding blue ones.

* __Agent Reward Function__: 
	* +1 for collecting yellow bananas
	* -1 for collecting blue bananas

* __Behaviour Parameters__:
	* State space (_continuous_): 37 dimensions corresponding to agent's velocity, along with ray-based perception of objects around agent's forward direction.
	* Action space (_discrete_): 4 actions:
		- **0**: move forward
		- **1**: move backward
		- **2**: turn left
		- **3**: turn right

The task is episodic, and in order to solve the environment agent must get an average score of +13 over 100 consecutive episodes. 

### Getting Started

* __Dependencies__: Project uses [PyTorch](https://pytorch.org/) and [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents). To set up your python environment follow instruction on following link:
https://github.com/udacity/deep-reinforcement-learning

* __Environment__: Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)  

* Unzip the file inside the same folder as this code. 

### Instructions

* __Train__: To train the agent yourself, run _dqn.py_, with the train argument _-t_ or _-train_. Additional arguments can be passed to especify learning algorithm, these include:
	* _-p_ or _--per_: enables prioritized experience replay
	* _-d_ or _--double_: enables double q-learning
	* _-n_ or _--dueling_: enables dueling DQN architecture
If no arguments are passed the agent is trained using a vanilla deep q-learning algorithm. Bellow is an example of training an agent that incorporates all three modifications in the learning algorith.

```console
python dqn.py -t -p -d -n
```

* __Run trained model__: To run the model with the pre-trained weights (_dueling_dqn.pt_), run _dqn.py_ without any arguments. 

```console
python dqn.py
```

* __Compare Models__: To compare different algorithm implementations open _DQN_comparison.pynb_ jupter notebook provided. Implementations include:
	* DQN
	* DDQN
	* Prioritized experience replay
	* Dueling network