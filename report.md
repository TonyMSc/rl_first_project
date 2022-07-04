
#Background
The objective of this project is to train an agent to collect bananas in a large square world.  We want the agent to collect the yellow bananas and avoid the blue bananas.  The agent has a positive reward of +1 for collecting a yellow banana and a negative reward of -1 for collecting a blue banana.  Success is measured by the agent getting an average score of +13 over 100 consecutive episodes.

## Enviornment Details
State Space - state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.
Action Space - the agent has 4 discrete actions 
* State Space - state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.
* Action Space - the agent has 4 discrete actions 
  * 0 = Move Forward
  * 1 = Move Backward
  * 2 = Turn Left
  * 3 = Turn Right

When the agent achives an average score of +13 over 100 consecutive episodes, the enviornment is solved.

The following Algorithms were tested. 

# Learning Algorithm
 
## Deep Q
For deep Q learning we combine temporal difference learning with neural networks.  
### Learning Learning Algorithm
$Q(S_{t}, A_t) = (1-\alpha)Q(S_{t}, A_{t}) + \alpha(R_{t} + \lambda(max_{a}Q(S_{t+1},a)) $
### Model Hyperparameters
An infinate number of hyperparameter combinations can be used in this problem.  Including:
1. number of hidden layers in the neural network
2. number of nodes in each layer in the neural network
3. the optimizer used in the neural network
4. the learning rate of the optimizer
5. starting epsilon for the epsilon greedy policy
6. decay rate of epsilon
7. the mini batch size
8. the buffer size for the replay
9. the gamma rate for the discount factor

To limit the 
### Model Architectures
The neural network architecture is a simple feed forward neural network:  
1. The inputs are the state size (for this problem it is a state space of 37)
2. The hidden layer consists of several fully connected linear layers with a relu activation function
3. The output is the number of actions we can take in the enviornment (for this problem the agent can take 4 actions)
4. The optimizer for this network is Adam with a learning rate of 5e-4
5. The loss function to minimize is the mean squared error of the $Q_{expected}$ and the $Q_{target}$
## Double Deep Q
In Double Deep Q learning we use two identical neural network models to mitigate the oversestimation of the action value function problem in DQN. Two neural network models are now used to build the target
### Learning Learning Algorithm
One DQN is used to select the action \
$a_{t+1}^{\star} := \[\argmax_{a_{t+1}}\] Q_{1}(s_{t+1},a_{t+1})$ \
A second DQN evaluates it \
$R + \gamma Q_{2}(s_{t+1}, a^{\star}_{t+1})$ \
$\newcommand{\argmin}{\mathop{\mathrm{arg\,min}}$
### Model Hyperparameters

### Model Architectures
## Dueling Deep Q
The Dueling Deep Q changes the neural network 
### Learning Learning Algorithm

$Q_{\pi}(s, a) := V_{\pi}(s) + A_{\pi}(s, a)$ \
where \
$Q_{\pi}(s, a)$ is the expected return selecting action a using policy $\pi$ \
$V_{\pi}(s)$ The value of a state is the expected return using policy $\pi$ \
$A_{\pi}(s, a)$ The advantage is the excess expected return of action a above the value of state \
Or written another way: \
$A_{\pi}(s, a) : = Q_{\pi}(s, a) - V_{\pi}(s)$ 
		

### Model Hyperparameters
### Model Architectures
The model architecture changes from a neural network that estimates the Q values for each action (first model) to a neural network that computes the value of a state and the advantages seperatly (second model).  The seperate values are then added together to create the Q value estimates.  This should make the training process faster.
![](images/dueling_dqn.png)
## Prioritized Experience Replay
###Deep Q
### Double Deep Q
### Dueling Deep Q


# Plot of Rewards from Experiments
Results from the experiments are as follows:

# Ideas for Future Work
**Neural Net Architecture**-Possibly use a CNN layer with the feed forward neural net to help identify colors.  Experiment with different number of layers and neuron combinations.  
**Aditional Experiments**-Experiment with with a Dueling DDQN and other extensions of the DQN such as: 
1. Distributional DQN 
2. Noisy DQN 
3. Prioritized DDQN




