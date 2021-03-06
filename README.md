# Project Details
This is the first project for the Deep Reinforcement Learning Nanodegree.  

# Objective
The objective of this project is to train an agent to collect bananas in a large square world.  We want the agent to collect the yellow bananas and avoid the blue bananas.  The agent has a positive reward of +1 for collecting a yellow banana and a negative reward of -1 for collecting a blue banana.  Success is measured by the agent getting an average score of +13 over 100 consecutive episodes.
![rl_bannana](https://user-images.githubusercontent.com/54339413/177430670-8de2f98f-4ca6-4a8e-9aaa-00027fdfaf82.gif)

## Environment Details
* State Space - state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.
* Action Space - the agent has 4 discrete actions 
  * 0 = Move Forward
  * 1 = Move Backward
  * 2 = Turn Left
  * 3 = Turn Right
<br> When the agent archives an average score of +13 over 100 consecutive episodes, the environment is solved.

# Getting Started
Step1:
Install Anaconda distribution at:
https://www.anaconda.com/

Step2:
Follow the instructions to setup the environment for you system (window, mac, etc.)
https://github.com/udacity/Value-based-methods#dependencies

Step3:
Clone my project repo

```bash
git clone https://github.com/TonyMSc/rl_first_project.git
```

Step4:
Copy the Navagation_main.ipynb notebook and all .py files cloned from the repo and move them to \Value-based-methods\p1_navigation\ folder from the environment you created in Step 2 instructions.


# Instructions
Open the Navagation_main.ipynb notebook and change the file "path env = UnityEnvironment(file_name=".../p1_navigation/Banana_Windows_x86_64/Banana.x86_64")". You should be able to run all the cells (graphs will print out).

### A total of 9 experiments are performed
1. DQN with 2 hidden NN layers, 64 nodes, epsilon start 1.0, decay 0.995
2. Double DQN with 2 hidden NN layers, 64 nodes, epsilon start 1.0, decay 0.995
3. Dueling DQN with 2 hidden NN layers, 64 nodes, epsilon start 1.0, decay 0.995
4. DQN with 3 hidden NN layers, 128 nodes, epsilon start 1.0, decay 0.8
5. Double DQN with 3 hidden NN layers, 128 nodes, epsilon start 1.0, decay 0.8
6. Dueling DQN with 3 hidden NN layers, 128 nodes, epsilon start 1.0, decay 0.8
7. DQN with 2 hidden NN layers, 64 nodes, epsilon start 1.0, decay 0.995, Prioritized Experience Replay
8. Double DQN with 2 hidden NN layers, 64 nodes, epsilon start 1.0, decay 0.995, Prioritized Experience Replay
9. Dueling DQN with 2 hidden NN layers, 64 nodes, epsilon start 1.0, decay 0.995, Prioritized Experience Replay
