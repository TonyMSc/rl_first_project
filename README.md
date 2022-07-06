# Project Details
This is the first project for the Deep Reinforcement Learning Nanodegree.  

# Objective
The objective of this project is to train an agent to collect bananas in a large square world.  We want the agent to collect the yellow bananas and avoid the blue bananas.  The agent has a positive reward of +1 for collecting a yellow banana and a negative reward of -1 for collecting a blue banana.  Success is measured by the agent getting an average score of +13 over 100 consecutive episodes.
![rl_bannana](https://user-images.githubusercontent.com/54339413/177430670-8de2f98f-4ca6-4a8e-9aaa-00027fdfaf82.gif)

## Enviornment Details
* State Space - state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.
* Action Space - the agent has 4 discrete actions 
  * 0 = Move Forward
  * 1 = Move Backward
  * 2 = Turn Left
  * 3 = Turn Right
<br> When the agent achives an average score of +13 over 100 consecutive episodes, the enviornment is solved.

# Getting Started
Step1:
Install Anaconda distribution at:
https://www.anaconda.com/

Step2:
Follow the instrutions to setup the enviornment for you system (window, mac, etc.)
https://github.com/udacity/Value-based-methods#dependencies

Step3:
Clone my project repo
git clone https://github.com/TonyMSc/rl_first_project.git

# Instructions
The main file to train is Navagation.ipynb
After the markdown section "4. It's Your Turn!", change the file "path env = UnityEnvironment(file_name="<your location>/p1_navigation/Banana_Windows_x86_64/Banana.x86_64")"
