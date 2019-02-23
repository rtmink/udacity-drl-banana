# Udacity DeepRL Banana Collector

## Background
This project involves training an agent to navigate in a squared world and collect as many yellow bananas as possible while avoiding blue bananas.

### State Space
There are 37 dimensions comprised of the agent's velocity as well as ray-based perception of objects around the target's forward direction.

### Action Space
There are 4 discrete actions as follows:
* ```0``` - move forward
* ```1``` - move backward
* ```2``` - turn left
* ```3``` - turn right

### Reward
* +1 for collecting a yellow banana
* -1 for collecting a blue banana

### Benchmark Mean Reward
The environment is considered solved for an average reward of +13 over 100 consecutive episodes.


## Installation
Follow this link to get started:
https://github.com/udacity/deep-reinforcement-learning#dependencies

Then execute the following commands to access the necessary files:

`cd deep-reinforcement-learning`

`git clone git@github.com:rtmink/udacity-drl-banana.git`

`cd drl-banana`

`unzip Banana.app.zip`

## Training & Report
Refer to `Navigation_train.ipynb` to see how the agent is implemented which includes the model architecture of the neural network. A plot of rewards per episode is also shown to show the number of episodes needed to solve the environment. Lastly, it highlights ideas for future work.

## Evaluating
Refer to `Navigation_eval.ipynb` to see how the trained agent performs in Unity Banana Collector environment built for Udacity.
