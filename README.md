# Udacity Deep Reinforcement Learning Banana Collector

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
1. Follow this link to get started:

https://github.com/udacity/deep-reinforcement-learning#dependencies

2. Navigate to `deep-reinforcement-learning` directory in your `drlnd` environment
```
cd deep-reinforcement-learning
```

3. Clone the repo
```
git clone git@github.com:rtmink/udacity-drl-banana.git
```

4. Navigate to `udacity-drl-banana` folder
```
cd drl-banana
```

5. Unzip the unity environment
```
unzip Banana.app.zip
```

## Training & Report
Run the following command in the `udacity-drl-banana` folder:
```
jupyter notebook
```

In the notebook, refer to `Report.ipynb` to see how the agent is implemented and trained. The implementation includes the model architecture of the neural network. A plot of rewards per episode for each agent is also shown to show the number of episodes needed to solve the environment. Lastly, it highlights ideas for future work.

## Evaluation
Refer to `Navigation.ipynb` to see how the trained agent performs in Unity Banana Collector environment built for Udacity.
