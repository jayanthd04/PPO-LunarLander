# PPO-LunarLander
This repo implements Proximal Policy Optimization from scratch in PyTorch to solve OpenAI gym's LunarLander environment.

## What is Proximal Policy Optimization? 
Proximal Policy Optimization is a Policy-based method, where we essentially try to stabilize training by limiting the amount of changes we make to the policy at each epoch. The idea is that we don't want to make too big of a change to the policy at each epoch such that if we find a find somewhat of a optimal policy, we don't step over the peak. But how do we know how much our policy changes at each epoch? We calculate a ratio that measures by how much our current policy differs from the old, the ratio is simply the probability of taking the action taken at the current timestep given the current state divided by the probability of taking the same action at the state in the old policy. More succinctly, the ratio function is $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t) }$ where $\pi_\theta$ is policy $\pi$ with paramters $\theta$. We then clip the ratios that are less than $1-\epsilon$ and greater than $1+\epsilon$ to avoid changing the policy too much from the old policy while also avoiding not changing it enough. $\epsilon$ is a hyperparameter that we control. In the original research paper as well as this notebook, $\epsilon$ is set to 0.2.

## Pre-requisites 
```
pip install torch 
```
```
pip install gym 
```
```
pip install tqdm
```

## About LunarLander
In the LunarLander environment, we control a lander that is entering the atmosphere of the moon. It starts at a random location in the horizon, our goal is to safely land the lander to a designated landing pad. The lander has three rockets, one to the left of the lander, one to the right, and one in the middle. The lander has infinite rocket fuel. The observation space of the environment is a 8-dimensional vector which has the x,y coordinates of the lander, the linear velocities of the lander in the x and y directions, the angle of the lander with respect to the ground, the lander's angular velocity and boolen values to represent whether the ladner's left and right legs are touching the ground. The lander can take four discrete actions: do nothing, fire the left rocket, fire the middle rocket, fire the right rocket. If the lander, crashes it receives a reward of -100, however, if it safely comes to rest, it receives a reward of +100 and a reward of +10 for each leg that is in contact with the ground. There is a minimal cost for firing each of the rockets, -0.3 for the main engine and -0.03 for firing the side engines. The episode terminates if the lander crashes, or the lander goes out of frame or if the lander is still, ie lands.  
