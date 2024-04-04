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
