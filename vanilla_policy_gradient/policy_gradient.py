import torch
import torch.nn as nn
import torch.optim
from torch.distributions.categorical import Categorical
import gymnasium as gym

env = gym.make("Cartpole-v1", render_mode='rgb-array')

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Softmax):
    """
    Create an MLP for the policy
    """
    layers = []
    for i in range(len(sizes)):
        layer = nn.Parameter(torch.rand(sizes[i], sizes[i+1]))
        layers.append(layer)
        layers.append(activation)

    layers.pop()
    layers.append(output_activation)

    return nn.Sequential(*layers)

def weighted_cumsum(reward:torch.Tensor, gamma:torch.FloatTensor)->torch.FloatTensor:
    num_step = reward.shape
    weight = gamma ** torch.arange(0, num_step)
    value = reward * weight
    cumsum = torch.zeros(num_step)
    cur_sum = 0
    for i in range(len(value)):
        cumsum[i] = cur_sum + value[i]
        cur_sum = cumsum[i]
    return cumsum

def train(env_id="CartPole-v1", hidden_sizes=[32], lr=1e-7,
          num_epochs=500, batch_size=5000, render=False, max_episode_steps=500):
    env = gym.make(id=env_id, max_episode_steps=max_episode_steps)

    obs_dim = env.observation_space.shape[0]
    # get number of action
    n_action = env.action_space

    policy_net = mlp([obs_dim]+hidden_sizes+[n_action])
                     
    def get_policy(obs):
        logits = policy_net(obs)
        return Categorical(logits)
    
    def get_action(obs):
        return get_policy(obs).sample().item()
    
    def compute_loss(obs, act, weight):
        # Weight are computed using several options (advantage function,
        # return function, shifted return function, e.t.c). We compute
        # this loss by using Monte Carlo methods as we do not have direct
        # access to the underlying distribution
        logp = get_policy(obs).log_prob(act)
        return -(logp * weight).mean()
    
    
    def run_epoch():
        batch_obs = []
        batch_acts = []
        batch_rets = [] # weight for gradient ascent
        batch_weights = []
        batch_ep_len = []
        batch_ep_rew = []

        obs = env.reset()

        terminated, truncated = False, False
        finished_render_eps = False

        while True:
            if not finished_render_eps and render:
                env.render()
            # Collect data
            act = get_action(obs)
            obs, rew, terminated, truncated, info = env.step(act)

            batch_obs.append(obs.copy())
            batch_acts.append(act.copy())

            break
        optimizer = torch.optim.Adam(params=policy_net.parameters(),
                                     lr=lr)
        
        loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32), 
                            act=torch.as_tensor(batch_acts, dtype=torch.float32),
                            weight=torch.as_tensor(batch_weights, dtype=torch.float32))
        
        optimizer.zero_grad()
        loss.
            

            
