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

class VPG:
    def __init__(self, sizes, activation=nn.Tanh, output_activation=nn.Softmax):
        self.policy_net = mlp(sizes, activation, output_activation)

    def get_policy(self, obs):
        logits = self.policy_net(obs)
        return Categorical(logits)

    def get_action(self, obs):
        return self.get_policy(obs).sample().item()

    def compute_loss(self, logits, weight):
        logp = torch.log(logits)
        return (logp * weight).mean()

def compute_loss():
    pass
def run_train_one_epoch(step_size=1e-7, epochs=500, ):

def train():
    global env

    obs_dim = env.observation_space.shape[0]
    num_action = env.action_space
    env.reset()