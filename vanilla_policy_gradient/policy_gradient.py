import torch
import torch.nn as nn
import torch.optim
from torch.distributions.categorical import Categorical
import gymnasium as gym
from tqdm import tqdm

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    """
    Create an MLP for the policy
    """
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i < len(sizes) - 2 else output_activation
        layer = nn.Linear(sizes[i], sizes[i+1])
        layers += [layer, act()]
    return nn.Sequential(*layers)

def train(env_id="CartPole-v1", hidden_sizes=[32], lr=1e-7,
          num_epochs=50, batch_size=5000, render=False):
    env = gym.make(id=env_id, render_mode="human")

    obs_dim = env.observation_space.shape[0]
    # get number of action
    n_action = env.action_space.n

    policy_net = mlp([obs_dim]+hidden_sizes+[n_action])
                     
    def get_policy(obs):
        logits = policy_net(obs)
        return Categorical(logits=logits)
    
    def get_action(obs):
        return get_policy(obs).sample().item()
    
    def compute_loss(obs, act, weight):
        # Weight are computed using several options (advantage function,
        # return function, shifted return function, e.t.c). We compute
        # this loss by using Monte Carlo methods as we do not have direct
        # access to the underlying distribution
        logp = get_policy(obs).log_prob(act)
        return -(logp * weight).mean()
    
    optimizer = torch.optim.Adam(params=policy_net.parameters(),
                                 lr=lr)
    def run_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = [] # weight for gradient ascent, R(\tau)
        batch_ep_len = []
        batch_ep_rew = []

        obs, info = env.reset()
        terminated, truncated = False, False
        finished_render_eps = False
        ep_rew = []
        cur_sample_size = 0
        while True:
            if not finished_render_eps and render:
                env.render()
            # Collect data
            obs = torch.as_tensor(obs, dtype=torch.float32)
            act = get_action(obs)
            batch_obs.append(obs)
            obs, rew, terminated, truncated, info = env.step(act)

            ep_rew.append(rew)
            batch_acts.append(act)

            if (terminated or truncated):
                ep_ret, ep_len = sum(ep_rew), len(ep_rew)
                batch_ep_len.append(ep_len)
                batch_ep_rew.append(ep_ret)
                batch_weights += [ep_ret] * ep_len
                cur_sample_size += ep_len

                obs, info = env.reset()
                ep_rew, terminated, truncated = [], False, False
                finished_render_eps = True
                if cur_sample_size > batch_size:
                    break

        batch_obs = torch.stack(batch_obs)
        batch_acts = torch.as_tensor(batch_acts, dtype=torch.float32)
        batch_weights = torch.as_tensor(batch_weights, dtype=torch.float32)

        loss = compute_loss(obs=batch_obs, 
                            act=batch_acts,
                            weight=batch_weights)
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        return loss, batch_ep_rew, batch_ep_len
    
            
    for epoch in tqdm(range(num_epochs)):
        loss, batch_ep_rew, batch_ep_len = run_epoch()
        print(f"Loss: {loss} | Reward: {sum(batch_ep_rew)} | Sampling Len: {sum(batch_ep_len)}")

if __name__=="__main__":
    print("Start training")
    train(render=True)