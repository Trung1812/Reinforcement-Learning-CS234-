import gymnasium as gym

env = gym.make("CartPole-v1")
print(env)
print(env.action_space.n)