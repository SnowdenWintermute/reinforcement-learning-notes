import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import time
from gymnasium.envs.registration import register


env = gym.make('CartPole-v1', render_mode="human")
env.reset()

for step in range(100):
    action = env.action_space.sample()
    observation,reward,terminated,truncated,info = env.step(action)
    time.sleep(0.02)

env.close()
