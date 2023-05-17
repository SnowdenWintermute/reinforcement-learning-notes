import numpy as np
import sys
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.envs.registration import register

def create_bins(num_bins_per_observation=25):
    bins_car_x_position = np.linspace(-1.2, .6, num_bins_per_observation)
    bins_car_velocity = np.linspace(-.07, .07, num_bins_per_observation)
    bins = np.array([bins_car_x_position, bins_car_velocity])
    return bins

NUM_BINS = 40
BINS = create_bins(NUM_BINS)

q_table = np.load('./mountain-car-q-table.npy')
print(q_table)

def discretize_observation(observations, bins):
    binned_observations = []
    for i, observation in enumerate(observations):
        discretized_observation = np.digitize(observation, bins[i])
        binned_observations.append(discretized_observation)
    return tuple(binned_observations)

env = gym.make('MountainCar-v0', render_mode="human")

observation = env.reset()[0]

for counter in range(3000):
    discrete_state = discretize_observation(observation, BINS)  # Get discretized observation
    action = np.argmax(q_table[discrete_state])  # and chose action from the Q-Table
    observation, reward, terminated, truncated, info = env.step(action) # Finally perform the action
    if terminated or truncated:
        print(f"done")
        break
env.close()
