import numpy as np
import sys
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.envs.registration import register

env = gym.make('MountainCar-v0')

def create_bins(num_bins_per_observation=25):
    bins_car_x_position = np.linspace(-1.2, .6, num_bins_per_observation)
    bins_car_velocity = np.linspace(-.07, .07, num_bins_per_observation)
    bins = np.array([bins_car_x_position, bins_car_velocity])
    return bins

NUM_BINS = 40
BINS = create_bins(NUM_BINS)

def discretize_observation(observations, bins):
    binned_observations = []
    for i, observation in enumerate(observations):
        discretized_observation = np.digitize(observation, bins[i])
        binned_observations.append(discretized_observation)
    return tuple(binned_observations)

q_table_shape = (NUM_BINS, NUM_BINS, env.action_space.n)
q_table = np.zeros(q_table_shape)

def epsilon_greedy_action_selection(epsilon, q_table, discrete_state):
    if np.random.random() > epsilon:
        action = np.argmax(q_table[discrete_state])
    else:
        action = np.random.randint(0, env.action_space.n)
    return action

def compute_next_q_value(old_q_value, reward, next_optimal_q_value):
    return old_q_value +  ALPHA * (reward + GAMMA * next_optimal_q_value - old_q_value)

def reduce_epsilon(epsilon, epoch):
    if BURN_IN <= epoch <= EPSILON_END:
        epsilon -= EPSILON_REDUCE
    return epsilon

EPOCHS = 30000
BURN_IN = 100
epsilon = 1

EPSILON_END= 10000
EPSILON_REDUCE = 0.0001 

ALPHA = 0.8
GAMMA = 0.9

log_interval = 100

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()
fig.canvas.draw()


max_position_log = []  # to store all achieved points
mean_positions_log = []  # to store a running mean of the last 30 results
epochs = []  # store the epoch for plotting

for epoch in range(EPOCHS):
    initial_state = env.reset()[0]
    discretized_state = discretize_observation(initial_state, BINS)
    terminated = False
    truncated = False
    max_position = -np.inf  
    epochs.append(epoch)

    while not terminated and not truncated:
        action = epsilon_greedy_action_selection(epsilon, q_table, discretized_state)
        next_state,reward,terminated,truncated,info = env.step(action)
        position,velocity = next_state
        next_state_discretized = discretize_observation(next_state, BINS)
        old_q_value = q_table[discretized_state + (action,)]
        next_optimal_q_value = np.max(q_table[next_state_discretized])
        next_q = compute_next_q_value(old_q_value, reward, next_optimal_q_value)
        q_table[discretized_state + (action,)] = next_q
        discretized_state = next_state_discretized
        if position > max_position:  
            max_position = position 

    epsilon = reduce_epsilon(epsilon, epoch)

    max_position_log.append(max_position)  # log the highest position the car was able to reach
    running_mean = round(np.mean(max_position_log[-30:]), 2)  # Compute running mean of position over the last 30 epochs
    mean_positions_log.append(running_mean)

    if epoch % log_interval == 0:
        plt.pause(0.0001)
        ax.clear()
        ax.scatter(epochs, max_position_log)
        ax.plot(epochs, max_position_log)
        ax.plot(epochs, mean_positions_log, label=f"Running Mean: {running_mean}")
        plt.legend()
        fig.canvas.draw()

env.close()






np.set_printoptions(threshold=sys.maxsize)
print(q_table)

np.save("mountain-car-q-table", q_table)
