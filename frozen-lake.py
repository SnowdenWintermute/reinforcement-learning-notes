import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import time
from gymnasium.envs.registration import register

try:
    register(id='FrozenLakeNotSlippery-v0',
             entry_point='gym.envs.toy_text:FrozenLakeEnv',
             kwargs={'map_name': '4x4', 'is_slippery': False},
             max_episode_steps=100,
             reward_threshold=0.78
             )
except:
    print("Already registered")

# env = gym.make('FrozenLakeNotSlippery-v0')
# env = gym.make('FrozenLake-v1', desc=None, render_mode="human", map_name="4x4", is_slippery=False)
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
env.reset()

action_size = env.action_space.n
state_size = env.observation_space.n
q_table = np.zeros([state_size, action_size])

EPOCHS=20000 # aka episodes (how many times the agent plays a game until termination)
ALPHA = 0.8 # learning rate
GAMMA = 0.9  # discount rate (exponential multiplier against future rewards)
epsilon = 1.0 # exploration vs exploitation
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001

def epsilon_greedy_action_selection(epsilon, q_table, discrete_state):
    random_number = np.random.random()
    # EXPLOITATION
    if random_number > epsilon:
        state_row = q_table[discrete_state,:]
        action = np.argmax(state_row)
    # EXPLORATION
    else:
        action = env.action_space.sample()
    return action

def compute_next_q_value(old_q_value,reward,next_optimal_q_value):

    return old_q_value + ALPHA * (reward + GAMMA * next_optimal_q_value - old_q_value)

def reduce_epsilon(epsilon, epoch):
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*epoch)

rewards = []
log_interval = 1000

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()
fig.canvas.draw()
plt.show(block=True)
epoch_plot_tracker = []
total_reward_plot_tracker = []

for episode in range(EPOCHS):
    state = env.reset()[0]
    game_complete = False
    total_rewards = 0

    while not game_complete:
        action = epsilon_greedy_action_selection(epsilon, q_table, state)    
        new_state,reward,terminated,truncated,info = env.step(action)
        # OLD (Current) Q VALUE Q(st,at)
        old_q_value =  q_table[state,action]  # get next optimal Q value Q(st+1,at+1)
        next_optimal_q_value = np.max(q_table[new_state,:])
        # Compute next Q value
        next_q = compute_next_q_value(old_q_value, reward, next_optimal_q_value)
        # Update the table
        q_table[state,action] = next_q
        # track rewards
        total_rewards = total_rewards + reward
        state = new_state
        game_complete = terminated

    episode += 1
    epsilon = reduce_epsilon(epsilon, episode)
    rewards.append(total_rewards)

    total_reward_plot_tracker.append(np.sum(rewards))
    epoch_plot_tracker.append(episode)
    ########################################
    if episode % log_interval == 0:
        ax.clear()
        ax.plot(epoch_plot_tracker, total_reward_plot_tracker)
        fig.canvas.draw()
    ########################################

env.close()
np.savetxt("q_table.csv", q_table, delimiter=",")

# q_table = np.genfromtxt("./q_table.csv", delimiter=",")
# q_table = np.reshape(q_table, (state_size, action_size))
# state = env.reset()[0]

# for steps in range(100):
#     action = np.argmax(q_table[state,:])
#     print("action",action)
#     new_state,reward,terminated,truncated,info = env.step(action)
#     time.sleep(1)
#     if terminated:
#         break;

# for step in range(5):
#     env.render()
#     action = env.action_space.sample()
#     observation,reward,terminated,truncated,info = env.step(action=action)
#     if terminated:
#         env.reset()
#     time.sleep(.5)

# env.close()
