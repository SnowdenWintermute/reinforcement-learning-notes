import numpy as np
import random
from collections import deque
import gymnasium as gym

from tensorflow.keras.models import Sequential,clone_model
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import disable_interactive_logging

disable_interactive_logging()

env_name = 'CartPole-v1'
# env = gym.make(env_name, render_mode="human")
env = gym.make(env_name)

env.reset()

# for step in range(1000):
#     random_action = env.action_space.sample()
#     env.step(random_action)

# env.close()

# 4 observations for the cart pole environment
num_observations = env.observation_space.shape[0]
num_actions = env.action_space.n

# ANN (input_shape = number of observations or something that matches up to 4)
model = Sequential()
model.add(Dense(16, input_shape = (1,num_observations))) # input shape parameter lets you match up input layer neurons if you have more than the number of observations
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))

# Neurons == action_space
model.add(Dense(num_actions))
model.add(Activation('linear'))

target_model = clone_model(model)

EPOCHS = 1000
epsilon = 1.0
EPSILON_REDUCE = 0.995
BATCH_SIZE = 32
LEARNING_RATE = 0.001
GAMMA = 0.95

def epsilon_greedy_action_selection(model, epsilon, observation):
    if np.random.random() > epsilon:
        prediction = model.predict(observation.reshape(1,1,4))
        action = np.argmax(prediction)
    else:
        action = np.random.randint(0,env.action_space.n)
    return action

replay_buffer = deque(maxlen=20000)
update_target_model = 10

def replay(replay_buffer, batch_size, model, target_model):
    if len(replay_buffer) < batch_size:
        return
    samples = random.sample(replay_buffer, batch_size)
    target_batch = []
    zipped_samples = list(zip(*samples))
    states, actions, rewards, new_states, dones = zipped_samples
    targets = target_model.predict(np.array(states))
    q_values = model.predict(np.array(new_states))

    for i in range(batch_size):
        q_value = max(q_values[i][0])
        target = targets[i].copy()

        if dones[i]:
            target[0][actions[i]] = rewards[i]
        else:
            target[0][actions[i]] = rewards[i] + q_value * GAMMA

        target_batch.append(target)

    model.fit(np.array(states), np.array(target_batch), epochs=1, verbose=0)

def update_model_handler(epoch, update_target_model, model, target_model):
    if epoch > 0 and epoch % update_target_model == 0:
        target_model.set_weights(model.get_weights())

model.compile(loss='mse', optimizer=(Adam(lr=LEARNING_RATE)))

model.summary()

best_so_far = 0

for epoch in range(EPOCHS):
    observation = env.reset()[0]
    # keras expects the observation in this shape:
    # (1,X)
    # it comes like this: [a,b,c,d] so we must .reshape --> (1,4) one by num_observations
    observation = observation.reshape([1,4])
    print("epoch: ", epoch)
    terminated = False
    truncated = False

    points = 0

    while not terminated and not truncated:
        action = epsilon_greedy_action_selection(model, epsilon, observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_observation = next_observation.reshape([1,4])

        replay_buffer.append((observation, action, reward, next_observation, terminated or truncated))
        observation = next_observation
        points += 1
        replay(replay_buffer, BATCH_SIZE, model, target_model)

    epsilon *= EPSILON_REDUCE

    update_model_handler(epoch, update_target_model, model, target_model)

    if points > best_so_far:
        best_so_far = points
    if epoch % 25 == 0:
        print(f"{epoch}: POINTS: {points} eps: {epsilon} BSF: {best_so_far}")

