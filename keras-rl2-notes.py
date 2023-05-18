import gymnasium as gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent

env_name = 'CartPole-v0'

env = gym.make(env_name, render_mode="human")


nb_actions = env.action_space.n
nb_obs = env.observation_space.shape
# print("nb_actions: ", nb_actions)
# print("nb_obs: ", nb_obs)
# print("shape of plussed thing: ", (1,) + nb_obs)

model = Sequential()
model.add(Flatten(input_shape = (1,) + nb_obs))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
# print(model.summary())

from rl.memory import SequentialMemory

memory = SequentialMemory(limit=20000, window_length=1)

from rl.policy import LinearAnnealedPolicy,EpsGreedyQPolicy

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                              attr="eps",
                              value_max=1.0,
                              value_min=0.1,
                              value_test=0.05,
                              nb_steps=20000)


dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               memory=memory,
               nb_steps_warmup=10,
               target_model_update=100,
               policy=policy)

dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=20000, visualize=False, verbose=2)
