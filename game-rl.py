import gymnasium as gym
import matplotlib.pyplot as plt
import time

def simple_agent(observation):
    # observation
    position,velocity = observation
    print(position, velocity, "<---------------")
    # when to go right
    if -0.1 < position < 0.4:
        action = 2
    # or left...
    elif velocity < 0 and position < -0.2:
        action = 0
    # or nothing
    else:
        action = 1
    return action

env = gym.make('MountainCar-v0', render_mode = 'human')

observation = env.reset(seed=42)[0]

for step in range(600):
    env.render()
    action = simple_agent(observation)
    observation,reward,terminated,truncated,info = env.step(action=action)
    time.sleep(0.001)

env.close()

    # action
# env = gym.make('ALE/Breakout-v5', render_mode = 'human')
# env.reset()

# for step in range(2000):
#     random_action = env.action_space.sample()
    
#     observation,reward,terminated,truncated,info = env.step(action=random_action)
#     print(f"reward {reward}")
#     print(f"terminated {terminated}")
#     print(f"info {info}")
#     if terminated:
#         break

# env.close()
