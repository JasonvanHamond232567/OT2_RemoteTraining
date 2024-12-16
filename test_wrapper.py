from stable_baselines3.common.env_checker import check_env
from ot2_gym_wrapper import OT2Env
import gymnasium as gym
import numpy as np

# Load your custom environment
env = OT2Env()

# Number of episodes
num_episodes = 1

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    step = 0

    while not done and step < 1000:
        # Take a random action from the environment's action space
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Episode: {episode + 1}, Step: {step + 1}, Action: {action}, Reward: {reward}")

        step += 1
        if done:
            print(f"Episode finished after {step} steps. Info: {info}")
            break
