#THIS FILE IS TO TEST YOUR ENVIRONMENT IN A BASIC MUJOCO ENVIRONMENT TO ENSURE THE CUSTOM ENVIRONMENT WORKS CORRECTLY

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from environment import custom_environment

# Create Humanoid-v4 environment
env = gym.make("Humanoid-v4", render_mode="human")
#env = custom_environment()

# Reset environment
obs, info = env.reset()

# Run a random action loop
for _ in range(1000):
    action = env.action_space.sample()  # Sample random action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
