import gymnasium as gym
from environment import custom_environment
env = custom_environment()
obs, info = env.reset()
print(f"OBS: {obs}")