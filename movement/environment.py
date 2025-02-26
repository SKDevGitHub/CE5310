import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

#CUSTOM ENV FOR ROBOT???
class custom_environment(gym.Wrapper):
    def __init__(self, env_name = "Humanoid-v4"):
        super().__init__(gym.make(env_name, render_mode = "human"))
        self.prev_torso_pos = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        unwrapped_env = self.env.unwrapped
        if hasattr(unwrapped_env, 'sim'):
            self.prev_torso_pos = unwrapped_env.sim.data.qpos[0]
        else:
            raise AttributeError("Simulation data (sim) not found.")
        print("Torso position:", self.env.sim.data.qpos[0])
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward += self.custom_reward(obs, action)
        return obs, reward, terminated, truncated, info
    
    def custom_reward(self, obs, action):
        torso_position = self.env.data.qpos[0]
        velocity_reward = (torso_position - self.prev_torso_pos) * 10
        self.prev_torso_pos = torso_position
        
        energy_penalty = -np.sum(np.abs(action)) * 0.01
        balance_penalty = -abs(obs[2])
        return velocity_reward + energy_penalty + balance_penalty