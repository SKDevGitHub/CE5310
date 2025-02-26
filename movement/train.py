from stable_baselines3 import PPO
from environment import custom_environment

def train():
    train_env = custom_environment()
    model = PPO("MlpPolicy", train_env, verbose = 1, tensorboard_log = "./ppo_custom_humanoid")
    model.learn(total_timesteps = 1000000)
    model.save("robot_ppo")
    train_env.close()
    
if __name__ == "__main__":
    train()