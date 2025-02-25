import time
from stable_baselines3 import PPO
from environment import custom_environment

def test():
    model = PPO.load("robot_ppo")
    test_env = custom_environment()
    obs, info = test_env.reset()
    for _ in range(10000):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = test_env.step(action)
        test_env.render()
        time.sleep(1/60)
        if terminated or truncated:
            obs, info = test_env.reset()
    test_env.close()
    
if __name__ == "__main__":
    test()