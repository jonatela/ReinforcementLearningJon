import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
import numpy as np

class SafeRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        shaping = 0
        
        # SUAVE penalización por SOLO rotar (NO -0.3 brutal)
        if action in [0, 1, 2] and self.step_count % 5 == 0:  
            shaping -= 0.02  # MUY suave
            
        # BONUS moderado por TOGGLE
        if action == 5:
            shaping += 0.3  # Reducido de 2.0
        
        # SUAVÍSIMA penalización tiempo
        shaping -= 0.001
        
        # CLIP para evitar explosiones
        shaping = np.clip(shaping, -0.1, 0.5)
        
        total_reward = reward + shaping
        info["shaping"] = shaping
        
        return obs, total_reward, terminated, truncated, info

def make_safe_env():
    env = gym.make("MiniGrid-MultiRoom-N2-S4-v0", render_mode=None)
    env = SafeRewardWrapper(env)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    return env

# DETENER EL ANTERIOR (Ctrl+C) Y EJECUTAR ESTE
print("✅ PPO SAFE - N2-S4 con shaping SUAVIZADO")
train_env = make_vec_env(make_safe_env, n_envs=8)  # 8 en vez de 16
log_dir = "safe_logs"
os.makedirs(log_dir, exist_ok=True)

model = PPO(
    "CnnPolicy", 
    train_env, 
    learning_rate=3e-4,  # Learning rate por defecto
    verbose=1, 
    tensorboard_log=log_dir
)

model.learn(total_timesteps=200_000)
model.save(os.path.join(log_dir, "ppo_safe_final"))
train_env.close()
