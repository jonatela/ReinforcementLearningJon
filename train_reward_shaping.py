import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os
import numpy as np

class RewardShapingWrapper(gym.Wrapper):
    """Wrapper que da recompensas intermedias por acercarse a la meta"""
    
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Acceder al grid del entorno MiniGrid
        grid = self.env.unwrapped.grid
        
        # Posición del agente
        agent_pos = self.env.unwrapped.agent_pos
        
        # Buscar la posición de la meta (Goal, código 4 en MiniGrid)
        goal_pos = None
        for i in range(grid.width):
            for j in range(grid.height):
                if grid.get(i, j) and grid.get(i, j).type == 'Goal':
                    goal_pos = (i, j)
                    break
            if goal_pos:
                break
        
        if goal_pos:
            # Distancia Manhattan a la meta
            dist_to_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
            reward_proximity = -0.05 * dist_to_goal  # penalización por distancia
            # Penalización por tiempo (explora más rápido)
            reward_time = -0.001  # penalización pequeña por cada paso
            reward_shaped = reward + reward_proximity + reward_time
        else:
            reward_proximity = 0
            dist_to_goal = 0
        
        # Recompensa total: original + shaping
        reward_shaped = reward + reward_proximity
        reward_shaped = max(-1.0, min(1.0, reward_shaped))  # clip
        
        info["reward_original"] = reward
        info["reward_shaped"] = reward_shaped
        info["dist_to_goal"] = dist_to_goal
        
        return obs, reward_shaped, terminated, truncated, info

# 1) Función para crear entorno con reward shaping
def make_shaped_env():
    env = gym.make("MiniGrid-MultiRoom-N4-S5-v0", render_mode=None)
    env = RewardShapingWrapper(env)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    return env

# 2) Entrenar con reward shaping
print("=== PPO con Reward Shaping en MultiRoom-N4-S5 ===")
train_env = make_vec_env(make_shaped_env, n_envs=8)
log_dir = "reward_shaping_logs"
os.makedirs(log_dir, exist_ok=True)

model = PPO("CnnPolicy", train_env, verbose=1, tensorboard_log=log_dir)

eval_env = make_vec_env(make_shaped_env, n_envs=1)
eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir, log_path=log_dir, eval_freq=20_000, n_eval_episodes=10)

model.learn(total_timesteps=400_000, callback=eval_callback)
model.save(os.path.join(log_dir, "ppo_shaping_final"))

train_env.close()
eval_env.close()
