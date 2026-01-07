import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO
import os

# 1) Cargar el modelo entrenado
log_dir = "ppo_multiroom_logs"
model_path = os.path.join(log_dir, "ppo_multiroom_final.zip")
model = PPO.load(model_path)

# 2) Crear el entorno con renderizado
def make_env():
    env = gym.make("MiniGrid-MultiRoom-N2-S4-v0", render_mode="human")
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    return env

env = make_env()

# 3) Probar el agente
obs, info = env.reset()
for step in range(200):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

env.close()
