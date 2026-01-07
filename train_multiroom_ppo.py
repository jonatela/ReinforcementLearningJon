import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os

# 1) Crear el entorno base
def make_env():
    env = gym.make("MiniGrid-MultiRoom-N2-S4-v0", render_mode=None)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    return env

# 2) Vectorizar el entorno (varias copias en paralelo)
train_env = make_vec_env(make_env, n_envs=8)

# 3) Carpeta para guardar modelos y logs
log_dir = "ppo_multiroom_logs"
os.makedirs(log_dir, exist_ok=True)

# 4) Entorno de evaluación (una sola copia)
eval_env = make_vec_env(make_env, n_envs=1)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
)

# 5) Definir el modelo PPO
model = PPO(
    "CnnPolicy",          # política con CNN para entradas tipo imagen
    train_env,
    verbose=1,
    tensorboard_log=log_dir,
)

# 6) Entrenar el agente
total_timesteps = 200_000
model.learn(
    total_timesteps=total_timesteps,
    callback=eval_callback,
)

# 7) Guardar el modelo final
model.save(os.path.join(log_dir, "ppo_multiroom_final"))
train_env.close()
eval_env.close()
