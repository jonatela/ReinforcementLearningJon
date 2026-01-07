import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
import os

# 1) Función para crear entorno con nivel de dificultad
def make_env(room_count, seed=0):
    env_id = f"MiniGrid-MultiRoom-N{room_count}-S4-v0" if room_count == 2 else f"MiniGrid-MultiRoom-N{room_count}-S5-v0"
    env = gym.make(env_id, render_mode=None)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    return env

# 2) FASE 1: Entrenar en MultiRoom-N2-S4 (fácil)
print("=== FASE 1: MultiRoom-N2-S4 ===")
train_env1 = make_vec_env(lambda: make_env(2), n_envs=8)
log_dir1 = "curriculum_logs/fase1"
os.makedirs(log_dir1, exist_ok=True)

model = PPO("CnnPolicy", train_env1, verbose=1, tensorboard_log=log_dir1)
model.learn(total_timesteps=100_000)
model.save(os.path.join(log_dir1, "ppo_fase1_final"))
train_env1.close()

# 3) FASE 2: Continuar en MultiRoom-N4-S5 (difícil)
print("\n=== FASE 2: MultiRoom-N4-S5 ===")
train_env2 = make_vec_env(lambda: make_env(4), n_envs=8)
log_dir2 = "curriculum_logs/fase2"
os.makedirs(log_dir2, exist_ok=True)

# Cargar modelo de la fase 1 y continuar
model = PPO.load(os.path.join(log_dir1, "ppo_fase1_final"), env=train_env2)
model.set_logger(configure(log_dir2, ["stdout", "tensorboard"]))

eval_env2 = make_vec_env(lambda: make_env(4), n_envs=1)
eval_callback2 = EvalCallback(eval_env2, best_model_save_path=log_dir2, log_path=log_dir2, eval_freq=20_000, n_eval_episodes=10)

model.learn(total_timesteps=300_000, callback=eval_callback2, tb_log_name="fase2")
model.save(os.path.join(log_dir2, "ppo_curriculum_final"))
train_env2.close()
eval_env2.close()
