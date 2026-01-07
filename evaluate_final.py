import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO
import os
import numpy as np

# CAMBIAR A TUS LOGS ACTUALES
log_dir = "safe_logs"  # ← safe_logs del nuevo agente exitoso
model_path = os.path.join(log_dir, "ppo_safe_final.zip")
model = PPO.load(model_path)

print("🎉 EVALUACIÓN AGENTE EXITOSO (Safe Shaping)")
print("Vamos a ver si AHORA SÍ abre puertas...")

def make_eval_env():
    env = gym.make("MiniGrid-MultiRoom-N2-S4-v0", render_mode="human")
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    return env

env = make_eval_env()
obs, info = env.reset(seed=42)

# Contadores completos
action_counts = {}
toggle_attempts = 0
total_steps = 0
episode_rewards = []

print("\nMiniGrid ACCIONES:")
print("0=left, 1=right, 2=back, 3=forward, 4=forward, 5=TOGGLE (ABRIR PUERTA)")
print("-" * 60)

for episode in range(5):  # 5 episodios
    print(f"\n🔄 EPISODIO {episode+1}")
    obs, info = env.reset(seed=42+episode)
    episode_reward = 0
    steps = 0
    
    while steps < 200:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        
        if action not in action_counts:
            action_counts[action] = 0
        action_counts[action] += 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        total_steps += 1
        
        env.render()
        
        # Diagnóstico
        if steps % 25 == 0:
            print(f"  Paso {steps:3d} | Acción: {action} | Reward: {reward:.3f}")
        
        if action == 5:
            toggle_attempts 
