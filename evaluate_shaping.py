import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO
import os
import numpy as np

# 1) Cargar el modelo entrenado con reward shaping
log_dir = "reward_shaping_logs"
model_path = os.path.join(log_dir, "ppo_shaping_final.zip")
model = PPO.load(model_path)

print("Cargando agente PPO con reward shaping...")
print("Evaluando en MultiRoom-N4-S5 SIN shaping (recompensas originales)")

# 2) Crear entorno de evaluación SIN reward shaping (para ver si resuelve realmente)
def make_eval_env():
    env = gym.make("MiniGrid-MultiRoom-N4-S5-v0", render_mode="human")
    env = RGBImgObsWrapper(env)  # Solo imagen
    env = ImgObsWrapper(env)
    return env

env = make_eval_env()
obs, info = env.reset(seed=42)

episodes = 10
total_rewards = []
episode_lengths = []

print("\n=== EVALUACIÓN: 10 episodios ===")
print("OBS: agente moviéndose | VERDE=meta | ROJO=pared | AZUL=puerta")

for ep in range(episodes):
    obs, info = env.reset(seed=42+ep)
    episode_reward = 0
    steps = 0
    
    while steps < 500:  # máximo 500 pasos por episodio
        # El agente usa su política aprendida
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        
        # Mostrar info del paso
        if steps % 20 == 0:
            print(f"Ep {ep+1} paso {steps}: reward={reward:.3f}, total={episode_reward:.3f}")
        
        env.render()  # muestra la ventana
        
        if terminated or truncated:
            break
    
    total_rewards.append(episode_reward)
    episode_lengths.append(steps)
    print(f"EPISODIO {ep+1}: reward={episode_reward:.3f}, steps={steps}")

env.close()

# 3) Estadísticas finales
print("\n=== RESULTADOS FINALES ===")
print(f"Reward media: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
print(f"Longitud media: {np.mean(episode_lengths):.3f} ± {np.std(episode_lengths):.3f}")
print(f"Mejor episodio: {np.max(total_rewards):.3f} reward")

if np.max(total_rewards) > 0:
    print("✅ ¡ÉXITO! El agente resuelve algunos episodios")
else:
    print("❌ No resuelve episodios (reward=0 significa no llega a la meta)")
