import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper

env = gym.make("MiniGrid-MultiRoom-N2-S4-v0", render_mode="human")
env = RGBImgObsWrapper(env)
env = ImgObsWrapper(env)

obs, info = env.reset(seed=0)

for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
