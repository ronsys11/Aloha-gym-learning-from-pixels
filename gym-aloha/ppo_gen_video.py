from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import gym_aloha
from stable_baselines3 import PPO
# Load trained model
model = PPO.load("ppo_transfer_cube2")
env = gym.make("gym_aloha/AlohaTransferCube-v0", render_mode="rgb_array")
env = RecordVideo(env, video_folder="./videos2/", episode_trigger=lambda x: True)

obs, _ = env.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        obs, _ = env.reset()

env.close()
print("🎥 Video saved in './videos2/' folder.")
