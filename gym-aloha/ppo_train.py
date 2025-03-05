from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import gym_aloha

# Create a vectorized environment
env = make_vec_env(lambda: gym.make("gym_aloha/AlohaTransferCube-v0"), n_envs=1)

# Use MultiInputPolicy for dictionary observations with SAC
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")

# Train the model
model.learn(total_timesteps=200000)

# Save the trained model
model.save("ppo_transfer_cube2")
