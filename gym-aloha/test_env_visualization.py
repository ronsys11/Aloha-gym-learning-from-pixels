import gymnasium as gym
import gym_aloha
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory if it doesn't exist
os.makedirs("visualization_output", exist_ok=True)

# Create the environment with top_only observation type
env = gym.make(
    "gym_aloha/AlohaEndEffectorTransferCube-v0",
    obs_type="top_only",
    observation_width=240,
    observation_height=140,
)

# Reset the environment and get initial observation
observation, info = env.reset()

# Get the observation (should be only top view)
pixels = observation['pixels']

# Convert from CHW to HWC format for displaying
image = pixels.transpose(1, 2, 0)

# Save initial observation
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.title('Top View Observation')
plt.axis('off')
plt.savefig('visualization_output/initial_observation.png')
plt.close()

# Take a few random actions and save visualizations
for i in range(5):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    
    image = obs['pixels'].transpose(1, 2, 0)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(f'Step {i + 1}, Reward: {reward}')
    plt.axis('off')
    plt.savefig(f'visualization_output/step_{i+1}.png')
    plt.close()
    
    if done:
        break

env.close()
print("Visualizations saved in 'visualization_output' directory") 