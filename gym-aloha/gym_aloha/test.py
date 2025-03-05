# example.py
import imageio
import gymnasium as gym
import numpy as np
import gym_aloha

env = gym.make("gym_aloha/AlohaInsertion-v0")
observation, info = env.reset()

# Create separate lists for each camera view
top_frames = []
angle_frames = []
vis_frames = []

for _ in range(100):  # Reduced to 100 frames for quicker testing
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    images = env.render()
    
    # Store each camera view separately
    top_frames.append(images["top"])
    angle_frames.append(images["angle"])
    vis_frames.append(images["vis"])
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()

# Save separate videos for each camera angle
imageio.mimsave("top_view.mp4", np.stack(top_frames), fps=25)
imageio.mimsave("angle_view.mp4", np.stack(angle_frames), fps=25)
imageio.mimsave("vis_view.mp4", np.stack(vis_frames), fps=25)

print("Videos saved successfully. Check all three video files to verify the camera angles.")