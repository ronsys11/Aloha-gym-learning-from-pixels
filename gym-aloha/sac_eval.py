from stable_baselines3 import SAC
import gymnasium as gym
import gym_aloha
from collections import Counter
import numpy as np
import cv2
import os
from gymnasium.wrappers import RecordVideo

# Create output directories
os.makedirs("evaluation_videos", exist_ok=True)
os.makedirs("evaluation_images", exist_ok=True)

# Load the trained model
print("Loading SAC model...")
model = SAC.load("hybrid_bc_sac_continued_10")

# Create evaluation environment with EXACT same parameters as training
env = gym.make(
    "gym_aloha/AlohaTransferCube-v0",
    obs_type="top_only",
    observation_width=240,
    observation_height=140,
    render_mode="rgb_array"
)

# Print spaces to debug
print("Environment observation space:", env.observation_space)
print("Environment action space:", env.action_space)

# Add action smoothing to match training environment
original_step = env.step
def smoothed_step(action):
    # Apply action smoothing
    if not hasattr(env, '_last_action'):
        env._last_action = action.copy()
    else:
        # Use same smoothing as in training
        action = 0.8 * env._last_action + 0.2 * action
    env._last_action = action.copy()
    return original_step(action)

env.step = smoothed_step

# Wrap environment for video recording
env = RecordVideo(
    env,
    video_folder="evaluation_videos/",
    episode_trigger=lambda x: True  # Record every episode
)

# Evaluate the agent
print("Starting evaluation...")
num_episodes = 10  # Reduced for testing
episode_rewards = []
total_steps = 0

for episode in range(num_episodes):
    obs, _ = env.reset()
    episode_reward = 0
    done = False
    step_count = 0
    max_steps = 300  # Match training episode length
    episode_frames = []

    print(f"\nStarting episode {episode + 1}")
    
    while not done and step_count < max_steps:
        # Try both deterministic and stochastic actions
        action, _ = model.predict(obs, deterministic=False)  # Use stochastic actions
        
        # Debug action values
        if step_count == 0 or step_count % 50 == 0:
            print(f"Step {step_count}, Action: min={action.min():.3f}, max={action.max():.3f}, mean={action.mean():.3f}")
            
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        
        # Store frames for visualization
        if isinstance(obs, dict) and "pixels" in obs:
            frame = obs["pixels"].transpose(1, 2, 0)  # CHW to HWC
            episode_frames.append(frame)
            
        if reward > 0:
            print(f"  Step {step_count}: Got reward {reward}")
            
        step_count += 1
        total_steps += 1
        
        if done or truncated:
            break

    episode_rewards.append(episode_reward)
    print(f"Episode {episode + 1} finished with total reward: {episode_reward} in {step_count} steps")
    
    # Save frames as video
    if episode_frames:
        video_path = f"evaluation_images/episode_{episode}_frames.mp4"
        height, width = episode_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
        
        for frame in episode_frames:
            # Convert from RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()

# Close the environment
env.close()

# Print evaluation summary
print("\nEvaluation Summary:")
print(f"Total Episodes: {num_episodes}")
print(f"Average Reward per Episode: {sum(episode_rewards) / num_episodes:.2f}")
print(f"Episode Rewards: {episode_rewards}")
print(f"Total Steps: {total_steps}")
print("\nVideos saved in 'evaluation_videos' directory")
print("Episode frames saved in 'evaluation_images' directory")
