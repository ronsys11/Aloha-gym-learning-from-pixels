import torch
import numpy as np
import gymnasium as gym
import gym_aloha
import os
import cv2
from tqdm import tqdm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 512):
        if isinstance(observation_space, gym.spaces.Dict):
            super().__init__(observation_space, features_dim)
            pixels_space = observation_space["pixels"]
            n_input_channels = pixels_space.shape[0]
        else:
            super().__init__(observation_space, features_dim)
            n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            if isinstance(observation_space, gym.spaces.Dict):
                sample = pixels_space.sample()[None]
            else:
                sample = observation_space.sample()[None]
            sample_tensor = torch.as_tensor(sample).float()
            n_flatten = self.cnn(sample_tensor).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if isinstance(observations, dict):
            return self.linear(self.cnn(observations["pixels"]))
        return self.linear(self.cnn(observations))

class BCPolicy(nn.Module):
    def __init__(self, features_extractor, action_space):  # Change parameter to action_space
        super().__init__()
        self.features_extractor = features_extractor
        self.action_dim = action_space.shape[0]  # Get dimension from action_space
        
        # Add joint ranges to match training model
        self.joint_ranges = [
            [-3.14158, 3.14158],  # waist
            [-1.85005, 1.25664],  # shoulder
            [-1.76278, 1.6057],   # elbow
            [-3.14158, 3.14158],  # forearm_roll
            [-1.8675, 2.23402],   # wrist_angle
            [-3.14158, 3.14158],  # wrist_rotate
            [0.0, 1.0],           # left gripper
            [-3.14158, 3.14158],  # joint 7
            [-3.14158, 3.14158],  # waist (right)
            [-1.85005, 1.25664],  # shoulder (right)
            [-1.76278, 1.6057],   # elbow (right)
            [-3.14158, 3.14158],  # forearm_roll (right)
            [-1.8675, 2.23402],   # wrist_angle (right)
            [0.0, 1.0],           # right gripper
        ]
        
        self.register_buffer('joint_mins', torch.tensor([x[0] for x in self.joint_ranges]))
        self.register_buffer('joint_maxs', torch.tensor([x[1] for x in self.joint_ranges]))
        
        self.policy_net = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
            nn.Tanh()
        )
    def forward(self, obs):
        features = self.features_extractor(obs)
        normalized_actions = self.policy_net(features)
        # Scale from [-1,1] to actual joint ranges
        actions = (normalized_actions + 1.0) * (self.joint_maxs - self.joint_mins) / 2.0 + self.joint_mins
        return actions
    def get_normalized_actions(self, obs):
        """Helper method to get the normalized actions before scaling"""
        features = self.features_extractor(obs)
        return self.policy_net(features)


def make_env():
    return gym.make(
        "gym_aloha/AlohaTransferCube-v0",
        obs_type="top_only",
        observation_width=240,
        observation_height=140,
        render_mode="rgb_array"
    )

def evaluate_policy(policy, env, num_episodes=10, video_dir="bc_evaluation_videos"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = policy.to(device)
    policy.eval()
    
    # Create video directory
    os.makedirs(video_dir, exist_ok=True)
    
    episode_rewards = []
    episode_lengths = []
    success_rate = 0
    
    print(f"\nEvaluating policy for {num_episodes} episodes...")
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        frames = []
        
        # Setup video writer
        video_path = os.path.join(video_dir, f"episode_{episode}.mp4")
        frame = env.render()
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
        frames.append(frame)
        
        print(f"\nEpisode {episode+1}/{num_episodes}")
        
        while not (done or truncated):
            # Process observation
            processed_obs = {k: torch.as_tensor(v / 255.0, device=device).float().unsqueeze(0) for k, v in obs.items()}
            
            # Get action from policy
            with torch.no_grad():
                action = policy(processed_obs).cpu().numpy().squeeze()
            
            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)
            print(f"Action: {action}")
            
            # Render and save frame
            frame = env.render()
            frames.append(frame)
            
            episode_reward += reward
            step_count += 1
            
            if step_count % 10 == 0:
                print(f"  Step: {step_count}, Reward: {episode_reward:.4f}")
        
        # Check if episode was successful
        if 'success' in info:
            success_rate += int(info['success'])
        
        # Write frames to video
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        print(f"Episode {episode+1} complete:")
        print(f"  Total steps: {step_count}")
        print(f"  Total reward: {episode_reward:.4f}")
        if 'success' in info:
            print(f"  Success: {info['success']}")
    
    # Print summary statistics
    success_rate = (success_rate / num_episodes) * 100
    print("\nEvaluation Summary:")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Mean episode reward: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    return episode_rewards, episode_lengths, success_rate

def main():
    # Create environment
    env = make_env()
    
    # Create policy with same architecture as training
    features_extractor = CustomCNN(env.observation_space, features_dim=512)
    policy = BCPolicy(features_extractor, env.action_space)
    
    # Load the trained weights
    print("Loading BC policy...")
    checkpoint = torch.load("bc_models/best_bc_policy3.pth")
    
    # Handle both formats: direct state dict or nested in 'state_dict' key
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint  # Assume it's the state dict directly
        
    policy.load_state_dict(state_dict)
    print("BC policy loaded successfully")
    
    # Rest of the code...
    
    # Evaluate policy
    rewards, lengths, success_rate = evaluate_policy(
        policy, 
        env, 
        num_episodes=10,
        video_dir="bc_evaluation_videos"
    )
    
    # Save evaluation metrics
    os.makedirs("evaluation_metrics", exist_ok=True)
    np.save("evaluation_metrics/bc_eval_rewards.npy", rewards)
    np.save("evaluation_metrics/bc_eval_lengths.npy", lengths)
    
    # Close environment
    env.close()
    
    print("\nEvaluation complete! Videos saved to bc_evaluation_videos/")

if __name__ == "__main__":
    main()