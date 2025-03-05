import torch
import numpy as np
import h5py
import os
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import gym_aloha
import cv2
# Add model classes
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
    def __init__(self, features_extractor, action_space):
        super().__init__()
        self.features_extractor = features_extractor
        # Fix: Get action dimension from action_space
        self.action_dim = action_space.shape[0]  # For Box space
        
        # Rest of the initialization remains the same...
        
        # Updated joint ranges to match demo data
        self.joint_ranges = [
            [-3.14158, 3.14158],  # waist
            [-1.85005, 1.25664],  # shoulder
            [-1.76278, 1.6057],   # elbow
            [-3.14158, 3.14158],  # forearm_roll
            [-1.8675, 2.23402],   # wrist_angle
            [-3.14158, 3.14158],  # wrist_rotate
            [0.0, 1.0],           # left gripper (not finger)
            [-3.14158, 3.14158],  # joint 7
            [-3.14158, 3.14158],  # waist (right)
            [-1.85005, 1.25664],  # shoulder (right)
            [-1.76278, 1.6057],   # elbow (right)
            [-3.14158, 3.14158],  # forearm_roll (right)
            [-1.8675, 2.23402],   # wrist_angle (right)
            [0.0, 1.0],           # right gripper (not finger)
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

# Rest of your debug code...
def load_sample_demo():
    demo_dir = "/home/ronak/Projects/gym-aloha/gym-aloha/data/sim_transfer_cube_scripted/"
    episode_file = os.path.join(demo_dir, "episode_0.hdf5")
    
    with h5py.File(episode_file, 'r') as f:
        obs = np.array(f['observations/images/top'])[0]  # Get first frame
        # Transpose from [H, W, C] to [C, H, W] format
        obs = np.transpose(obs, (2, 0, 1))
        # Resize to match environment dimensions (240x140)
        obs = cv2.resize(obs.transpose(1, 2, 0), (240, 140)).transpose(2, 0, 1)
        action = np.array(f['action'])[0]  # Get first action
    return obs, action

def debug_model_output():
    env = gym.make(
        "gym_aloha/AlohaTransferCube-v0",
        obs_type="top_only",
        observation_width=240,
        observation_height=140,
        render_mode="rgb_array"
    )
    
    # Log action space bounds
    print("Environment Action Space:")
    print(f"Shape: {env.action_space.shape}")
    print(f"Bounds: [{env.action_space.low}, {env.action_space.high}]")
    
    # Load and process demo
    demo_obs, demo_action = load_sample_demo()
    processed_obs = demo_obs.astype(np.float32) / 255.0
    obs_tensor = torch.tensor(processed_obs).float().unsqueeze(0)
    
    # Create and load model
    features_extractor = CustomCNN(env.observation_space, features_dim=512)
    policy = BCPolicy(features_extractor, env.action_space)
    policy.load_state_dict(torch.load("bc_models/best_bc_policy2.pth", weights_only=True))
    policy.eval()
    
    with torch.no_grad():
        # Get both normalized and final actions
        normalized_actions = policy.get_normalized_actions(obs_tensor)
        predicted_actions = policy(obs_tensor)
        
        # Convert to numpy for printing
        normalized_np = normalized_actions.cpu().numpy()[0]
        predicted_np = predicted_actions.cpu().numpy()[0]
        
        print("\nAction Analysis:")
        print(f"Demo Action (Original):", demo_action)
        print(f"Demo Action Range: [{demo_action.min()}, {demo_action.max()}]")
        print(f"\nModel Output (Normalized [-1,1]):", normalized_np)
        print(f"Normalized Range: [{normalized_np.min()}, {normalized_np.max()}]")
        print(f"\nModel Output (Scaled):", predicted_np)
        print(f"Scaled Range: [{predicted_np.min()}, {predicted_np.max()}]")
        print(f"\nAction Difference (MAE):", np.abs(demo_action - predicted_np).mean())
        
        # Print per-joint comparison
        print("\nPer-Joint Analysis:")
        for i in range(len(demo_action)):
            print(f"Joint {i}: Demo={demo_action[i]:.4f}, Predicted={predicted_np[i]:.4f}, "
                  f"Diff={abs(demo_action[i] - predicted_np[i]):.4f}") 
if __name__ == "__main__":
    debug_model_output()