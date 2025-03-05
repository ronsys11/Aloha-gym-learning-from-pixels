import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import cv2
import gymnasium as gym
import gym_aloha
import os
from tqdm import tqdm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Reuse the CustomCNN class
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 512):
        # For Dict observation spaces, we need to extract the pixels part
        if isinstance(observation_space, gym.spaces.Dict):
            super().__init__(observation_space, features_dim)
            pixels_space = observation_space["pixels"]
            n_input_channels = pixels_space.shape[0]
        else:
            super().__init__(observation_space, features_dim)
            n_input_channels = observation_space.shape[0]
        
        # Create CNN with appropriate architecture for 240x140 images
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            if isinstance(observation_space, gym.spaces.Dict):
                # Use a sample from the pixels space
                sample = pixels_space.sample()[None]
            else:
                sample = observation_space.sample()[None]
                
            sample_tensor = torch.as_tensor(sample).float()
            n_flatten = self.cnn(sample_tensor).shape[1]
            print(f"CNN output flattened dimension: {n_flatten}")
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # If observations is a dict, extract just the pixels
        if isinstance(observations, dict):
            return self.linear(self.cnn(observations["pixels"]))
        return self.linear(self.cnn(observations))
    
class DemonstrationDataset(Dataset):
    def __init__(self, demonstrations):
        self.observations = []
        self.actions = []
        
        # Updated joint ranges based on actual demo data
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
        
        self.joint_mins = np.array([x[0] for x in self.joint_ranges])
        self.joint_maxs = np.array([x[1] for x in self.joint_ranges])
        
        # Process demonstrations
        for obs_demo, actions_demo in demonstrations:
            for t in range(len(actions_demo)):
                pixels = obs_demo['pixels'][t].astype(np.float32) / 255.0
                actions = actions_demo[t]
                
                # Normalize actions to [-1,1] range using joint ranges
                normalized_actions = 2.0 * (actions - self.joint_mins) / (self.joint_maxs - self.joint_mins) - 1.0
                
                self.observations.append({'pixels': pixels})
                self.actions.append(normalized_actions)
        
        print(f"Created dataset with {len(self.observations)} samples")
        print(f"Action range after normalization: [{np.min(self.actions):.4f}, {np.max(self.actions):.4f}]")
    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]

def load_demonstrations():
    """Load demonstrations with detailed statistics"""
    demos = []
    total_frames = 0
    action_magnitudes = []
    
    # Get list of all episode files in the directory
    demo_dir = "/home/ronak/Projects/gym-aloha/gym-aloha/data/sim_transfer_cube_scripted/"
    episode_files = sorted([f for f in os.listdir(demo_dir) if f.startswith("episode_") and f.endswith(".hdf5")])
    
    print("\nDemonstration Statistics:")
    for episode_file in episode_files:
        file_path = os.path.join(demo_dir, episode_file)
        try:
            with h5py.File(file_path, 'r') as episode_file_data:
                top_view = np.array(episode_file_data['observations/images/top'])
                top_view = np.transpose(top_view, (0, 3, 1, 2))
                resized_view = []
                for frame in top_view:
                    frame = frame.transpose(1, 2, 0)
                    # Resize to match environment dimensions (240x140)
                    resized = cv2.resize(frame, (240, 140))
                    resized = resized.transpose(2, 0, 1)
                    resized_view.append(resized)
                top_view = np.array(resized_view)
                
                actions = np.array(episode_file_data['action'])
                action_mag = np.linalg.norm(actions, axis=1)
                action_magnitudes.extend(action_mag)
                
                # Extract episode number from filename
                episode_num = int(os.path.basename(file_path).split('_')[1].split('.')[0])
                
                print(f"Episode {episode_num}: Frames = {len(top_view)}, "
                      f"Avg Action Magnitude = {np.mean(action_mag):.4f}, "
                      f"Max Action Magnitude = {np.max(action_mag):.4f}")
                
                total_frames += len(top_view)
                demos.append(({"pixels": top_view.astype(np.uint8)}, actions))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"\nTotal Statistics:")
    print(f"Total Episodes: {len(demos)}")
    print(f"Total Frames: {total_frames}")
    print(f"Average Action Magnitude: {np.mean(action_magnitudes):.4f}")
    print(f"Std Dev Action Magnitude: {np.std(action_magnitudes):.4f}")
    
    return demos


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
            [0.0, 1.0],           # left gripper (not finge r)
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

def make_env():
    """Create the AlohaTransferCube environment with specific settings"""
    env = gym.make(
        "gym_aloha/AlohaTransferCube-v0",
        obs_type="top_only",
        observation_width=240,
        observation_height=140,
        render_mode="rgb_array"
    )
    return env

def train_bc(
    demonstrations,
    env,
    n_epochs=100,
    batch_size=32,
    lr=1e-4,
    save_path="bc_models",
):
    """Train behavioral cloning policy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Create networks
    features_extractor = CustomCNN(env.observation_space, features_dim=512)
    features_extractor.to(device)
    
    # Fix: Pass the env.action_space instead of action_space.shape[0]
    policy = BCPolicy(features_extractor, env.action_space)
    policy.to(device)

    # Create dataset and dataloader
    dataset = DemonstrationDataset(demonstrations)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    # Setup training
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()
    
    # Training loop setup
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 15
    min_lr = 1e-6
    epoch_losses = []
    epoch_maes = []
    
    print("\nStarting BC Training:")
    for epoch in range(n_epochs):
        policy.train()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        for batch_obs, batch_actions in tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            batch_actions = batch_actions.to(device).float()
            processed_obs = {k: torch.as_tensor(v, device=device).float() for k, v in batch_obs.items()}
            
            # Get normalized predicted actions (in [-1, 1])
            normalized_predicted_actions = policy.get_normalized_actions(processed_obs)
            
            # Compute losses using normalized actions
            loss = criterion(normalized_predicted_actions, batch_actions)
            mae = torch.mean(torch.abs(normalized_predicted_actions - batch_actions))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
        
        # Epoch statistics
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        epoch_losses.append(avg_loss)
        epoch_maes.append(avg_mae)
        
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        print(f"Loss: {avg_loss:.6f}")
        print(f"MAE: {avg_mae:.6f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Learning rate scheduling and early stopping
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(policy.state_dict(), "bc_models/best_bc_policy3.pth")
        else:
            patience_counter += 1
        
        if patience_counter >= patience_limit:
            print("Early stopping triggered!")
            break
        
        if optimizer.param_groups[0]['lr'] < min_lr:
            print("Learning rate too small, stopping training!")
            break
    
    # Save training metrics
    os.makedirs("training_metrics", exist_ok=True)
    np.save("training_metrics/bc_training_losses.npy", epoch_losses)
    np.save("training_metrics/bc_training_maes.npy", epoch_maes)
    
    # Load best model
    policy.load_state_dict(torch.load("bc_models/best_bc_policy3.pth"))
    return policy, epoch_losses, epoch_maes

def main():
    # Load demonstrations
    demonstrations = load_demonstrations()
    
    # Create environment
    env = gym.make(
        "gym_aloha/AlohaTransferCube-v0",
        obs_type="top_only",
        observation_width=240,
        observation_height=140,
        render_mode="rgb_array"
    )
    
    print("\nStarting behavioral cloning training...")
    
    # Fix: Changed 'epochs' to 'n_epochs' to match function definition
    policy, losses, maes = train_bc(
        demonstrations=demonstrations,
        env=env,
        n_epochs=100,  # Changed from epochs to n_epochs
        batch_size=32,
        lr=1e-4,
        save_path="bc_models"
    )

if __name__ == "__main__":
    main()