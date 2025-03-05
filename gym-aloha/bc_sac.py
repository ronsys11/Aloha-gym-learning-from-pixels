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
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from collections import deque
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def load_demonstrations():
    """Load demonstrations with detailed statistics"""
    demos = []
    total_frames = 0
    action_magnitudes = []
    
    print("\nDemonstration Statistics:")
    for episode in range(50):
        file_path = f"/home/ronak/Projects/gym-aloha/gym-aloha/data/sim_transfer_cube_scripted/episode_{episode}.hdf5"
        with h5py.File(file_path, 'r') as episode_file:
            top_view = np.array(episode_file['observations/images/top'])
            top_view = np.transpose(top_view, (0, 3, 1, 2))
            resized_view = []
            for frame in top_view:
                frame = frame.transpose(1, 2, 0)
                # Resize to match environment dimensions (240x140)
                resized = cv2.resize(frame, (240, 140))
                resized = resized.transpose(2, 0, 1)
                resized_view.append(resized)
            top_view = np.array(resized_view)
            
            actions = np.array(episode_file['action'])
            action_mag = np.linalg.norm(actions, axis=1)
            action_magnitudes.extend(action_mag)
            
            print(f"Episode {episode}: Frames = {len(top_view)}, "
                  f"Avg Action Magnitude = {np.mean(action_mag):.4f}, "
                  f"Max Action Magnitude = {np.max(action_mag):.4f}")
            
            total_frames += len(top_view)
            demos.append(({"pixels": top_view.astype(np.uint8)}, actions))
    
    print(f"\nTotal Statistics:")
    print(f"Total Frames: {total_frames}")
    print(f"Average Action Magnitude: {np.mean(action_magnitudes):.4f}")
    print(f"Std Dev Action Magnitude: {np.std(action_magnitudes):.4f}")
    
    return demos

class BCTrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.losses = []
        self.maes = []
        self.entropies = []
        
    def _on_step(self) -> bool:
        if self.model.ep_info_buffer and len(self.model.ep_info_buffer) > 0:
            reward = self.model.ep_info_buffer[-1]['r']
            
            # Get current observation safely
            # Instead of trying to access the raw environment, use the last observation
            last_obs = self.model.rollout_buffer.observations[-1] if hasattr(self.model, 'rollout_buffer') and \
                       self.model.rollout_buffer is not None and \
                       len(self.model.rollout_buffer.observations) > 0 else None
            
            # Calculate policy entropy if we have an observation
            if last_obs is not None:
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(last_obs, device=self.model.device)
                    entropy = self.model.policy.get_distribution(obs_tensor).entropy().mean().item()
                    self.entropies.append(entropy)
                    print(f"Timestep {self.num_timesteps}")
                    print(f"Episode Reward: {reward:.4f}")
                    print(f"Policy Entropy: {entropy:.4f}")
            else:
                print(f"Timestep {self.num_timesteps}")
                print(f"Episode Reward: {reward:.4f}")
                
        return True

class DemonstrationDataset(Dataset):
    def __init__(self, demonstrations):
        self.observations = []
        self.actions = []
        
        # Process demonstrations
        for obs_demo, actions_demo in demonstrations:
            # Process each timestep in the demonstration
            for t in range(len(actions_demo)):
                # Normalize pixels to [0,1] range and convert to float32
                pixels = obs_demo['pixels'][t].astype(np.float32) / 255.0
                self.observations.append({'pixels': pixels})
                self.actions.append(actions_demo[t])
        
        print(f"Created dataset with {len(self.observations)} samples")
        
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]
    

def train_bc(model, demonstrations, epochs=100, batch_size=256, lr=1e-4):
    """Train the SAC policy using behavioral cloning with comprehensive monitoring"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    os.makedirs("bc_models", exist_ok=True)
    
    dataset = DemonstrationDataset(demonstrations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Access the actor network directly
    actor = model.actor.to(device)
    
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 15
    min_lr = 1e-6
    epoch_losses = []
    epoch_maes = []
    
    print("\nBC Training Progress:")
    for epoch in range(epochs):
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        for batch_obs, batch_actions in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_actions = batch_actions.to(device).float()
            processed_obs = {k: torch.as_tensor(v, device=device).float() for k, v in batch_obs.items()}
            
            # Forward pass through the actor network
            # Pass the features_extractor explicitly to extract_features
            features = actor.extract_features(processed_obs, actor.features_extractor)
            
            # Then get the latent policy representation
            latent_pi = actor.latent_pi(features)
            
            # Finally get the action mean
            action_mean = actor.mu(latent_pi)
            
            loss = criterion(action_mean, batch_actions)
            mae = torch.mean(torch.abs(action_mean - batch_actions))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        epoch_losses.append(avg_loss)
        epoch_maes.append(avg_mae)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Loss: {avg_loss:.6f}")
        print(f"MAE: {avg_mae:.6f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(actor.state_dict(), "bc_models/best_bc_policy.pth")
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print("Early stopping triggered!")
            break
            
        if optimizer.param_groups[0]['lr'] < min_lr:
            print("Learning rate too small, stopping training!")
            break
    
    actor.load_state_dict(torch.load("bc_models/best_bc_policy.pth"))
    model.actor.load_state_dict(actor.state_dict())
    return model, epoch_losses, epoch_maes

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

    
def main():
    # Create environment
    env = make_vec_env(make_env, n_envs=1)
    
    # Initialize SAC with custom CNN and MUCH smaller buffer size
    model = SAC(
        "MultiInputPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50000,  # Reduced from 1,000,000 to 5,000
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        learning_starts=1000,
        train_freq=(1, "episode"),
        gradient_steps=8,
        ent_coef="auto",
        target_entropy=-14,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[512, 512, 256],
                qf=[512, 512, 256]
            ),
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(
                features_dim=512
            ),
            log_std_init=-3,
        ),
        verbose=1,
        tensorboard_log="./hybrid_tensorboard/",
        device="cuda"
    )
    
    # Load and analyze demonstrations
    print("Loading and analyzing demonstrations...")
    demonstrations = load_demonstrations()
    
    # BC pretraining with monitoring - capture return values
    print("\nStarting BC pretraining...")
    model, bc_losses, bc_maes = train_bc(  # Changed variable names to be more descriptive
        model,
        demonstrations,
        epochs=200,
        batch_size=128,
        lr=2e-4
    )
    
    # Save BC metrics with more descriptive names
    os.makedirs("training_metrics", exist_ok=True)  # Create directory if it doesn't exist
    np.save("training_metrics/bc_training_losses.npy", bc_losses)
    np.save("training_metrics/bc_training_maes.npy", bc_maes)
    
    # Create custom callback for SAC training
    sac_callback = BCTrainingCallback(verbose=1)
    
    # Rest of your code...
    
    # SAC fine-tuning with monitoring
    print("\nStarting SAC fine-tuning...")
    model.learn(
        total_timesteps=500000,
        callback=[
            CheckpointCallback(
                save_freq=500000,
                save_path="./hybrid_checkpoints2/",
                name_prefix="hybrid_model"
            ),
            sac_callback
        ]
    )
    
    # Save final model and training statistics
    model.save("hybrid_bc_sac_final")
    np.save("sac_training_entropies2.npy", sac_callback.entropies)

if __name__ == "__main__":
    main()