import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import gym_aloha
import os
from tqdm import tqdm
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from collections import deque

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
    
    # Load your latest model (version 9)
    model_path = "/home/ronak/Projects/gym-aloha/hybrid_bc_sac_continued_9.zip"  # Updated to version 9
    print(f"Loading model from {model_path}")
    
    model = SAC.load(
        model_path,
        env=env,
        device="cuda",
        buffer_size=100000,  # Keeping the same buffer size
        batch_size=128,
        verbose=1,
        tensorboard_log="./continued_training_tensorboard_10/"  # Increment to version 10
    )
    
    # Create directories for saving results
    os.makedirs("./continued_checkpoints_10/", exist_ok=True)  # Increment to version 10
    os.makedirs("./continued_metrics_10/", exist_ok=True)      # Increment to version 10
    
    # Create custom callback for continued SAC training
    sac_callback = BCTrainingCallback(verbose=1)
    
    # Continue SAC training
    print("\nContinuing SAC training (round 10)...")  # Update round number
    model.learn(
        total_timesteps=700000,  # Additional timesteps
        callback=[
            CheckpointCallback(
                save_freq=500000,  # Save checkpoints every 50k steps
                save_path="./continued_checkpoints_10/",
                name_prefix="continued_model_10"
            ),
            sac_callback
        ],
        reset_num_timesteps=False  # Keep the timestep counter going
    )
    
    # Save final model and training statistics
    final_model_path = "hybrid_bc_sac_continued_10"  # Increment to version 10
    print(f"Saving continued model to {final_model_path}")
    model.save(final_model_path)
    
    # Save training metrics
    np.save("./continued_metrics_10/sac_training_entropies.npy", sac_callback.entropies)
    print("Training metrics saved to ./continued_metrics_10/")
    
    print("Tenth round of continued training complete!")

if __name__ == "__main__":
    main()