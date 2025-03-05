import torch
import numpy as np
import gymnasium as gym
import h5py
import os
from pure_bc import CustomCNN, BCPolicy, DemonstrationDataset, load_demonstrations
from torch.utils.data import DataLoader

def validate_demo_loading():
    """Validate demonstration loading and statistics"""
    print("\n=== Demonstration Loading Validation ===")
    demos = load_demonstrations()
    
    # Check first demo
    first_demo = demos[0]
    obs_dict, actions = first_demo
    
    print(f"Number of demonstrations: {len(demos)}")
    print("\nFirst demonstration stats:")
    print(f"Observation shape: {obs_dict['pixels'].shape}")
    print(f"Observation range: [{obs_dict['pixels'].min()}, {obs_dict['pixels'].max()}]")
    print(f"Actions shape: {actions.shape}")
    print(f"Actions range: [{actions.min()}, {actions.max()}]")
    
    # Print per-joint statistics for the first demo
    print("\nPer-joint action ranges (first demo):")
    for joint_idx in range(actions.shape[1]):
        joint_actions = actions[:, joint_idx]
        print(f"Joint {joint_idx}: [{joint_actions.min():.4f}, {joint_actions.max():.4f}]")
    
    return demos


def validate_environment():
    """Validate environment setup and action space"""
    print("\n=== Environment Validation ===")
    env = gym.make(
        "gym_aloha/AlohaTransferCube-v0",
        obs_type="top_only",
        observation_width=240,
        observation_height=140,
        render_mode="rgb_array"
    )
    
    print("Observation Space:")
    print(f"Shape: {env.observation_space['pixels'].shape}")
    print(f"Type: {env.observation_space['pixels'].dtype}")
    
    print("\nAction Space:")
    print(f"Shape: {env.action_space.shape}")
    print(f"Bounds: [{env.action_space.low}, {env.action_space.high}]")
    
    return env

def validate_dataset(env, demos):
    """Validate dataset creation and normalization"""
    print("\n=== Dataset Validation ===")
    
    # Create dataset without action_space
    dataset = DemonstrationDataset(demos)
    
    # Check first item
    first_obs, first_action = dataset[0]
    original_action = demos[0][1][0]
    
    print("First dataset item:")
    print(f"Observation shape: {first_obs['pixels'].shape}")
    print(f"Observation range: [{first_obs['pixels'].min()}, {first_obs['pixels'].max()}]")
    print(f"Normalized action range: [{first_action.min()}, {first_action.max()}]")
    
    # Verify per-joint normalization
    print("\nPer-joint normalization check:")
    for joint_idx in range(len(dataset.joint_ranges)):
        joint_min, joint_max = dataset.joint_ranges[joint_idx]
        original_joint = original_action[joint_idx]
        normalized_joint = first_action[joint_idx]
        
        # Manual normalization check
        expected_norm = 2.0 * (original_joint - joint_min) / (joint_max - joint_min) - 1.0
        
        print(f"\nJoint {joint_idx}:")
        print(f"Original: {original_joint:.4f}")
        print(f"Joint limits: [{joint_min:.4f}, {joint_max:.4f}]")
        print(f"Normalized: {normalized_joint:.4f}")
        print(f"Expected normalized: {expected_norm:.4f}")
        
        # Check if original action is within joint limits
        if original_joint < joint_min or original_joint > joint_max:
            print(f"WARNING: Original action outside joint limits!")
        
        # Check normalization accuracy
        if abs(normalized_joint - expected_norm) > 1e-5:
            print(f"WARNING: Normalization mismatch! Diff: {abs(normalized_joint - expected_norm):.6f}")
    
    return dataset

def validate_model(env, dataset):
    """Validate model architecture and forward pass"""
    print("\n=== Model Validation ===")
    
    features_extractor = CustomCNN(env.observation_space, features_dim=512)
    policy = BCPolicy(features_extractor, env.action_space)
    
    # Test forward pass
    first_obs, first_action = dataset[0]
    obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0) for k, v in first_obs.items()}
    
    with torch.no_grad():
        normalized_actions = policy.get_normalized_actions(obs_tensor)
        final_actions = policy(obs_tensor)
        
        print("\nForward pass test:")
        print(f"Input observation shape: {obs_tensor['pixels'].shape}")
        
        # Print per-joint predictions
        print("\nPer-joint predictions:")
        for joint_idx in range(len(policy.joint_ranges)):
            norm_val = normalized_actions[0, joint_idx].item()
            final_val = final_actions[0, joint_idx].item()
            joint_min, joint_max = policy.joint_ranges[joint_idx]
            
            print(f"\nJoint {joint_idx}:")
            print(f"Normalized: {norm_val:.4f}")
            print(f"Final: {final_val:.4f}")
            print(f"Expected range: [{joint_min:.4f}, {joint_max:.4f}]")
            
            if final_val < joint_min or final_val > joint_max:
                print("WARNING: Prediction outside joint range!")
def validate_training_batch(env, dataset):
    """Validate a single training batch"""
    print("\n=== Training Batch Validation ===")
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    features_extractor = CustomCNN(env.observation_space, features_dim=512)
    policy = BCPolicy(features_extractor, env.action_space)
    optimizer = torch.optim.Adam(policy.parameters(), lr=2e-4)
    criterion = torch.nn.MSELoss()
    
    # Process one batch
    batch_obs, batch_actions = next(iter(dataloader))
    batch_actions = batch_actions.float()
    
    print("\nBatch statistics:")
    print(f"Batch observation shape: {batch_obs['pixels'].shape}")
    print(f"Batch observation range: [{batch_obs['pixels'].min()}, {batch_obs['pixels'].max()}]")
    print(f"Batch actions shape: {batch_actions.shape}")
    print(f"Batch actions range: [{batch_actions.min()}, {batch_actions.max()}]")
    
    # Forward pass
    optimizer.zero_grad()
    predicted_normalized = policy.get_normalized_actions(batch_obs)
    predicted_actions = policy(batch_obs)
    loss = criterion(predicted_normalized, batch_actions)
    
    print("\nTraining step results:")
    print(f"Predicted normalized range: [{predicted_normalized.min().item()}, {predicted_normalized.max().item()}]")
    print(f"Predicted final range: [{predicted_actions.min().item()}, {predicted_actions.max().item()}]")
    print(f"Loss: {loss.item()}")
    
    # Backward pass
    loss.backward()
    print("\nGradient statistics:")
    for name, param in policy.named_parameters():
        if param.grad is not None:
            print(f"{name} grad range: [{param.grad.min().item()}, {param.grad.max().item()}]")

def validate_joint_ranges(demo_actions, policy):
    """Validate joint range handling"""
    print("\n=== Joint Range Validation ===")
    
    for joint_idx in range(len(policy.joint_ranges)):
        joint_min, joint_max = policy.joint_ranges[joint_idx]
        demo_min = np.min(demo_actions[:, joint_idx])
        demo_max = np.max(demo_actions[:, joint_idx])
        
        print(f"\nJoint {joint_idx}:")
        print(f"Allowed range: [{joint_min:.4f}, {joint_max:.4f}]")
        print(f"Demo range: [{demo_min:.4f}, {demo_max:.4f}]")
        if demo_min < joint_min or demo_max > joint_max:
            print("WARNING: Demo actions outside allowed range!")

def main():
    print("Starting comprehensive validation...")
    
    # Run all validation steps
    demos = validate_demo_loading()
    env = validate_environment()
    dataset = validate_dataset(env, demos)
    validate_model(env, dataset)
    validate_training_batch(env, dataset)
    
    print("\nValidation complete!")

if __name__ == "__main__":
    main()