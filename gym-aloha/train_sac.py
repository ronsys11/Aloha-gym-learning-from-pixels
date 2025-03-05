from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium as gym
import gym_aloha
import torch
import h5py
import numpy as np
import cv2

def load_demonstrations():
    demos = []
    with h5py.File("/home/ronak/Projects/gym-aloha/gym-aloha/data/sim_transfer_cube_scripted/episode_0.hdf5", 'r') as f:
        print("First episode data shape:", f['observations/images/top'].shape)
        
        for episode in range(50):
            file_path = f"/home/ronak/Projects/gym-aloha/gym-aloha/data/sim_transfer_cube_scripted/episode_{episode}.hdf5"
            with h5py.File(file_path, 'r') as episode_file:
                # Shape is (timesteps, H, W, C) -> need (timesteps, C, H, W)
                top_view = np.array(episode_file['observations/images/top'])
                # Transpose each timestep individually
                top_view = np.transpose(top_view, (0, 3, 1, 2))
                # Resize to match environment dimensions (140, 240)
                resized_view = []
                for frame in top_view:
                    frame = frame.transpose(1, 2, 0)  # CHW -> HWC 
                    resized = cv2.resize(frame, (240, 140))
                    resized = resized.transpose(2, 0, 1)  # HWC -> CHW
                    resized_view.append(resized)
                top_view = np.array(resized_view)
                
                obs = {
                    'pixels': top_view.astype(np.uint8)
                }
                actions = np.array(episode_file['action'])
                
                # Generate synthetic rewards based on the task structure
                # Assuming the demonstration is successful and follows the reward structure:
                # 1: right gripper touches cube
                # 2: right gripper lifts cube
                # 3: left gripper touches cube
                # 4: successful transfer (left gripper has cube, right released, not on table)
                timesteps = len(actions)
                rewards = np.zeros(timesteps)
                
                # Approximate rewards based on demonstration progress
                first_quarter = timesteps // 4
                second_quarter = timesteps // 2
                third_quarter = 3 * timesteps // 4
                
                # Assign increasing rewards as the demonstration progresses
                rewards[:first_quarter] = 1  # Right gripper touches
                rewards[first_quarter:second_quarter] = 2  # Right gripper lifts
                rewards[second_quarter:third_quarter] = 3  # Left gripper touches
                rewards[third_quarter:] = 4  # Successful transfer
                
                # Set done flag for last step
                dones = np.zeros(timesteps, dtype=bool)
                dones[-1] = True
                
                demos.append((obs, actions, rewards, dones))
    
    if demos:
        print(f"Demonstration observation shape: {demos[0][0]['pixels'].shape}")
        print(f"Demonstration action shape: {demos[0][1].shape}")
        print(f"Demonstration rewards shape: {demos[0][2].shape}")
        print(f"Demonstration dones shape: {demos[0][3].shape}")
    
    return demos

def make_env():
    env = gym.make(
        "gym_aloha/AlohaTransferCube-v0",
        obs_type="top_only",
        observation_width=240,
        observation_height=140,
    )
    return env

# print observation space for debugging
env = make_vec_env(make_env, n_envs=1)
print("Environment observation space:", env.observation_space)

demonstrations = load_demonstrations()
print(f"Loaded {len(demonstrations)} demonstrations")
if demonstrations:
    print(f"First demonstration shape - Observations: {demonstrations[0][0]['pixels'].shape}, Actions: {demonstrations[0][1].shape}")

# Configure SAC with improved hyperparameters
model = SAC(
    "MultiInputPolicy",
    env,
    learning_rate=3e-4,  # Reduced from 3e-4 for more stable learning
    buffer_size=100000,
    batch_size=128,      # Increased from 64 for better gradient estimates
    tau=0.005,           # Reduced from 0.02 for smoother target updates
    gamma=0.98,
    learning_starts=0,  # Increased from 5000 for better initial exploration
    train_freq=(1, "episode"),
    gradient_steps=4,    # Reduced from 32 to prevent overfitting and improve speed
    ent_coef="auto",
    target_entropy=-14,
    policy_kwargs=dict(
        net_arch=dict(
            pi=[256, 256],
            qf=[256, 256]
        ),
        log_std_init=-2,  # More conservative initial actions
        
    ),
    verbose=1,
    tensorboard_log="./sac_tensorboard/",
    device="cuda"
)

print("Loading demonstrations into replay buffer...")
for obs_demo, actions_demo, rewards_demo, dones_demo in demonstrations:
    for t in range(len(actions_demo) - 1):
        next_obs = {
            'pixels': obs_demo['pixels'][t + 1]
        }
        obs = {
            'pixels': obs_demo['pixels'][t]
        }
        action = actions_demo[t]
        reward = rewards_demo[t]  # Use the generated rewards
        done = dones_demo[t]      # Use the generated done flags
        
        model.replay_buffer.add(
            obs,
            next_obs,
            action,
            reward,
            done,
            [{}]
        )

# Create a checkpoint callback to save models during training
checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path="./sac_checkpoints/",
    name_prefix="sac_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

print("Starting training...")
model.learn(
    total_timesteps=250000,  # Increased from 250000 for better convergence
    progress_bar=True,
    callback=checkpoint_callback
)

model.save("sac_end_effector_with_demos4")