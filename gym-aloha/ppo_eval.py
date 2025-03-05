from stable_baselines3 import PPO
import gymnasium as gym
import gym_aloha
from collections import Counter

# Load the trained model
print("Loading PPO model...")
model = PPO.load("ppo_transfer_cube2")

# Create evaluation environment
env = gym.make("gym_aloha/AlohaTransferCube-v0", render_mode="rgb_array")

# Evaluate the agent
print("Starting evaluation...")
num_episodes = 50
episode_max_rewards = []
reward_counts = Counter()

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    max_reward = float('-inf')
    step_count = 0
    max_steps = 200000  # Prevent infinite loops per episode

    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        max_reward = max(max_reward, reward)

        reward_counts[reward] += 1
        step_count += 1

        if done or truncated:
            break

    episode_max_rewards.append(max_reward)

# Close the environment
env.close()

# Print evaluation summary
print("\nEvaluation Summary:")
print(f"Total Episodes: {num_episodes}")
print(f"Average Max Reward per Episode: {sum(episode_max_rewards) / num_episodes:.2f}")

print("\nReward Occurrences:")
for reward, count in reward_counts.items():
    print(f"Reward {reward}: {count} times")

print("\nMax Reward per Episode:")
for i, reward in enumerate(episode_max_rewards, 1):
    print(f"Episode {i}: {reward}")
