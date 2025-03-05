import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns

# Set up the plotting style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]

def load_tensorboard_data(log_dir):
    """Load TensorBoard data into a pandas DataFrame"""
    print(f"Loading TensorBoard data from {log_dir}")
    
    # Find all event files
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return None
    
    # Load the first event file
    event_file = event_files[0]
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()
    
    # Get available tags (metrics)
    tags = event_acc.Tags()['scalars']
    print(f"Available metrics: {tags}")
    
    # Create a DataFrame for each tag
    dfs = {}
    for tag in tags:
        events = event_acc.Scalars(tag)
        dfs[tag] = pd.DataFrame(events)
    
    # Merge all DataFrames
    if dfs:
        # Create a common step column
        all_steps = sorted(set().union(*[set(df['step']) for df in dfs.values()]))
        result = pd.DataFrame({'step': all_steps})
        
        # Add each metric
        for tag, df in dfs.items():
            # Rename value column to tag name
            df = df.rename(columns={'value': tag})
            # Merge with result
            result = pd.merge(result, df[['step', tag]], on='step', how='left')
        
        return result
    
    return None

def load_bc_metrics():
    """Load BC training metrics"""
    print("Loading BC training metrics")
    
    losses_path = "training_metrics/bc_training_losses.npy"
    maes_path = "training_metrics/bc_training_maes.npy"
    
    if os.path.exists(losses_path) and os.path.exists(maes_path):
        losses = np.load(losses_path)
        maes = np.load(maes_path)
        
        # Create DataFrame
        epochs = np.arange(1, len(losses) + 1)
        bc_df = pd.DataFrame({
            'epoch': epochs,
            'loss': losses,
            'mae': maes
        })
        
        return bc_df
    else:
        print("BC metrics files not found")
        return None

def load_sac_entropies():
    """Load SAC training entropies"""
    print("Loading SAC entropies")
    
    entropies_path = "sac_training_entropies.npy"
    
    if os.path.exists(entropies_path):
        entropies = np.load(entropies_path)
        
        # Create DataFrame
        steps = np.arange(1, len(entropies) + 1)
        sac_df = pd.DataFrame({
            'step': steps,
            'entropy': entropies
        })
        
        return sac_df
    else:
        print("SAC entropies file not found")
        return None

def analyze_demonstration_statistics():
    """Analyze demonstration statistics from the logs"""
    print("Analyzing demonstration statistics from logs")
    
    # This would require parsing the text logs
    # For now, we'll just provide a summary based on what we've seen
    print("Demonstration Summary:")
    print("- 50 episodes with consistent frame counts (400 frames each)")
    print("- Average action magnitude around 2.4-2.5")
    print("- Low standard deviation in action magnitudes (consistent demonstrations)")

def plot_bc_training_metrics(bc_df):
    """Plot BC training metrics"""
    if bc_df is None:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot loss
    ax1.plot(bc_df['epoch'], bc_df['loss'], 'b-', linewidth=2)
    ax1.set_title('BC Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.grid(True)
    
    # Plot MAE
    ax2.plot(bc_df['epoch'], bc_df['mae'], 'r-', linewidth=2)
    ax2.set_title('BC Training Mean Absolute Error')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('bc_training_metrics.png')
    plt.close()
    
    print(f"BC training started with loss {bc_df['loss'].iloc[0]:.4f} and ended with {bc_df['loss'].iloc[-1]:.4f}")
    print(f"BC training started with MAE {bc_df['mae'].iloc[0]:.4f} and ended with {bc_df['mae'].iloc[-1]:.4f}")
    
    # Check for convergence
    if len(bc_df) > 10:
        last_10_loss_change = (bc_df['loss'].iloc[-1] - bc_df['loss'].iloc[-10]) / bc_df['loss'].iloc[-10]
        print(f"Loss change in last 10 epochs: {last_10_loss_change:.2%}")
        
        if abs(last_10_loss_change) < 0.05:
            print("BC training appears to have converged (less than 5% change in last 10 epochs)")
        else:
            print("BC training may not have fully converged")

def plot_sac_training_metrics(tb_df, sac_entropy_df):
    """Plot SAC training metrics"""
    if tb_df is None:
        return
    
    # Plot rewards
    if 'rollout/ep_rew_mean' in tb_df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(tb_df['step'], tb_df['rollout/ep_rew_mean'], 'g-', linewidth=2)
        plt.title('SAC Training Rewards')
        plt.xlabel('Step')
        plt.ylabel('Mean Episode Reward')
        plt.grid(True)
        plt.savefig('sac_rewards.png')
        plt.close()
        
        print(f"SAC training started with reward {tb_df['rollout/ep_rew_mean'].iloc[0]:.4f} and ended with {tb_df['rollout/ep_rew_mean'].iloc[-1]:.4f}")
    
    # Plot success rate if available
    if 'rollout/success_rate' in tb_df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(tb_df['step'], tb_df['rollout/success_rate'], 'm-', linewidth=2)
        plt.title('SAC Success Rate')
        plt.xlabel('Step')
        plt.ylabel('Success Rate')
        plt.grid(True)
        plt.savefig('sac_success_rate.png')
        plt.close()
        
        print(f"SAC training ended with success rate {tb_df['rollout/success_rate'].iloc[-1]:.4f}")
    
    # Plot entropy if available
    if sac_entropy_df is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(sac_entropy_df['step'], sac_entropy_df['entropy'], 'c-', linewidth=2)
        plt.title('SAC Policy Entropy')
        plt.xlabel('Step')
        plt.ylabel('Entropy')
        plt.grid(True)
        plt.savefig('sac_entropy.png')
        plt.close()
        
        print(f"SAC policy entropy started at {sac_entropy_df['entropy'].iloc[0]:.4f} and ended at {sac_entropy_df['entropy'].iloc[-1]:.4f}")

def main():
    """Main analysis function"""
    print("Starting analysis of training logs")
    
    # Load TensorBoard data
    tb_df = load_tensorboard_data("./hybrid_tensorboard/")
    
    # Load BC metrics
    bc_df = load_bc_metrics()
    
    # Load SAC entropies
    sac_entropy_df = load_sac_entropies()
    
    # Analyze demonstration statistics
    analyze_demonstration_statistics()
    
    # Plot BC training metrics
    plot_bc_training_metrics(bc_df)
    
    # Plot SAC training metrics
    plot_sac_training_metrics(tb_df, sac_entropy_df)
    
    # Provide overall analysis
    print("\nOverall Analysis:")
    
    if bc_df is not None and bc_df['mae'].iloc[-1] > 0.2:
        print("- BC training achieved moderate accuracy (MAE > 0.2)")
        print("  Suggestion: Increase BC training epochs or use a more expressive policy network")
    
    if tb_df is not None and 'rollout/ep_rew_mean' in tb_df.columns:
        reward_improvement = tb_df['rollout/ep_rew_mean'].iloc[-1] - tb_df['rollout/ep_rew_mean'].iloc[0]
        if reward_improvement <= 0:
            print("- SAC training did not improve rewards")
            print("  Suggestion: The BC policy may be suboptimal, or SAC hyperparameters need tuning")
        else:
            print(f"- SAC training improved rewards by {reward_improvement:.4f}")
    
    if sac_entropy_df is not None:
        avg_entropy = sac_entropy_df['entropy'].mean()
        if avg_entropy < 5.0:
            print(f"- Policy entropy is low ({avg_entropy:.4f})")
            print("  Suggestion: Increase exploration by adjusting ent_coef or target_entropy")
        else:
            print(f"- Policy entropy is reasonable ({avg_entropy:.4f})")
    
    print("\nRecommendations:")
    print("1. Increase BC training epochs to 200 for better imitation")
    print("2. Use a deeper network for the policy (e.g., [1024, 512, 256])")
    print("3. Increase SAC exploration by setting a higher target_entropy")
    print("4. Consider using a curriculum learning approach")

if __name__ == "__main__":
    main()