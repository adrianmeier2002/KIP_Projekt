"""Training script for Hybrid Agent (Constraint Solver + DQN)."""

import os
import argparse
import numpy as np
from src.reinforcement_learning.hybrid_agent import HybridAgent
from src.reinforcement_learning.dqn_agent import DQNAgent  # For pure RL comparison
from src.reinforcement_learning.environment import MinesweeperEnvironment


class LinearSchedule:
    """Simple linear schedule for epsilon or other scalar hyperparameters."""
    
    def __init__(self, start: float, end: float, duration: int):
        self.start = start
        self.end = end
        self.duration = max(1, duration)
    
    def value(self, step: int) -> float:
        """Return scheduled value for the given step (starting at 1)."""
        progress = min(1.0, max(0.0, step / self.duration))
        return self.start + (self.end - self.start) * progress


def run_greedy_evaluation(
    agent,  # Can be DQNAgent or HybridAgent
    difficulty: str,
    width: int,
    height: int,
    episodes: int,
    use_flag_actions: bool
):
    """Run evaluation episodes with epsilon=0 to measure actual policy quality."""
    if episodes <= 0:
        return None
    
    eval_env = MinesweeperEnvironment(
        difficulty, width, height, use_flag_actions=use_flag_actions
    )
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    # Reset solver statistics if hybrid agent
    if hasattr(agent, 'reset_stats'):
        agent.reset_stats()
    
    wins = 0
    total_rewards = []
    total_lengths = []
    
    for _ in range(episodes):
        state = eval_env.reset()
        done = False
        reward_sum = 0.0
        steps = 0
        
        while not done:
            valid_actions = eval_env.get_valid_actions()
            # Pass game object for hybrid agent
            action = agent.select_action(state, valid_actions, game=eval_env.game)
            state, reward, done, info = eval_env.step(action)
            reward_sum += reward
            steps += 1
        
        if info["won"]:
            wins += 1
        total_rewards.append(reward_sum)
        total_lengths.append(steps)
    
    agent.epsilon = original_epsilon
    
    result = {
        "episodes": episodes,
        "win_rate": (wins / episodes) * 100.0,
        "avg_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "avg_length": float(np.mean(total_lengths)) if total_lengths else 0.0
    }
    
    # Add solver statistics if hybrid agent
    if hasattr(agent, 'get_stats'):
        result['solver_stats'] = agent.get_stats()
    
    return result


def train_with_visualization(
    episodes: int = 1000,
    difficulty: str = "medium",
    width: int = None,
    height: int = None,
    save_path: str = "models/dqn_model.pth",
    log_interval: int = 100,
    visualization_callback: callable = None,
    thread=None,
    use_flag_actions: bool = False,
    eval_episodes: int = 20,
    use_hybrid: bool = True
):
    """
    Train agent on Minesweeper with visualization support.
    
    Args:
        episodes: Number of training episodes
        difficulty: Game difficulty
        width: Board width (uses current game width if None)
        height: Board height (uses current game height if None)
        save_path: Path to save model
        log_interval: Episodes between logging
        visualization_callback: Callback function(agent, episode) for visualization
        thread: Thread object for checking if training should continue
        use_flag_actions: Whether the action space should include flag toggles
        eval_episodes: Number of greedy evaluation episodes (epsilon=0) per log interval
        use_hybrid: Use Hybrid Agent (Solver + RL) vs Pure RL (default: True)
    """
    from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT
    
    # Use defaults if not provided
    if width is None:
        width = BOARD_WIDTH
    if height is None:
        height = BOARD_HEIGHT
    
    # Create directories
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Initialize environment and agent
    env = MinesweeperEnvironment(difficulty, width, height, use_flag_actions=use_flag_actions)
    
    # OPTIMIZED: Adjust hyperparameters based on board size
    # Smaller boards need faster learning, larger boards need more exploration
    board_size_factor = (width * height) / 600.0  # Normalize to default 30x20
    
    # OPTIMIZED: Learning rate - slightly lower for more stable training
    base_lr = 0.0005  # Reduced from 0.001
    lr = base_lr * (1.0 + 0.3 * (1.0 / max(board_size_factor, 0.5)))
    
    # Discount factor: prioritize near-term progress on default boards
    gamma = 0.95 if board_size_factor <= 1.0 else 0.98
    
    # Exploration schedule derived from episode count
    epsilon_start = 1.0
    epsilon_floor = {
        "easy": 0.03,
        "medium": 0.05,
        "hard": 0.1
    }.get(difficulty, 0.05)
    decay_span = max(int(episodes * 0.7), 1)
    epsilon_decay = (epsilon_floor / epsilon_start) ** (1.0 / decay_span)
    epsilon_schedule = LinearSchedule(epsilon_start, epsilon_floor, decay_span)
    
    # OPTIMIZED: Batch size - slightly larger for better gradient estimation
    batch_size = min(128, max(48, int(64 * (1.0 / max(board_size_factor, 0.6)))))  # Increased from 32-96 to 48-128
    
    # OPTIMIZED: Buffer size - larger for more diverse experiences
    buffer_size = min(100000, max(20000, int(50000 * max(board_size_factor, 0.5))))  # Default was 10000
    
    # OPTIMIZED: Target network update frequency - more frequent updates
    target_update = max(50, int(100 * board_size_factor))  # Default was 100
    
    # Create agent (Hybrid or Pure RL)
    AgentClass = HybridAgent if use_hybrid else DQNAgent
    agent = AgentClass(
        state_channels=env.state_channels,
        action_space_size=env.action_space_size,
        board_height=height,
        board_width=width,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_floor,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        buffer_size=buffer_size,
        target_update=target_update,
        use_solver=use_hybrid  # Only relevant for HybridAgent
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    episode_wins = []  # Track wins for each episode
    
    print(f"Starting training for {episodes} episodes...")
    print(f"Mode: {'HYBRID (Solver + RL)' if use_hybrid else 'PURE RL'}")
    print(f"Difficulty: {difficulty}")
    print(f"Board size: {width}x{height}")
    print(f"Device: {agent.device}")
    print("-" * 50)
    
    for episode in range(episodes):
        # Check if thread wants to stop
        if thread and not thread.running:
            print("Training stopped by user")
            break
        
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Get valid actions
            valid_actions = env.get_valid_actions()
            
            # Select action (pass game object for hybrid agent)
            action = agent.select_action(state, valid_actions, game=env.game)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            next_valid_actions = env.get_valid_actions()
            
            # Store experience
            agent.remember(state, action, reward, next_state, done, next_valid_actions)
            
            # Train agent (only if we have enough experience)
            # Train every step once we have enough samples for better learning
            loss = agent.train_step()
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
        
        # Update epsilon via schedule (fixed slope, avoids early stagnation)
        agent.epsilon = epsilon_schedule.value(episode + 1)
        
        # Track statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_wins.append(1 if info['won'] else 0)  # Track win/loss per episode
        
        # Logging and visualization
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_length = np.mean(episode_lengths[-log_interval:])
            # Calculate win rate from last log_interval episodes
            win_rate = np.sum(episode_wins[-log_interval:]) / log_interval * 100
            
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            
            if eval_episodes > 0:
                eval_stats = run_greedy_evaluation(
                    agent, difficulty, width, height, eval_episodes, use_flag_actions
                )
                if eval_stats:
                    print(
                        f"  Eval (ε=0) → Win Rate: {eval_stats['win_rate']:.1f}% | "
                        f"Avg Len: {eval_stats['avg_length']:.1f}"
                    )
                    # Show solver statistics for hybrid agent
                    if 'solver_stats' in eval_stats and use_hybrid:
                        stats = eval_stats['solver_stats']
                        print(
                            f"  Solver Usage → {stats['solver_percentage']:.1f}% | "
                            f"RL Guesses: {stats['rl_percentage']:.1f}%"
                        )
            print("-" * 50)
            
            # Visualization callback (for GUI) - every 100 episodes
            if visualization_callback:
                try:
                    visualization_callback(agent, episode + 1)
                except Exception as e:
                    print(f"Visualization callback error: {e}")
        
        # Save model periodically
        if (episode + 1) % (log_interval * 5) == 0:
            agent.save(save_path)
            print(f"Model saved to {save_path}")
    
    # Final save
    agent.save(save_path)
    print(f"\nTraining complete! Model saved to {save_path}")
    
    # Calculate final statistics
    final_episodes = min(100, len(episode_rewards))
    if final_episodes > 0:
        final_avg_reward = np.mean(episode_rewards[-final_episodes:])
        final_win_rate = np.sum(episode_wins[-final_episodes:]) / final_episodes * 100
        print(f"Final average reward (last {final_episodes} episodes): {final_avg_reward:.2f}")
        print(f"Final win rate (last {final_episodes} episodes): {final_win_rate:.1f}%")
    else:
        print(f"Final average reward: {np.mean(episode_rewards):.2f}")
        print(f"Final win rate: {np.sum(episode_wins) / len(episode_wins) * 100:.1f}%")
    
    if eval_episodes > 0:
        final_eval = run_greedy_evaluation(
            agent, difficulty, width, height, max(25, eval_episodes), use_flag_actions
        )
        if final_eval:
            print(
                f"Greedy evaluation ({final_eval['episodes']} episodes) → "
                f"Win Rate: {final_eval['win_rate']:.1f}% | "
                f"Avg Reward: {final_eval['avg_reward']:.2f} | "
                f"Avg Length: {final_eval['avg_length']:.1f}"
            )
    
    # Final test visualization
    if visualization_callback:
        print("\nFinal test run...")
        try:
            visualization_callback(agent, -1)  # -1 indicates final test
        except Exception as e:
            print(f"Final visualization error: {e}")


def train(
    episodes: int = 1000,
    difficulty: str = "medium",
    save_path: str = "models/dqn_model.pth",
    log_interval: int = 100,
    use_flag_actions: bool = False,
    width: int = None,
    height: int = None,
    eval_episodes: int = 20,
    use_hybrid: bool = True
):
    """
    Train agent on Minesweeper (without visualization).
    
    Args:
        episodes: Number of training episodes
        difficulty: Game difficulty
        save_path: Path to save model
        log_interval: Episodes between logging
        use_hybrid: Use Hybrid Agent (default: True)
    """
    train_with_visualization(
        episodes=episodes,
        difficulty=difficulty,
        save_path=save_path,
        log_interval=log_interval,
        visualization_callback=None,
        thread=None,
        use_flag_actions=use_flag_actions,
        width=width,
        height=height,
        eval_episodes=eval_episodes,
        use_hybrid=use_hybrid
    )


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train Hybrid Agent (Solver + RL) on Minesweeper")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--difficulty", type=str, default="medium", 
                       choices=["easy", "medium", "hard"], help="Game difficulty")
    parser.add_argument("--save-path", type=str, default="models/hybrid_model.pth",
                       help="Path to save model")
    parser.add_argument("--log-interval", type=int, default=100,
                       help="Episodes between logging")
    parser.add_argument("--use-flags", action="store_true",
                       help="Enable flagging actions in the action space")
    parser.add_argument("--width", type=int, default=None,
                       help="Optional board width override")
    parser.add_argument("--height", type=int, default=None,
                       help="Optional board height override")
    parser.add_argument("--eval-episodes", type=int, default=20,
                       help="Greedy evaluation episodes per log interval (0 to disable)")
    parser.add_argument("--no-hybrid", action="store_true",
                       help="Disable hybrid mode (use pure RL instead of Solver + RL)")
    
    args = parser.parse_args()
    
    train(
        episodes=args.episodes,
        difficulty=args.difficulty,
        save_path=args.save_path,
        log_interval=args.log_interval,
        use_flag_actions=args.use_flags,
        width=args.width,
        height=args.height,
        eval_episodes=args.eval_episodes,
        use_hybrid=not args.no_hybrid
    )


if __name__ == "__main__":
    main()
