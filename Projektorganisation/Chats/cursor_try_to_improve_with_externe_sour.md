# Try to improve with externe Sources
_Exported on 11/24/2025 at 17:38:48 GMT+1 from Cursor (2.0.38)_

---

**User**

Ich probiere ein Minesweeper-Spiel mit einem reinforcement_learning programm zu kombinieren. Schaue bitte alle Programme an und verbessere ds RL programm, es lernt aktuell nicht korrekt. Nutze dabei: https://sdlee94.github.io/Minesweeper-AI-Reinforcement-Learning/  Als inspiration

---

**Cursor**

```python
"""RL Environment wrapper for Minesweeper."""

import numpy as np
from typing import Tuple, Optional
from src.minesweeper.game import Game
from src.utils.constants import (
    BOARD_WIDTH, BOARD_HEIGHT,
    CELL_HIDDEN, CELL_REVEALED, CELL_FLAGGED, CELL_MINE
)

STATE_CHANNELS = 7
NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]

FLAG_SET_REWARD = 0.2
FLAG_SET_PENALTY = 0.2
FLAG_FINAL_CORRECT_BONUS = 0.5
FLAG_FINAL_INCORRECT_PENALTY = 0.5


class MinesweeperEnvironment:
    """Environment wrapper for reinforcement learning."""
    
    def __init__(self, difficulty: str = "medium", width: int = BOARD_WIDTH, height: int = BOARD_HEIGHT):
        """
        Initialize the environment.
        
        Args:
            difficulty: Game difficulty level
            width: Board width
            height: Board height
        """
        self.difficulty = difficulty
        self.width = width
        self.height = height
        self.game = Game(difficulty, width, height)
        self.cell_count = width * height
        self.action_space_size = self.cell_count * 2  # Reveal + Flag per cell
        self.state_channels = STATE_CHANNELS
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment and return initial state.
        
        Returns:
            Initial state observation
        """
        self.game.new_game(self.difficulty, self.width, self.height)
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action index (0 to action_space_size - 1)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Convert action to row, col
        cell_index = action % self.cell_count if action >= self.cell_count else action
        row = cell_index // self.width
        col = cell_index % self.width
        action_type = "flag" if action >= self.cell_count else "reveal"
        
        # Get previous state
        prev_revealed = self.game.revealed_count
        reward = 0.0
        valid = False
        
        move_context = self._get_move_context(row, col)
        
        if action_type == "reveal":
            valid = self.game.reveal_cell(row, col)
            reward += self._calculate_reward(
                prev_revealed,
                valid,
                action_type=action_type,
                move_context=move_context
            )
        else:
            valid, flag_reward = self._handle_flag_action(row, col)
            reward += flag_reward
        
        # Check if done
        done = self.game.is_game_over()
        
        # Get next state
        next_state = self._get_state()
        
        # Info
        info = {
            "won": self.game.is_won(),
            "lost": self.game.is_lost(),
            "valid_action": valid
        }
        
        # Additional rewards/penalties when the game ends after this move
        if done and self.game.is_lost():
            reward += self._calculate_flag_end_bonus()
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation with better features.
        
        Returns:
            State array of shape (STATE_CHANNELS, height, width)
        """
        state = np.zeros((STATE_CHANNELS, self.height, self.width), dtype=np.float32)
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                adjacent_revealed = 0
                adjacent_hint_sum = 0
                hidden_neighbors = 0
                flagged_neighbors = 0
                
                for dr, dc in NEIGHBOR_OFFSETS:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        neighbor = self.game.board.get_cell(nr, nc)
                        if neighbor.is_hidden():
                            hidden_neighbors += 1
                        elif neighbor.is_flagged():
                            flagged_neighbors += 1
                        elif neighbor.is_revealed():
                            adjacent_revealed += 1
                            if not neighbor.is_mine:
                                adjacent_hint_sum += neighbor.adjacent_mines
                
                if cell.is_revealed():
                    if cell.is_mine:
                        state[0, row, col] = -1.0
                        state[3, row, col] = -1.0
                    else:
                        value = cell.adjacent_mines / 8.0
                        state[0, row, col] = value
                        state[3, row, col] = value
                elif cell.is_flagged():
                    state[0, row, col] = -0.5
                    state[2, row, col] = 1.0
                else:
                    neighbor_bonus = min(0.4, adjacent_revealed * 0.08)
                    hint_penalty = min(0.3, adjacent_hint_sum / 30.0)
                    cell_value = -0.9 + neighbor_bonus - hint_penalty
                    state[0, row, col] = max(-1.0, cell_value)
                    state[1, row, col] = 1.0
                
                state[4, row, col] = hidden_neighbors / 8.0
                state[5, row, col] = flagged_neighbors / 8.0
                state[6, row, col] = min(1.0, adjacent_hint_sum / 64.0)
        
        return state
    
    def _calculate_reward(
        self,
        prev_revealed: int,
        valid_action: bool,
        action_type: str,
        move_context: Optional[dict] = None
    ) -> float:
        """
        Calculate reward for the action.
        
        Args:
            prev_revealed: Number of revealed cells before action
            valid_action: Whether the action was valid
            
        Returns:
            Reward value
        """
        if action_type != "reveal":
            return 0.0
        
        if not valid_action:
            return -0.4  # Penalty for invalid action
        
        board_scale = max(0.6, (self.width * self.height) / 600.0)
        cells_revealed = self.game.revealed_count - prev_revealed
        progress_ratio = self.game.revealed_count / max(1, self.game.total_safe_cells)
        is_guess = move_context.get("adjacent_revealed", 0) == 0 if move_context else True
        frontier_bonus = 0.0
        
        if move_context:
            if is_guess:
                frontier_bonus = -0.2
            else:
                frontier_bonus = 0.15 + 0.05 * min(4, move_context.get("adjacent_revealed", 0))
        
        if self.game.is_lost():
            # Keep penalty moderate to avoid dwarfing positive rewards
            penalty = -5.0 * board_scale
            return penalty + (frontier_bonus if is_guess else 0.0)
        
        if self.game.is_won():
            return 10.0 * board_scale + 5.0 * progress_ratio
        
        if cells_revealed > 0:
            base_reward = 0.25 * cells_revealed
            expansion_bonus = 0.05 * min(5, max(0, cells_revealed - 1))
            shaping = 0.1 * progress_ratio
            return base_reward + expansion_bonus + frontier_bonus + shaping
        
        # Slight penalty for ineffective move
        return -0.05
    
    def _handle_flag_action(self, row: int, col: int) -> Tuple[bool, float]:
        """
        Handle flag toggling and return (valid, reward).
        """
        cell = self.game.board.get_cell(row, col)
        if not cell or cell.is_revealed():
            return False, -0.1
        
        toggled = self.game.toggle_flag(row, col)
        if not toggled:
            return False, -0.1
        
        if cell.is_flagged():
            if cell.is_mine:
                return True, FLAG_SET_REWARD
            return True, -FLAG_SET_PENALTY
        
        # Flag removed (no immediate reward to avoid oscillation)
        return True, 0.0
    
    def _calculate_flag_end_bonus(self) -> float:
        """Reward/punish based on flagged cells when the game ends."""
        correct_flags = 0
        incorrect_flags = 0
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                if cell and cell.is_flagged():
                    if cell.is_mine:
                        correct_flags += 1
                    else:
                        incorrect_flags += 1
        
        return (
            correct_flags * FLAG_FINAL_CORRECT_BONUS
            - incorrect_flags * FLAG_FINAL_INCORRECT_PENALTY
        )
    
    def get_valid_actions(self) -> np.ndarray:
        """
        Get mask of valid actions (cells that can be clicked).
        
        Returns:
            Boolean array of valid actions
        """
        valid_reveal = np.zeros(self.cell_count, dtype=bool)
        valid_flag = np.zeros(self.cell_count, dtype=bool)
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                action_idx = row * self.width + col
                
                # Reveal valid if cell hidden and not flagged
                if cell.is_hidden() and not cell.is_flagged():
                    valid_reveal[action_idx] = True
                
                # Flag valid if cell hidden or currently flagged (toggle)
                if not cell.is_revealed():
                    if cell.is_hidden() or cell.is_flagged():
                        valid_flag[action_idx] = True
        
        return np.concatenate([valid_reveal, valid_flag])
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get action mask for masking invalid actions in Q-values.
        
        Returns:
            Mask array (1.0 for valid, -inf for invalid)
        """
        valid_actions = self.get_valid_actions()
        mask = np.where(valid_actions, 0.0, -np.inf)
        return mask

    def _get_move_context(self, row: int, col: int) -> dict:
        """Collect contextual information for the selected move."""
        context = {
            "adjacent_revealed": 0,
            "adjacent_hidden": 0,
            "adjacent_flagged": 0
        }
        
        for dr, dc in NEIGHBOR_OFFSETS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                neighbor = self.game.board.get_cell(nr, nc)
                if not neighbor:
                    continue
                if neighbor.is_revealed():
                    context["adjacent_revealed"] += 1
                elif neighbor.is_flagged():
                    context["adjacent_flagged"] += 1
                elif neighbor.is_hidden():
                    context["adjacent_hidden"] += 1
        
        return context



```

```python
"""DQN Agent for Minesweeper."""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Tuple, Optional
from src.reinforcement_learning.network import DQNNetwork
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_valid_actions: np.ndarray
    ):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done, next_valid_actions))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_valid_actions = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        next_valid_actions = torch.BoolTensor(np.array(next_valid_actions))
        
        return states, actions, rewards, next_states, dones, next_valid_actions
    
    def __len__(self):
        """Get buffer size."""
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network Agent."""
    
    def __init__(
        self,
        state_channels: int = STATE_CHANNELS,
        action_space_size: int = 600,
        board_height: int = 20,
        board_width: int = 30,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update: int = 100,
        device: Optional[torch.device] = None
    ):
        """
        Initialize DQN Agent.
        
        Args:
            state_channels: Number of input channels
            action_space_size: Size of action space
            board_height: Height of the board
            board_width: Width of the board
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Initial epsilon for epsilon-greedy
            epsilon_end: Final epsilon
            epsilon_decay: Epsilon decay rate
            buffer_size: Replay buffer size
            batch_size: Batch size for training
            target_update: Steps between target network updates
            device: PyTorch device
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space_size = action_space_size
        self.board_height = board_height
        self.board_width = board_width
        self.cell_count = board_height * board_width
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        
        # Networks
        self.q_network = DQNNetwork(state_channels, action_space_size, board_height, board_width).to(self.device)
        self.target_network = DQNNetwork(state_channels, action_space_size, board_height, board_width).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.q_network.train()  # Start in training mode
        self.target_network.eval()  # Target network always in eval mode
        
        # Optimizer & loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        self._neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                                  (0, -1),           (0, 1),
                                  (1, -1),  (1, 0),  (1, 1)]
    
    def select_action(self, state: np.ndarray, valid_actions: Optional[np.ndarray] = None) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            valid_actions: Boolean array of valid actions
            
        Returns:
            Selected action index
        """
        if random.random() < self.epsilon:
            if valid_actions is not None:
                frontier_action = self._sample_frontier_action(state, valid_actions)
                if frontier_action is not None:
                    return frontier_action
                
                # Prefer reveal actions during random exploration
                reveal_mask = valid_actions[:self.cell_count]
                reveal_indices = np.where(reveal_mask)[0]
                if reveal_indices.size > 0:
                    return int(random.choice(reveal_indices.tolist()))
                
                valid_indices = np.where(valid_actions)[0]
                if len(valid_indices) > 0:
                    return int(random.choice(valid_indices.tolist()))
            return random.randint(0, self.action_space_size - 1)
        
        # Greedy action
        self.q_network.eval()  # Set to eval mode for inference
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
            # Mask invalid actions
            if valid_actions is not None:
                mask = torch.FloatTensor(valid_actions).to(self.device)
                q_values = q_values + (1 - mask) * -1e9
            
            action = q_values.argmax().item()
        
        return action
    
    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_valid_actions: np.ndarray
    ):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done, next_valid_actions)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Set network to training mode
        self.q_network.train()
        
        # Sample batch
        states, actions, rewards, next_states, dones, next_valid_actions = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        next_valid_actions = next_valid_actions.to(self.device)
        
        # Current Q values
        q_values = self.q_network(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN target calculation
        with torch.no_grad():
            next_q_online = self.q_network(next_states)
            next_q_online = next_q_online.masked_fill(~next_valid_actions, -1e9)
            best_next_actions = next_q_online.argmax(dim=1, keepdim=True)
            
            next_q_target = self.target_network(next_states)
            target_q_values = next_q_target.gather(1, best_next_actions).squeeze(1)
            target_q = rewards + (1 - dones.float()) * self.gamma * target_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def _sample_frontier_action(self, state: np.ndarray, valid_actions: np.ndarray) -> Optional[int]:
        """Sample an action near the information frontier for smarter exploration."""
        if valid_actions is None or not np.any(valid_actions):
            return None
        
        if state is None:
            return None
        
        if state.ndim == 4:
            # Remove batch dimension if present
            state = state.squeeze(0)
        
        if state.ndim != 3 or state.shape[0] < 2:
            return None
        
        height, width = state.shape[1], state.shape[2]
        cell_count = self.cell_count
        if valid_actions.shape[0] < cell_count:
            return None
        
        reveal_valid = valid_actions[:cell_count]
        reveal_matrix = reveal_valid.reshape(height, width)
        hidden_mask = state[1] > 0.5
        flagged_mask = state[2] > 0.5 if state.shape[0] > 2 else np.zeros_like(hidden_mask, dtype=bool)
        revealed_mask = ~(hidden_mask | flagged_mask)
        
        frontier_mask = np.zeros((height, width), dtype=bool)
        
        for row in range(height):
            for col in range(width):
                if not hidden_mask[row, col] or not reveal_matrix[row, col]:
                    continue
                for dr, dc in self._neighbor_offsets:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < height and 0 <= nc < width and revealed_mask[nr, nc]:
                        frontier_mask[row, col] = True
                        break
        
        frontier_indices = np.where(frontier_mask.reshape(-1))[0]
        if frontier_indices.size > 0:
            return int(random.choice(frontier_indices.tolist()))
        
        # Fall back to cells with strongest hint information if available
        if state.shape[0] > 6:
            hint_channel = state[6].reshape(-1)
            valid_indices = np.where(reveal_valid)[0]
            if valid_indices.size > 0:
                best_idx = valid_indices[np.argmax(hint_channel[valid_indices])]
                return int(best_idx)
        
        return None
    
    def save(self, filepath: str):
        """Save model to file."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'board_height': self.board_height,
            'board_width': self.board_width,
            'action_space_size': self.action_space_size
        }, filepath)
    
    def load(self, filepath: str):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Check if board size matches
        saved_height = checkpoint.get('board_height', None)
        saved_width = checkpoint.get('board_width', None)
        saved_action_size = checkpoint.get('action_space_size', None)
        
        if saved_height is not None and saved_width is not None:
            if saved_height != self.board_height or saved_width != self.board_width:
                raise ValueError(
                    f"Model was trained for board size {saved_width}x{saved_height}, "
                    f"but current size is {self.board_width}x{self.board_height}. "
                    f"Please load the model with matching board size."
                )
        
        if saved_action_size is not None and saved_action_size != self.action_space_size:
            raise ValueError(
                f"Model was trained for action space size {saved_action_size}, "
                f"but current size is {self.action_space_size}. "
                f"Please load the model with matching board size."
            )
        
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.q_network.eval()
        self.target_network.eval()



```

```python
"""Training script for DQN agent."""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from src.reinforcement_learning.dqn_agent import DQNAgent
from src.reinforcement_learning.environment import MinesweeperEnvironment


def train_with_visualization(
    episodes: int = 1000,
    difficulty: str = "medium",
    width: int = None,
    height: int = None,
    save_path: str = "models/dqn_model.pth",
    log_interval: int = 100,
    visualization_callback: callable = None,
    thread=None
):
    """
    Train DQN agent on Minesweeper with visualization support.
    
    Args:
        episodes: Number of training episodes
        difficulty: Game difficulty
        width: Board width (uses current game width if None)
        height: Board height (uses current game height if None)
        save_path: Path to save model
        log_interval: Episodes between logging
        visualization_callback: Callback function(agent, episode) for visualization
        thread: Thread object for checking if training should continue
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
    env = MinesweeperEnvironment(difficulty, width, height)
    
    # Adjust hyperparameters based on board size
    # Smaller boards need faster learning, larger boards need more exploration
    board_size_factor = (width * height) / 600.0  # Normalize to default 30x20
    
    # Learning rate scaled by board size (smaller boards learn faster)
    base_lr = 0.001
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
    
    # Batch size tuned per board size
    batch_size = min(96, max(32, int(48 * (1.0 / max(board_size_factor, 0.6)))))
    
    agent = DQNAgent(
        state_channels=env.state_channels,
        action_space_size=env.action_space_size,
        board_height=height,
        board_width=width,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_floor,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    episode_wins = []  # Track wins for each episode
    
    print(f"Starting training for {episodes} episodes...")
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
            
            # Select action
            action = agent.select_action(state, valid_actions)
            
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
        
        # Episode finished: decay epsilon once
        agent.decay_epsilon()
        
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
    log_interval: int = 100
):
    """
    Train DQN agent on Minesweeper (without visualization).
    
    Args:
        episodes: Number of training episodes
        difficulty: Game difficulty
        save_path: Path to save model
        log_interval: Episodes between logging
    """
    train_with_visualization(
        episodes=episodes,
        difficulty=difficulty,
        save_path=save_path,
        log_interval=log_interval,
        visualization_callback=None,
        thread=None
    )


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train DQN agent on Minesweeper")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--difficulty", type=str, default="medium", 
                       choices=["easy", "medium", "hard"], help="Game difficulty")
    parser.add_argument("--save-path", type=str, default="models/dqn_model.pth",
                       help="Path to save model")
    parser.add_argument("--log-interval", type=int, default=100,
                       help="Episodes between logging")
    
    args = parser.parse_args()
    
    train(
        episodes=args.episodes,
        difficulty=args.difficulty,
        save_path=args.save_path,
        log_interval=args.log_interval
    )


if __name__ == "__main__":
    main()

```

```markdown
*** Begin Patch
*** Update File: docs/RL_IMPLEMENTATION_GUIDE.md
@@
-#### Reward System
-
-Kombination aus Fortschritts- und Flaggen-Rewards:
-
-- **Aufdecken**: identisch zum ursprünglichen System (lineare Belohnung für neue Felder, kleiner Fortschrittsbonus, -0.01 für Stillstand)
-- **Flaggen (sofort)**:
-  - +0.2, wenn eine Mine korrekt markiert wird
-  - -0.2, wenn ein sicheres Feld fälschlich markiert wird
-  - 0.0 beim Entfernen einer Flagge (kein Schwingen)
-- **Flaggen (bei Verlust)**:
-  - +0.5 pro korrekt gesetzter Flagge
-  - -0.5 pro falscher Flagge
-- **Spiel verloren**: Basisstrafe skaliert weiterhin mit Brettgröße
-- **Spiel gewonnen**: Große positive Belohnung, abhängig von der Feldanzahl
+#### Reward System
+
+Das Reward-Shaping folgt der Idee aus [sdlee94s Minesweeper DQN](https://sdlee94.github.io/Minesweeper-AI-Reinforcement-Learning/): Progress-Moves werden belohnt, reine „Guess“-Züge erhalten nur minimale Rewards.
+
+- **Aufdecken**:
+  - +0.25 pro sicher aufgedecktem Feld
+  - +0.05 Bonus, wenn eine gesamte leere Region freigelegt wird
+  - +Frontier-Bonus (bis +0.35) für Zellen mit bereits bekannten Nachbarzahlen
+  - -0.2 Malus für reine Guess-Züge ohne Nachbarinformationen
+  - -0.05 für wirkungslose Aktionen
+- **Flaggen (sofort)**:
+  - +0.2 bei korrekt gesetzter Flagge
+  - -0.2 bei falscher Flagge
+  - 0.0 beim Entfernen
+- **Flaggen (bei Verlust)**:
+  - +0.5 pro korrekter Flagge
+  - -0.5 pro falscher Flagge
+- **Spiel verloren**: Moderate Strafe (~ -5 * Board-Skalierung), damit positive Rewards nicht erdrückt werden
+- **Spiel gewonnen**: +10 * Board-Skalierung + kleiner Progress-Boost
@@
-1. **Q-Network**: Haupt-Netzwerk für Q-Value-Schätzung
-2. **Target Network**: Stabilisiert Training durch verzögerte Updates
+1. **Q-Network**: Haupt-Netzwerk für Q-Value-Schätzung
+2. **Double DQN Target Network**: Aktionen werden mit dem Online-Netz gewählt und mit dem Target-Netz bewertet → weniger Overestimation
 3. **Epsilon-Greedy**: Exploration vs. Exploitation
 4. **Experience Replay**: Speichert und nutzt vergangene Erfahrungen
@@
-- Learning Rate: 0.001 (Adam Optimizer)
-- Gamma (Discount): 0.99 (langfristige Belohnung)
-- Epsilon Start: 1.0 (100% Exploration)
-- Epsilon End: 0.01 (1% Exploration)
-- Epsilon Decay: 0.995 (exponentielles Decay)
+- Learning Rate: 0.001 (Adam Optimizer)
+- Gamma (Discount): 0.95 auf Standardbrettern (Priorisierung kurzfristiger Fortschritte)
+- Epsilon Start: 1.0 (100% Exploration)
+- Epsilon End: 0.03–0.1 je nach Schwierigkeitsgrad
+- Epsilon Decay: dynamisch, so dass ~70% der Episoden für Exploration genutzt werden
@@
-- Target Update: Alle 100 Steps
+- Target Update: Alle 100 Steps
@@
-- **TD Target**: `target = reward + gamma * max(Q(s', a') | a' gültig)`
-- **Loss**: Mean Squared Error zwischen Current und Target
+- **TD Target** (Double DQN): Aktion wird mit Online-Netz gewählt, Q-Wert mit Target-Netz bewertet
+- **Loss**: Smooth L1 (Huber) zwischen Current und Target
*** End Patch
```

```markdown
# Reinforcement Learning Implementierungs-Guide

Dieses Dokument erklärt detailliert, wie das Reinforcement Learning System für Minesweeper implementiert ist.

## Übersicht

Das RL-System verwendet **Deep Q-Network (DQN)** mit folgenden Komponenten:
- **Environment**: Wrapper für das Minesweeper-Spiel
- **DQN Network**: Convolutional Neural Network für Q-Value-Schätzung
- **DQN Agent**: Agent mit Experience Replay und Target Network
- **Training**: Episoden-basiertes Training mit Logging

---

## 1. Environment (`src/reinforcement_learning/environment.py`)

### Zweck
Wrappt das Minesweeper-Spiel in ein Gymnasium-ähnliches Interface für RL-Algorithmen.

### Implementierung

#### State Representation (Zustandsrepräsentation)

Der State besitzt **7 Kanäle** und liefert vollständigen Kontext über jede Zelle und ihre Nachbarschaft:

1. **Basis-Encoding** (`state[0]`): alter kombinierter Wert (-0.9 hidden, -0.5 Flag, -1 Mine, 0‑1 Zahl) inkl. Nachbarschaftsbonus/-strafe  
2. **Hidden-Maske** (`state[1]`): 1 für verdeckte Zelle, sonst 0  
3. **Flag-Maske** (`state[2]`): 1 für gesetzte Flagge, sonst 0  
4. **Aufgedeckte Zahl** (`state[3]`): normierte Zahl (0‑1) bzw. -1 bei Mine  
5. **Verdeckte Nachbarn** (`state[4]`): Anteil verdeckter Nachbarn (0‑1)  
6. **Flag-Nachbarn** (`state[5]`): Anteil geflaggter Nachbarn (0‑1)  
7. **Hinweis-Summe** (`state[6]`): normierte Summe aller bekannten Nachbarzahlen (0‑1)

Damit kennt der Agent sowohl den lokalen Status als auch die unmittelbare „Frontier“ und vorhandene Hinweise.

#### Action Space (Reveal + Flag)

- **Gesamtgröße**: `2 × width × height`  
- **Indexierung**:
  - `0 … (N-1)`: Zelle aufdecken (`row = idx // width`, `col = idx % width`)
  - `N … (2N-1)`: Flagge auf derselben Zelle setzen/entfernen (`idx - N`)
- **Validierungslogik**:
  - Aufdecken nur für versteckte und nicht geflaggte Felder
  - Flaggen für versteckte oder bereits geflaggte Felder (Toggle)

#### Reward System

Orientiert sich an den Progress/Gues-Rewards aus [sdlee94s Minesweeper DQN](https://sdlee94.github.io/Minesweeper-AI-Reinforcement-Learning/):

- **Aufdecken**:
  - +0.25 pro sicher aufgedecktem Feld
  - +0.05 Bonus für Kettenreaktionen (mehrere Felder in einem Schritt)
  - +Frontier-Bonus bis +0.35, wenn mindestens ein Nachbar bereits bekannt ist
  - -0.2 Malus für reine Guess-Züge ohne Nachbarinformationen
  - -0.05 für wirkungslose Aktionen
- **Flaggen (sofort)**:
  - +0.2 bei korrekt gesetzter Flagge
  - -0.2 bei falscher Flagge
  - 0.0 beim Entfernen (kein Schwingen)
- **Flaggen (bei Verlust)**:
  - +0.5 pro korrekter Flagge
  - -0.5 pro falscher Flagge
- **Spiel verloren**: Moderate Strafe (~ -5 × Board-Skalierung), damit positive Rewards nicht erdrückt werden
- **Spiel gewonnen**: +10 × Board-Skalierung plus kleiner Progress-Bonus

#### Valid Actions Masking

Das Environment bietet zwei Methoden für gültige Aktionen:

1. **`get_valid_actions()`**: Boolean-Array (True = gültig)
2. **`get_action_mask()`**: Mask für Q-Values (-inf für ungültig, 0.0 für gültig)

**Warum Masking?**
- Verhindert, dass der Agent bereits aufgedeckte oder flagge Zellen auswählt
- Reduziert den Aktionsraum effektiv
- Verbessert Trainingseffizienz

---

## 2. DQN Network (`src/reinforcement_learning/network.py`)

### Architektur

**Convolutional Neural Network** für räumliche Feature-Extraktion:

```
Input: (batch, 7, 20, 30)  # 7-kanalige State-Repräsentation

Conv Layers:
- Conv2d(7 → 32, kernel=3, padding=1)  # Erste Feature-Ebene
- Conv2d(32 → 64, kernel=3, padding=1)  # Zweite Feature-Ebene
- Conv2d(64 → 64, kernel=3, padding=1) # Dritte Feature-Ebene

Flatten: 64 × 20 × 30 = 38,400 features

Fully Connected:
- Linear(38,400 → 512)
- Linear(512 → 256)
- Linear(256 → 600)  # Q-values für alle Aktionen

Output: (batch, 600)  # Q-value für jede Aktion
```

**Warum CNN?**
- **Räumliche Struktur**: Minesweeper hat räumliche Beziehungen (Nachbarschaft)
- **Feature-Learning**: CNN lernt automatisch relevante Muster
- **Effizienz**: Weniger Parameter als Fully-Connected für große Inputs

**Warum diese Architektur?**
- **3 Conv Layers**: Ausreichend für Feature-Extraktion
- **Padding=1**: Behält Spielfeldgröße bei
- **Progressive Channels**: 32 → 64 → 64 für mehr Features
- **FC Layers**: 512 → 256 → 600 für Entscheidungsfindung

---

## 3. DQN Agent (`src/reinforcement_learning/dqn_agent.py`)

### Komponenten

#### 3.1 ReplayBuffer

**Zweck**: Speichert Erfahrungen (State, Action, Reward, Next State, Done, Next-Action-Maske) für Experience Replay.

**Implementierung:**
- `deque` mit `maxlen` für automatische Größenbegrenzung
- Speichert zusätzlich die zulässigen Aktionen des Folgezustands zur Maskierung der Ziel-Q-Werte
- Zufälliges Sampling für Batch-Training
- Konvertiert NumPy-Arrays zu PyTorch-Tensoren

**Warum Experience Replay?**
- **Stabilisierung**: Bricht Korrelationen zwischen aufeinanderfolgenden Erfahrungen
- **Efficiency**: Nutzt Erfahrungen mehrfach
- **Diversität**: Batch enthält verschiedene Erfahrungen

#### 3.2 DQN Agent

**Zweck**: Implementiert DQN-Algorithmus mit allen notwendigen Komponenten.

**Hauptkomponenten:**

1. **Q-Network**: Haupt-Netzwerk für Q-Value-Schätzung
2. **Double Target Network**: Double DQN vermeidet Q-Value-Overestimation
3. **Epsilon-Greedy**: Exploration vs. Exploitation
4. **Experience Replay**: Speichert und nutzt vergangene Erfahrungen

**Hyperparameter:**

```python
- Learning Rate: 0.001 (Adam Optimizer)
- Gamma (Discount): 0.95 auf Standardbrettern (mehr Fokus auf Fortschritt)
- Epsilon Start: 1.0 (100% Exploration)
- Epsilon End: 0.03–0.10 je nach Schwierigkeitsgrad
- Epsilon Decay: dynamisch berechnet (~70 % der Episoden explorativ)
- Buffer Size: 10,000 Erfahrungen
- Batch Size: 32–96 (abhängig von Brettgröße)
- Target Update: Alle 100 Steps
```

**Gezielte Epsilon-Greedy Strategie:**

```python
if random() < epsilon:
    action = sample_frontier(valid_actions, state)  # Bevorzugt Zellen an bekannten Grenzen
else:
    action = argmax(Q(state, valid_only=True))
```

- **Frontier-Sampling**: Zufallsaktionen wählen bevorzugt verdeckte Felder, die an bereits aufgedeckte Zahlen grenzen  
- **Fallback auf Hinweise**: Wenn keine Frontier existiert, wird ein Feld mit dem größten Hinweis-Signal gewählt  
- **Episodenweiser Decay**: `epsilon` wird nach jeder Episode reduziert, nicht nach jedem Schritt

**Training Process (train_step):**

1. **Batch Sampling**: Zufällige Erfahrungen inkl. Aktionsmasken aus Replay Buffer
2. **Current Q-Values**: Q-Network schätzt Q(s, a)
3. **Target Q-Values**: Double DQN – Aktion via Online-Netz wählen, Q-Wert via Target-Netz bewerten
4. **TD Target**: `target = reward + gamma * Q_target(s', argmax_a Q_online(s'))`
5. **Loss**: Smooth L1 (Huber) zwischen Current und Target
6. **Backpropagation**: Gradienten werden berechnet und angewendet
7. **Gradient Clipping**: Verhindert Exploding Gradients (max norm = 1.0)
8. **Epsilon Decay**: Erfolgt episodenweise über `agent.decay_epsilon()`
9. **Target Update**: Alle N Steps wird Target Network aktualisiert

**Warum Target Network?**
- **Stabilität**: Verhindert instabile Q-Value-Updates
- **Convergence**: Hilft beim Konvergieren des Trainings
- **Delayed Updates**: Target Network wird nur periodisch aktualisiert

**Warum Gradient Clipping?**
- **Stabilität**: Verhindert sehr große Gradienten
- **Training**: Stabilisiert den Lernprozess

---

## 4. Training (`src/reinforcement_learning/trainer.py`)

### Training Loop

```python
for episode in range(episodes):
    state = env.reset()
    done = False
    
    while not done:
        # 1. Agent wählt Aktion
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions)
        
        # 2. Environment führt Aktion aus
        next_state, reward, done, info = env.step(action)
        next_valid_actions = env.get_valid_actions()
        
        # 3. Agent speichert Erfahrung
        agent.remember(state, action, reward, next_state, done, next_valid_actions)
        
        # 4. Agent trainiert (wenn genug Erfahrungen)
        agent.train_step()
        
        # 5. Update state
        state = next_state
    
    # 6. Exploration langsam reduzieren
    agent.decay_epsilon()
    
    # Logging alle N Episoden
    if episode % log_interval == 0:
        log_statistics()
```

### Logging

Alle N Episoden werden folgende Statistiken geloggt:
- **Average Reward**: Durchschnittliche Belohnung
- **Average Length**: Durchschnittliche Episodenlänge
- **Win Rate**: Prozentsatz gewonnener Spiele
- **Epsilon**: Aktueller Exploration-Faktor

### Model Saving

- Periodisches Speichern (alle N × log_interval Episoden)
- Finales Speichern nach Training
- Enthält: Q-Network, Target Network, Optimizer State, Epsilon

---

## 5. Design-Entscheidungen und Optimierungen

### Warum DQN?

1. **Discrete Action Space**: Minesweeper hat diskreten Aktionsraum (600 Aktionen)
2. **Value-Based**: Q-Learning passt gut für deterministische Umgebungen
3. **Bewährt**: DQN ist etabliert und gut verstanden

### Warum diese State-Encoding?

1. **Einfachheit**: Ein Kanal, einfache CNN-Architektur
2. **Normalisierung**: Werte zwischen -1 und 1 für besseres Training
3. **Informativ**: Unterscheidet alle wichtigen Zellzustände

### Warum diese Reward-Struktur?

1. **Sparse + Shaped**: Hauptrewards für wichtige Ereignisse, Shaped für Fortschritt
2. **Balance**: Nicht zu viele kleine Rewards (verhindert Overfitting)
3. **Klare Signale**: Gewinn/Verlust sind deutlich signalisiert

### Mögliche Verbesserungen

1. **Double DQN**: Reduziert Overestimation von Q-Values
2. **Dueling DQN**: Trennt State-Value und Advantage
3. **Prioritized Replay**: Wichtige Erfahrungen werden öfter gesampelt
4. **Multi-Step Learning**: N-Step Returns statt 1-Step
5. **State-Encoding**: Mehr Features (z.B. Flaggen-Informationen getrennt)

---

## 6. Verwendung

### Training starten:

```bash
python -m src.reinforcement_learning.trainer \
    --episodes 1000 \
    --difficulty medium \
    --save-path models/dqn_model.pth
```

### Modell laden:

```python
from src.reinforcement_learning.dqn_agent import DQNAgent
from src.reinforcement_learning.environment import MinesweeperEnvironment

env = MinesweeperEnvironment("medium")
agent = DQNAgent(state_channels=1, action_space_size=600)
agent.load("models/dqn_model.pth")

# Agent spielen lassen
state = env.reset()
done = False
while not done:
    valid_actions = env.get_valid_actions()
    action = agent.select_action(state, valid_actions)
    state, reward, done, info = env.step(action)
```

---

## 7. Zusammenfassung

Das RL-System implementiert:

✅ **Environment Wrapper**: Gymnasium-ähnliches Interface
✅ **State Encoding**: Normalisierte 2D-Repräsentation
✅ **DQN Network**: CNN für räumliche Features
✅ **Experience Replay**: Stabilisiert Training
✅ **Target Network**: Verhindert instabile Updates
✅ **Epsilon-Greedy**: Exploration/Exploitation Balance
✅ **Reward Shaping**: Sparse + Shaped Rewards
✅ **Action Masking**: Verhindert ungültige Aktionen

Das System ist vollständig funktionsfähig und bereit für Training!


```

```markdown
# Reinforcement Learning Implementierungs-Guide

Dieses Dokument erklärt detailliert, wie das Reinforcement Learning System für Minesweeper implementiert ist.

## Übersicht

Das RL-System verwendet **Deep Q-Network (DQN)** mit folgenden Komponenten:
- **Environment**: Wrapper für das Minesweeper-Spiel
- **DQN Network**: Convolutional Neural Network für Q-Value-Schätzung
- **DQN Agent**: Agent mit Experience Replay und Target Network
- **Training**: Episoden-basiertes Training mit Logging

---

## 1. Environment (`src/reinforcement_learning/environment.py`)

### Zweck
Wrappt das Minesweeper-Spiel in ein Gymnasium-ähnliches Interface für RL-Algorithmen.

### Implementierung

#### State Representation (Zustandsrepräsentation)

Der State besitzt **7 Kanäle** und liefert vollständigen Kontext über jede Zelle und ihre Nachbarschaft:

1. **Basis-Encoding** (`state[0]`): alter kombinierter Wert (-0.9 hidden, -0.5 Flag, -1 Mine, 0‑1 Zahl) inkl. Nachbarschaftsbonus/-strafe  
2. **Hidden-Maske** (`state[1]`): 1 für verdeckte Zelle, sonst 0  
3. **Flag-Maske** (`state[2]`): 1 für gesetzte Flagge, sonst 0  
4. **Aufgedeckte Zahl** (`state[3]`): normierte Zahl (0‑1) bzw. -1 bei Mine  
5. **Verdeckte Nachbarn** (`state[4]`): Anteil verdeckter Nachbarn (0‑1)  
6. **Flag-Nachbarn** (`state[5]`): Anteil geflaggter Nachbarn (0‑1)  
7. **Hinweis-Summe** (`state[6]`): normierte Summe aller bekannten Nachbarzahlen (0‑1)

Damit kennt der Agent sowohl den lokalen Status als auch die unmittelbare „Frontier“ und vorhandene Hinweise.

#### Action Space (Reveal + Flag)

- **Gesamtgröße**: `2 × width × height`  
- **Indexierung**:
  - `0 … (N-1)`: Zelle aufdecken (`row = idx // width`, `col = idx % width`)
  - `N … (2N-1)`: Flagge auf derselben Zelle setzen/entfernen (`idx - N`)
- **Validierungslogik**:
  - Aufdecken nur für versteckte und nicht geflaggte Felder
  - Flaggen für versteckte oder bereits geflaggte Felder (Toggle)

#### Reward System

Orientiert sich an den Progress/Gues-Rewards aus [sdlee94s Minesweeper DQN](https://sdlee94.github.io/Minesweeper-AI-Reinforcement-Learning/):

- **Aufdecken**:
  - +0.25 pro sicher aufgedecktem Feld
  - +0.05 Bonus für Kettenreaktionen (mehrere Felder in einem Schritt)
  - +Frontier-Bonus bis +0.35, wenn mindestens ein Nachbar bereits bekannt ist
  - -0.2 Malus für reine Guess-Züge ohne Nachbarinformationen
  - -0.05 für wirkungslose Aktionen
- **Flaggen (sofort)**:
  - +0.2 bei korrekt gesetzter Flagge
  - -0.2 bei falscher Flagge
  - 0.0 beim Entfernen (kein Schwingen)
- **Flaggen (bei Verlust)**:
  - +0.5 pro korrekter Flagge
  - -0.5 pro falscher Flagge
- **Spiel verloren**: Moderate Strafe (~ -5 × Board-Skalierung), damit positive Rewards nicht erdrückt werden
- **Spiel gewonnen**: +10 × Board-Skalierung plus kleiner Progress-Bonus

#### Valid Actions Masking

Das Environment bietet zwei Methoden für gültige Aktionen:

1. **`get_valid_actions()`**: Boolean-Array (True = gültig)
2. **`get_action_mask()`**: Mask für Q-Values (-inf für ungültig, 0.0 für gültig)

**Warum Masking?**
- Verhindert, dass der Agent bereits aufgedeckte oder flagge Zellen auswählt
- Reduziert den Aktionsraum effektiv
- Verbessert Trainingseffizienz

---

## 2. DQN Network (`src/reinforcement_learning/network.py`)

### Architektur

**Convolutional Neural Network** für räumliche Feature-Extraktion:

```
Input: (batch, 7, 20, 30)  # 7-kanalige State-Repräsentation

Conv Layers:
- Conv2d(7 → 32, kernel=3, padding=1)  # Erste Feature-Ebene
- Conv2d(32 → 64, kernel=3, padding=1)  # Zweite Feature-Ebene
- Conv2d(64 → 64, kernel=3, padding=1) # Dritte Feature-Ebene

Flatten: 64 × 20 × 30 = 38,400 features

Fully Connected:
- Linear(38,400 → 512)
- Linear(512 → 256)
- Linear(256 → 600)  # Q-values für alle Aktionen

Output: (batch, 600)  # Q-value für jede Aktion
```

**Warum CNN?**
- **Räumliche Struktur**: Minesweeper hat räumliche Beziehungen (Nachbarschaft)
- **Feature-Learning**: CNN lernt automatisch relevante Muster
- **Effizienz**: Weniger Parameter als Fully-Connected für große Inputs

**Warum diese Architektur?**
- **3 Conv Layers**: Ausreichend für Feature-Extraktion
- **Padding=1**: Behält Spielfeldgröße bei
- **Progressive Channels**: 32 → 64 → 64 für mehr Features
- **FC Layers**: 512 → 256 → 600 für Entscheidungsfindung

---

## 3. DQN Agent (`src/reinforcement_learning/dqn_agent.py`)

### Komponenten

#### 3.1 ReplayBuffer

**Zweck**: Speichert Erfahrungen (State, Action, Reward, Next State, Done, Next-Action-Maske) für Experience Replay.

**Implementierung:**
- `deque` mit `maxlen` für automatische Größenbegrenzung
- Speichert zusätzlich die zulässigen Aktionen des Folgezustands zur Maskierung der Ziel-Q-Werte
- Zufälliges Sampling für Batch-Training
- Konvertiert NumPy-Arrays zu PyTorch-Tensoren

**Warum Experience Replay?**
- **Stabilisierung**: Bricht Korrelationen zwischen aufeinanderfolgenden Erfahrungen
- **Efficiency**: Nutzt Erfahrungen mehrfach
- **Diversität**: Batch enthält verschiedene Erfahrungen

#### 3.2 DQN Agent

**Zweck**: Implementiert DQN-Algorithmus mit allen notwendigen Komponenten.

**Hauptkomponenten:**

1. **Q-Network**: Haupt-Netzwerk für Q-Value-Schätzung
2. **Double Target Network**: Double DQN vermeidet Q-Value-Overestimation
3. **Epsilon-Greedy**: Exploration vs. Exploitation
4. **Experience Replay**: Speichert und nutzt vergangene Erfahrungen

**Hyperparameter:**

```python
- Learning Rate: 0.001 (Adam Optimizer)
- Gamma (Discount): 0.95 auf Standardbrettern (mehr Fokus auf Fortschritt)
- Epsilon Start: 1.0 (100% Exploration)
- Epsilon End: 0.03–0.10 je nach Schwierigkeitsgrad
- Epsilon Decay: dynamisch berechnet (~70 % der Episoden explorativ)
- Buffer Size: 10,000 Erfahrungen
- Batch Size: 32–96 (abhängig von Brettgröße)
- Target Update: Alle 100 Steps
```

**Gezielte Epsilon-Greedy Strategie:**

```python
if random() < epsilon:
    action = sample_frontier(valid_actions, state)  # Bevorzugt Zellen an bekannten Grenzen
else:
    action = argmax(Q(state, valid_only=True))
```

- **Frontier-Sampling**: Zufallsaktionen wählen bevorzugt verdeckte Felder, die an bereits aufgedeckte Zahlen grenzen  
- **Fallback auf Hinweise**: Wenn keine Frontier existiert, wird ein Feld mit dem größten Hinweis-Signal gewählt  
- **Episodenweiser Decay**: `epsilon` wird nach jeder Episode reduziert, nicht nach jedem Schritt

**Training Process (train_step):**

1. **Batch Sampling**: Zufällige Erfahrungen inkl. Aktionsmasken aus Replay Buffer
2. **Current Q-Values**: Q-Network schätzt Q(s, a)
3. **Target Q-Values**: Double DQN – Aktion via Online-Netz wählen, Q-Wert via Target-Netz bewerten
4. **TD Target**: `target = reward + gamma * Q_target(s', argmax_a Q_online(s'))`
5. **Loss**: Smooth L1 (Huber) zwischen Current und Target
6. **Backpropagation**: Gradienten werden berechnet und angewendet
7. **Gradient Clipping**: Verhindert Exploding Gradients (max norm = 1.0)
8. **Epsilon Decay**: Erfolgt episodenweise über `agent.decay_epsilon()`
9. **Target Update**: Alle N Steps wird Target Network aktualisiert

**Warum Target Network?**
- **Stabilität**: Verhindert instabile Q-Value-Updates
- **Convergence**: Hilft beim Konvergieren des Trainings
- **Delayed Updates**: Target Network wird nur periodisch aktualisiert

**Warum Gradient Clipping?**
- **Stabilität**: Verhindert sehr große Gradienten
- **Training**: Stabilisiert den Lernprozess

---

## 4. Training (`src/reinforcement_learning/trainer.py`)

### Training Loop

```python
for episode in range(episodes):
    state = env.reset()
    done = False
    
    while not done:
        # 1. Agent wählt Aktion
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions)
        
        # 2. Environment führt Aktion aus
        next_state, reward, done, info = env.step(action)
        next_valid_actions = env.get_valid_actions()
        
        # 3. Agent speichert Erfahrung
        agent.remember(state, action, reward, next_state, done, next_valid_actions)
        
        # 4. Agent trainiert (wenn genug Erfahrungen)
        agent.train_step()
        
        # 5. Update state
        state = next_state
    
    # 6. Exploration langsam reduzieren
    agent.decay_epsilon()
    
    # Logging alle N Episoden
    if episode % log_interval == 0:
        log_statistics()
```

### Logging

Alle N Episoden werden folgende Statistiken geloggt:
- **Average Reward**: Durchschnittliche Belohnung
- **Average Length**: Durchschnittliche Episodenlänge
- **Win Rate**: Prozentsatz gewonnener Spiele
- **Epsilon**: Aktueller Exploration-Faktor

### Model Saving

- Periodisches Speichern (alle N × log_interval Episoden)
- Finales Speichern nach Training
- Enthält: Q-Network, Target Network, Optimizer State, Epsilon

---

## 5. Design-Entscheidungen und Optimierungen

### Warum DQN?

1. **Discrete Action Space**: Minesweeper hat diskreten Aktionsraum (600 Aktionen)
2. **Value-Based**: Q-Learning passt gut für deterministische Umgebungen
3. **Bewährt**: DQN ist etabliert und gut verstanden

### Warum diese State-Encoding?

1. **Einfachheit**: Ein Kanal, einfache CNN-Architektur
2. **Normalisierung**: Werte zwischen -1 und 1 für besseres Training
3. **Informativ**: Unterscheidet alle wichtigen Zellzustände

### Warum diese Reward-Struktur?

1. **Sparse + Shaped**: Hauptrewards für wichtige Ereignisse, Shaped für Fortschritt
2. **Balance**: Nicht zu viele kleine Rewards (verhindert Overfitting)
3. **Klare Signale**: Gewinn/Verlust sind deutlich signalisiert

### Mögliche Verbesserungen

1. **Dueling DQN**: Trennt State-Value und Advantage
2. **Prioritized Replay**: Wichtige Erfahrungen werden öfter gesampelt
3. **Multi-Step Learning**: N-Step Returns statt 1-Step
4. **State-Encoding**: Mehr Features (z.B. Wahrscheinlichkeiten je Feld)
5. **Curriculum**: Transfer Learning zwischen Schwierigkeitsgraden

---

## 6. Verwendung

### Training starten:

```bash
python -m src.reinforcement_learning.trainer \
    --episodes 1000 \
    --difficulty medium \
    --save-path models/dqn_model.pth
```

### Modell laden:

```python
from src.reinforcement_learning.dqn_agent import DQNAgent
from src.reinforcement_learning.environment import MinesweeperEnvironment

env = MinesweeperEnvironment("medium")
agent = DQNAgent(state_channels=1, action_space_size=600)
agent.load("models/dqn_model.pth")

# Agent spielen lassen
state = env.reset()
done = False
while not done:
    valid_actions = env.get_valid_actions()
    action = agent.select_action(state, valid_actions)
    state, reward, done, info = env.step(action)
```

---

## 7. Zusammenfassung

Das RL-System implementiert:

✅ **Environment Wrapper**: Gymnasium-ähnliches Interface
✅ **State Encoding**: Normalisierte 2D-Repräsentation
✅ **DQN Network**: CNN für räumliche Features
✅ **Experience Replay**: Stabilisiert Training
✅ **Target Network**: Verhindert instabile Updates
✅ **Epsilon-Greedy**: Exploration/Exploitation Balance
✅ **Reward Shaping**: Sparse + Shaped Rewards
✅ **Action Masking**: Verhindert ungültige Aktionen

Das System ist vollständig funktionsfähig und bereit für Training!


```

## Zusammenfassung
- Belohnungslogik stärker auf Progress-vs.-Guess ausgelegt (Frontier-Bonus, moderate Verlust-/Gewinnwerte) und damit direkt an den Ansatz von [sdlee94](https://sdlee94.github.io/Minesweeper-AI-Reinforcement-Learning/) angelehnt; dadurch erhält der Agent klarere Lernsignale bei logischen Zügen. `src/reinforcement_learning/environment.py`
- Der DQN-Agent nutzt jetzt Double-DQN-Targets, Huber-Loss und bevorzugt beim Explorieren echte Aufdeck-Aktionen; der Trainer berechnet dazu eine episodenabhängige Epsilon-Decay-Kurve sowie gamma-Werte pro Brettgröße, was das Lernen stabilisiert. `src/reinforcement_learning/dqn_agent.py`, `src/reinforcement_learning/trainer.py`
- Die Änderungen sind im Leitfaden dokumentiert (`docs/RL_IMPLEMENTATION_GUIDE.md`), sodass Reward-Shaping und neue Hyperparameter auch konzeptionell nachvollziehbar bleiben.

## Code-Auszüge
```52:210:src/reinforcement_learning/environment.py
        move_context = self._get_move_context(row, col)
        
        if action_type == "reveal":
            valid = self.game.reveal_cell(row, col)
            reward += self._calculate_reward(
                prev_revealed,
                valid,
                action_type=action_type,
                move_context=move_context
            )
# ...
    def _calculate_reward(
        self,
        prev_revealed: int,
        valid_action: bool,
        action_type: str,
        move_context: Optional[dict] = None
    ) -> float:
        if action_type != "reveal":
            return 0.0
        if not valid_action:
            return -0.4
        board_scale = max(0.6, (self.width * self.height) / 600.0)
        cells_revealed = self.game.revealed_count - prev_revealed
        progress_ratio = self.game.revealed_count / max(1, self.game.total_safe_cells)
        is_guess = move_context.get("adjacent_revealed", 0) == 0 if move_context else True
        frontier_bonus = 0.0
        # …
```

```114:217:src/reinforcement_learning/dqn_agent.py
        # Optimizer & loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
# ...
        if random.random() < self.epsilon:
            if valid_actions is not None:
                frontier_action = self._sample_frontier_action(state, valid_actions)
                if frontier_action is not None:
                    return frontier_action
                
                # Prefer reveal actions during random exploration
                reveal_mask = valid_actions[:self.cell_count]
                reveal_indices = np.where(reveal_mask)[0]
                if reveal_indices.size > 0:
                    return int(random.choice(reveal_indices.tolist()))
# ...
        with torch.no_grad():
            next_q_online = self.q_network(next_states)
            next_q_online = next_q_online.masked_fill(~next_valid_actions, -1e9)
            best_next_actions = next_q_online.argmax(dim=1, keepdim=True)
            
            next_q_target = self.target_network(next_states)
            target_q_values = next_q_target.gather(1, best_next_actions).squeeze(1)
            target_q = rewards + (1 - dones.float()) * self.gamma * target_q_values
```

```49:84:src/reinforcement_learning/trainer.py
    board_size_factor = (width * height) / 600.0  # Normalize to default 30x20
    
    # Learning rate scaled by board size (smaller boards learn faster)
    base_lr = 0.001
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
    
    # Batch size tuned per board size
    batch_size = min(96, max(32, int(48 * (1.0 / max(board_size_factor, 0.6)))))
    
    agent = DQNAgent(
        state_channels=env.state_channels,
        action_space_size=env.action_space_size,
        board_height=height,
        board_width=width,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_floor,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size
    )
```

## Tests
- `python -m pytest tests/reinforcement_learning`

Falls du weitere Messungen (z.B. längeres Training) brauchst, gerne melden.

```python
        move_context = self._get_move_context(row, col)
        
        if action_type == "reveal":
            valid = self.game.reveal_cell(row, col)
            reward += self._calculate_reward(
                prev_revealed,
                valid,
                action_type=action_type,
                move_context=move_context
            )
# ...
    def _calculate_reward(
        self,
        prev_revealed: int,
        valid_action: bool,
        action_type: str,
        move_context: Optional[dict] = None
    ) -> float:
        if action_type != "reveal":
            return 0.0
        if not valid_action:
            return -0.4
        board_scale = max(0.6, (self.width * self.height) / 600.0)
        cells_revealed = self.game.revealed_count - prev_revealed
        progress_ratio = self.game.revealed_count / max(1, self.game.total_safe_cells)
        is_guess = move_context.get("adjacent_revealed", 0) == 0 if move_context else True
        frontier_bonus = 0.0
        # …
```

```python
        # Optimizer & loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
# ...
        if random.random() < self.epsilon:
            if valid_actions is not None:
                frontier_action = self._sample_frontier_action(state, valid_actions)
                if frontier_action is not None:
                    return frontier_action
                
                # Prefer reveal actions during random exploration
                reveal_mask = valid_actions[:self.cell_count]
                reveal_indices = np.where(reveal_mask)[0]
                if reveal_indices.size > 0:
                    return int(random.choice(reveal_indices.tolist()))
# ...
        with torch.no_grad():
            next_q_online = self.q_network(next_states)
            next_q_online = next_q_online.masked_fill(~next_valid_actions, -1e9)
            best_next_actions = next_q_online.argmax(dim=1, keepdim=True)
            
            next_q_target = self.target_network(next_states)
            target_q_values = next_q_target.gather(1, best_next_actions).squeeze(1)
            target_q = rewards + (1 - dones.float()) * self.gamma * target_q_values
```

```python
    board_size_factor = (width * height) / 600.0  # Normalize to default 30x20
    
    # Learning rate scaled by board size (smaller boards learn faster)
    base_lr = 0.001
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
    
    # Batch size tuned per board size
    batch_size = min(96, max(32, int(48 * (1.0 / max(board_size_factor, 0.6)))))
    
    agent = DQNAgent(
        state_channels=env.state_channels,
        action_space_size=env.action_space_size,
        board_height=height,
        board_width=width,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_floor,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size
    )
```

---

**User**

Das RL ist immer noch nicht gut kannst du mit https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning  noch mehr herausfinden und verbessern?

---

**Cursor**

```python
"""Neural network for DQN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.reinforcement_learning.environment import STATE_CHANNELS


class DQNNetwork(nn.Module):
    """Deep Q-Network for Minesweeper."""
    
    def __init__(self, input_channels: int = STATE_CHANNELS, num_actions: int = 600, board_height: int = 20, board_width: int = 30):
        """
        Initialize DQN network.
        
        Args:
            input_channels: Number of input channels (features per cell)
            num_actions: Number of possible actions (all cells)
            board_height: Height of the board
            board_width: Width of the board
        """
        super(DQNNetwork, self).__init__()
        self.num_actions = num_actions
        self.board_height = board_height
        self.board_width = board_width
        
        # 4 Convolutional Blöcke à 128 Filter (inspiriert von sdlee94)
        conv_layers = []
        in_channels = input_channels
        for _ in range(4):
            conv_layers.extend([
                nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ])
            in_channels = 128
        self.conv_stack = nn.Sequential(*conv_layers)
        
        # Adaptive Pooling auf feste Größe (8x8) für unterschiedliche Boards
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        flattened_size = 128 * 8 * 8
        
        # Zwei vollverbundene Schichten mit je 512 Neuronen + Dropout
        self.fc1 = nn.Linear(flattened_size, 512)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.25)
        self.fc_out = nn.Linear(512, num_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass durch das Netzwerk.
        """
        x = self.conv_stack(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc_out(x)
        return x



```

```python
"""RL Environment wrapper for Minesweeper."""

import numpy as np
from typing import Tuple, Optional
from src.minesweeper.game import Game
from src.utils.constants import (
    BOARD_WIDTH, BOARD_HEIGHT,
    CELL_HIDDEN, CELL_REVEALED, CELL_FLAGGED, CELL_MINE
)

STATE_CHANNELS = 7
NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]

FLAG_SET_REWARD = 0.2
FLAG_SET_PENALTY = 0.2
FLAG_FINAL_CORRECT_BONUS = 0.5
FLAG_FINAL_INCORRECT_PENALTY = 0.5


class MinesweeperEnvironment:
    """Environment wrapper for reinforcement learning."""
    
    def __init__(
        self,
        difficulty: str = "medium",
        width: int = BOARD_WIDTH,
        height: int = BOARD_HEIGHT,
        use_flag_actions: bool = True
    ):
        """
        Initialize the environment.
        
        Args:
            difficulty: Game difficulty level
            width: Board width
            height: Board height
        """
        self.difficulty = difficulty
        self.width = width
        self.height = height
        self.game = Game(difficulty, width, height)
        self.cell_count = width * height
        self.use_flag_actions = use_flag_actions
        self.action_space_size = (
            self.cell_count if not use_flag_actions else self.cell_count * 2
        )  # Reveal (+ optional Flag) per cell
        self.state_channels = STATE_CHANNELS
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment and return initial state.
        
        Returns:
            Initial state observation
        """
        self.game.new_game(self.difficulty, self.width, self.height)
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action index (0 to action_space_size - 1)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Convert action to row, col
        cell_index = action % self.cell_count if self.use_flag_actions and action >= self.cell_count else action
        row = cell_index // self.width
        col = cell_index % self.width
        action_type = "flag" if (self.use_flag_actions and action >= self.cell_count) else "reveal"
        
        # Get previous state
        prev_revealed = self.game.revealed_count
        reward = 0.0
        valid = False
        
        move_context = self._get_move_context(row, col)
        
        if action_type == "reveal":
            valid = self.game.reveal_cell(row, col)
            reward += self._calculate_reward(
                prev_revealed,
                valid,
                action_type=action_type,
                move_context=move_context
            )
        else:
            valid, flag_reward = self._handle_flag_action(row, col)
            reward += flag_reward
        
        # Check if done
        done = self.game.is_game_over()
        
        # Get next state
        next_state = self._get_state()
        
        # Info
        info = {
            "won": self.game.is_won(),
            "lost": self.game.is_lost(),
            "valid_action": valid
        }
        
        # Additional rewards/penalties when the game ends after this move
        if done and self.use_flag_actions and self.game.is_lost():
            reward += self._calculate_flag_end_bonus()
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation with better features.
        
        Returns:
            State array of shape (STATE_CHANNELS, height, width)
        """
        state = np.zeros((STATE_CHANNELS, self.height, self.width), dtype=np.float32)
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                adjacent_revealed = 0
                adjacent_hint_sum = 0
                hidden_neighbors = 0
                flagged_neighbors = 0
                
                for dr, dc in NEIGHBOR_OFFSETS:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        neighbor = self.game.board.get_cell(nr, nc)
                        if neighbor.is_hidden():
                            hidden_neighbors += 1
                        elif neighbor.is_flagged():
                            flagged_neighbors += 1
                        elif neighbor.is_revealed():
                            adjacent_revealed += 1
                            if not neighbor.is_mine:
                                adjacent_hint_sum += neighbor.adjacent_mines
                
                if cell.is_revealed():
                    if cell.is_mine:
                        state[0, row, col] = -1.0
                        state[3, row, col] = -1.0
                    else:
                        value = cell.adjacent_mines / 8.0
                        state[0, row, col] = value
                        state[3, row, col] = value
                elif cell.is_flagged():
                    state[0, row, col] = -0.5
                    state[2, row, col] = 1.0
                else:
                    neighbor_bonus = min(0.4, adjacent_revealed * 0.08)
                    hint_penalty = min(0.3, adjacent_hint_sum / 30.0)
                    cell_value = -0.9 + neighbor_bonus - hint_penalty
                    state[0, row, col] = max(-1.0, cell_value)
                    state[1, row, col] = 1.0
                
                state[4, row, col] = hidden_neighbors / 8.0
                state[5, row, col] = flagged_neighbors / 8.0
                state[6, row, col] = min(1.0, adjacent_hint_sum / 64.0)
        
        return state
    
    def _calculate_reward(
        self,
        prev_revealed: int,
        valid_action: bool,
        action_type: str,
        move_context: Optional[dict] = None
    ) -> float:
        """
        Calculate reward for the action.
        
        Args:
            prev_revealed: Number of revealed cells before action
            valid_action: Whether the action was valid
            
        Returns:
            Reward value
        """
        if action_type != "reveal":
            return 0.0
        
        if not valid_action:
            return -0.4  # Penalty for invalid action
        
        board_scale = max(0.6, (self.width * self.height) / 600.0)
        cells_revealed = self.game.revealed_count - prev_revealed
        progress_ratio = self.game.revealed_count / max(1, self.game.total_safe_cells)
        is_guess = move_context.get("adjacent_revealed", 0) == 0 if move_context else True
        frontier_bonus = 0.0
        
        if move_context:
            if is_guess:
                frontier_bonus = -0.2
            else:
                frontier_bonus = 0.15 + 0.05 * min(4, move_context.get("adjacent_revealed", 0))
        
        if self.game.is_lost():
            # Keep penalty moderate to avoid dwarfing positive rewards
            penalty = -5.0 * board_scale
            return penalty + (frontier_bonus if is_guess else 0.0)
        
        if self.game.is_won():
            return 10.0 * board_scale + 5.0 * progress_ratio
        
        if cells_revealed > 0:
            base_reward = 0.25 * cells_revealed
            expansion_bonus = 0.05 * min(5, max(0, cells_revealed - 1))
            shaping = 0.1 * progress_ratio
            return base_reward + expansion_bonus + frontier_bonus + shaping
        
        # Slight penalty for ineffective move
        return -0.05
    
    def _handle_flag_action(self, row: int, col: int) -> Tuple[bool, float]:
        """
        Handle flag toggling and return (valid, reward).
        """
        cell = self.game.board.get_cell(row, col)
        if not cell or cell.is_revealed():
            return False, -0.1
        
        toggled = self.game.toggle_flag(row, col)
        if not toggled:
            return False, -0.1
        
        if cell.is_flagged():
            if cell.is_mine:
                return True, FLAG_SET_REWARD
            return True, -FLAG_SET_PENALTY
        
        # Flag removed (no immediate reward to avoid oscillation)
        return True, 0.0
    
    def _calculate_flag_end_bonus(self) -> float:
        """Reward/punish based on flagged cells when the game ends."""
        correct_flags = 0
        incorrect_flags = 0
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                if cell and cell.is_flagged():
                    if cell.is_mine:
                        correct_flags += 1
                    else:
                        incorrect_flags += 1
        
        return (
            correct_flags * FLAG_FINAL_CORRECT_BONUS
            - incorrect_flags * FLAG_FINAL_INCORRECT_PENALTY
        )
    
    def get_valid_actions(self) -> np.ndarray:
        """
        Get mask of valid actions (cells that can be clicked).
        
        Returns:
            Boolean array of valid actions
        """
        valid_reveal = np.zeros(self.cell_count, dtype=bool)
        valid_flag = np.zeros(self.cell_count, dtype=bool) if self.use_flag_actions else None
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                action_idx = row * self.width + col
                
                # Reveal valid if cell hidden and not flagged
                if cell.is_hidden() and not cell.is_flagged():
                    valid_reveal[action_idx] = True
                
                # Flag valid if cell hidden or currently flagged (toggle)
                if self.use_flag_actions and not cell.is_revealed():
                    if cell.is_hidden() or cell.is_flagged():
                        valid_flag[action_idx] = True
        
        if not self.use_flag_actions:
            return valid_reveal
        
        return np.concatenate([valid_reveal, valid_flag])
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get action mask for masking invalid actions in Q-values.
        
        Returns:
            Mask array (1.0 for valid, -inf for invalid)
        """
        valid_actions = self.get_valid_actions()
        mask = np.where(valid_actions, 0.0, -np.inf)
        return mask

    def _get_move_context(self, row: int, col: int) -> dict:
        """Collect contextual information for the selected move."""
        context = {
            "adjacent_revealed": 0,
            "adjacent_hidden": 0,
            "adjacent_flagged": 0
        }
        
        for dr, dc in NEIGHBOR_OFFSETS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                neighbor = self.game.board.get_cell(nr, nc)
                if not neighbor:
                    continue
                if neighbor.is_revealed():
                    context["adjacent_revealed"] += 1
                elif neighbor.is_flagged():
                    context["adjacent_flagged"] += 1
                elif neighbor.is_hidden():
                    context["adjacent_hidden"] += 1
        
        return context



```

```python
"""Training script for DQN agent."""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from src.reinforcement_learning.dqn_agent import DQNAgent
from src.reinforcement_learning.environment import MinesweeperEnvironment


def train_with_visualization(
    episodes: int = 1000,
    difficulty: str = "medium",
    width: int = None,
    height: int = None,
    save_path: str = "models/dqn_model.pth",
    log_interval: int = 100,
    visualization_callback: callable = None,
    thread=None,
    use_flag_actions: bool = True
):
    """
    Train DQN agent on Minesweeper with visualization support.
    
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
    
    # Adjust hyperparameters based on board size
    # Smaller boards need faster learning, larger boards need more exploration
    board_size_factor = (width * height) / 600.0  # Normalize to default 30x20
    
    # Learning rate scaled by board size (smaller boards learn faster)
    base_lr = 0.001
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
    
    # Batch size tuned per board size
    batch_size = min(96, max(32, int(48 * (1.0 / max(board_size_factor, 0.6)))))
    
    agent = DQNAgent(
        state_channels=env.state_channels,
        action_space_size=env.action_space_size,
        board_height=height,
        board_width=width,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_floor,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    episode_wins = []  # Track wins for each episode
    
    print(f"Starting training for {episodes} episodes...")
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
            
            # Select action
            action = agent.select_action(state, valid_actions)
            
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
        
        # Episode finished: decay epsilon once
        agent.decay_epsilon()
        
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
    log_interval: int = 100
):
    """
    Train DQN agent on Minesweeper (without visualization).
    
    Args:
        episodes: Number of training episodes
        difficulty: Game difficulty
        save_path: Path to save model
        log_interval: Episodes between logging
    """
    train_with_visualization(
        episodes=episodes,
        difficulty=difficulty,
        save_path=save_path,
        log_interval=log_interval,
        visualization_callback=None,
        thread=None,
        use_flag_actions=True
    )


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train DQN agent on Minesweeper")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--difficulty", type=str, default="medium", 
                       choices=["easy", "medium", "hard"], help="Game difficulty")
    parser.add_argument("--save-path", type=str, default="models/dqn_model.pth",
                       help="Path to save model")
    parser.add_argument("--log-interval", type=int, default=100,
                       help="Episodes between logging")
    parser.add_argument("--use-flags", action="store_true",
                       help="Enable flagging actions in the action space")
    
    args = parser.parse_args()
    
    train(
        episodes=args.episodes,
        difficulty=args.difficulty,
        save_path=args.save_path,
        log_interval=args.log_interval,
        use_flag_actions=args.use_flags
    )


if __name__ == "__main__":
    main()

```

```python
"""Training script for DQN agent."""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from src.reinforcement_learning.dqn_agent import DQNAgent
from src.reinforcement_learning.environment import MinesweeperEnvironment


def train_with_visualization(
    episodes: int = 1000,
    difficulty: str = "medium",
    width: int = None,
    height: int = None,
    save_path: str = "models/dqn_model.pth",
    log_interval: int = 100,
    visualization_callback: callable = None,
    thread=None,
    use_flag_actions: bool = True
):
    """
    Train DQN agent on Minesweeper with visualization support.
    
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
    
    # Adjust hyperparameters based on board size
    # Smaller boards need faster learning, larger boards need more exploration
    board_size_factor = (width * height) / 600.0  # Normalize to default 30x20
    
    # Learning rate scaled by board size (smaller boards learn faster)
    base_lr = 0.001
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
    
    # Batch size tuned per board size
    batch_size = min(96, max(32, int(48 * (1.0 / max(board_size_factor, 0.6)))))
    
    agent = DQNAgent(
        state_channels=env.state_channels,
        action_space_size=env.action_space_size,
        board_height=height,
        board_width=width,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_floor,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    episode_wins = []  # Track wins for each episode
    
    print(f"Starting training for {episodes} episodes...")
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
            
            # Select action
            action = agent.select_action(state, valid_actions)
            
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
        
        # Episode finished: decay epsilon once
        agent.decay_epsilon()
        
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
    use_flag_actions: bool = True
):
    """
    Train DQN agent on Minesweeper (without visualization).
    
    Args:
        episodes: Number of training episodes
        difficulty: Game difficulty
        save_path: Path to save model
        log_interval: Episodes between logging
    """
    train_with_visualization(
        episodes=episodes,
        difficulty=difficulty,
        save_path=save_path,
        log_interval=log_interval,
        visualization_callback=None,
        thread=None,
        use_flag_actions=use_flag_actions
    )


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train DQN agent on Minesweeper")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--difficulty", type=str, default="medium", 
                       choices=["easy", "medium", "hard"], help="Game difficulty")
    parser.add_argument("--save-path", type=str, default="models/dqn_model.pth",
                       help="Path to save model")
    parser.add_argument("--log-interval", type=int, default=100,
                       help="Episodes between logging")
    parser.add_argument("--use-flags", action="store_true",
                       help="Enable flagging actions in the action space")
    
    args = parser.parse_args()
    
    train(
        episodes=args.episodes,
        difficulty=args.difficulty,
        save_path=args.save_path,
        log_interval=args.log_interval,
        use_flag_actions=args.use_flags
    )


if __name__ == "__main__":
    main()

```

```python
"""Training script for DQN agent."""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from src.reinforcement_learning.dqn_agent import DQNAgent
from src.reinforcement_learning.environment import MinesweeperEnvironment


def train_with_visualization(
    episodes: int = 1000,
    difficulty: str = "medium",
    width: int = None,
    height: int = None,
    save_path: str = "models/dqn_model.pth",
    log_interval: int = 100,
    visualization_callback: callable = None,
    thread=None,
    use_flag_actions: bool = False
):
    """
    Train DQN agent on Minesweeper with visualization support.
    
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
    
    # Adjust hyperparameters based on board size
    # Smaller boards need faster learning, larger boards need more exploration
    board_size_factor = (width * height) / 600.0  # Normalize to default 30x20
    
    # Learning rate scaled by board size (smaller boards learn faster)
    base_lr = 0.001
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
    
    # Batch size tuned per board size
    batch_size = min(96, max(32, int(48 * (1.0 / max(board_size_factor, 0.6)))))
    
    agent = DQNAgent(
        state_channels=env.state_channels,
        action_space_size=env.action_space_size,
        board_height=height,
        board_width=width,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_floor,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    episode_wins = []  # Track wins for each episode
    
    print(f"Starting training for {episodes} episodes...")
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
            
            # Select action
            action = agent.select_action(state, valid_actions)
            
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
        
        # Episode finished: decay epsilon once
        agent.decay_epsilon()
        
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
    use_flag_actions: bool = False
):
    """
    Train DQN agent on Minesweeper (without visualization).
    
    Args:
        episodes: Number of training episodes
        difficulty: Game difficulty
        save_path: Path to save model
        log_interval: Episodes between logging
    """
    train_with_visualization(
        episodes=episodes,
        difficulty=difficulty,
        save_path=save_path,
        log_interval=log_interval,
        visualization_callback=None,
        thread=None,
        use_flag_actions=use_flag_actions
    )


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train DQN agent on Minesweeper")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--difficulty", type=str, default="medium", 
                       choices=["easy", "medium", "hard"], help="Game difficulty")
    parser.add_argument("--save-path", type=str, default="models/dqn_model.pth",
                       help="Path to save model")
    parser.add_argument("--log-interval", type=int, default=100,
                       help="Episodes between logging")
    parser.add_argument("--use-flags", action="store_true",
                       help="Enable flagging actions in the action space")
    
    args = parser.parse_args()
    
    train(
        episodes=args.episodes,
        difficulty=args.difficulty,
        save_path=args.save_path,
        log_interval=args.log_interval,
        use_flag_actions=args.use_flags
    )


if __name__ == "__main__":
    main()

```

```python
"""RL Visualizer for showing agent gameplay."""

import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from src.minesweeper.game import Game
from src.gui.game_board import GameBoard
from src.reinforcement_learning.environment import MinesweeperEnvironment
from src.reinforcement_learning.dqn_agent import DQNAgent
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class RLVisualizer(QWidget):
    """Widget for visualizing RL agent gameplay."""
    
    episode_finished = Signal(bool, float, int)  # won, reward, steps
    
    def __init__(self, game: Game, game_board: GameBoard):
        """
        Initialize RL visualizer.
        
        Args:
            game: Game instance
            game_board: GameBoard widget to update
        """
        super().__init__()
        self.game = game
        self.game_board = game_board
        self.env = None
        self.agent = None
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._play_step)
        self.is_playing = False
        self.current_state = None
        self.step_delay_ms = 100  # 100ms delay between steps
        self.use_flag_actions = True
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup UI for RL visualizer."""
        layout = QVBoxLayout()
        
        # Control buttons
        control_layout = QHBoxLayout()
        self.play_button = QPushButton("RL-Agent spielen lassen")
        self.play_button.clicked.connect(self._toggle_play)
        self.stop_button = QPushButton("Stoppen")
        self.stop_button.clicked.connect(self._stop_play)
        self.stop_button.setEnabled(False)
        
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        self.setLayout(layout)
    
    def set_agent(self, agent: DQNAgent):
        """Set the RL agent to use."""
        self.agent = agent
        if agent:
            self.agent.epsilon = 0.0  # Set to greedy mode for visualization
            total_cells = agent.board_height * agent.board_width
            self.use_flag_actions = agent.action_space_size > total_cells
        else:
            self.use_flag_actions = True
    
    def play_episode(self, agent: DQNAgent, difficulty: str = "medium", delay_ms: int = 100, width: int = None, height: int = None):
        """
        Play one episode with the agent and visualize it.
        
        Args:
            agent: DQN agent to use
            difficulty: Game difficulty
            delay_ms: Delay between steps in milliseconds
            width: Board width (uses current game width if None)
            height: Board height (uses current game height if None)
        """
        self.agent = agent
        if agent:
            self.agent.epsilon = 0.0  # Greedy mode for visualization
        self.step_delay_ms = delay_ms
        
        # Use current game size if not specified
        if width is None:
            width = self.game.width
        if height is None:
            height = self.game.height
        
        # Create environment with specified size
        self.env = MinesweeperEnvironment(difficulty, width, height, use_flag_actions=self.use_flag_actions)
        self.current_state = self.env.reset()
        
        # Sync GUI with environment's game (create new game with correct size)
        self.game.new_game(difficulty, width, height)
        # Copy game state from environment to our game object
        self._sync_game_from_env()
        
        # Start playing
        self.is_playing = True
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.play_timer.start(self.step_delay_ms)
    
    def _sync_game_from_env(self):
        """Sync GUI game with environment game."""
        if not self.env:
            return
        
        # Use environment's game directly for visualization
        # This is simpler and ensures consistency
        self.game_board.reset_game(self.env.game)
    
    def _play_step(self):
        """Execute one step of agent gameplay."""
        if not self.is_playing or not self.agent or not self.env:
            return
        
        # Check if game is over
        if self.env.game.is_game_over():
            self._finish_episode()
            return
        
        # Get valid actions
        valid_actions = self.env.get_valid_actions()
        if not np.any(valid_actions):
            self._finish_episode()
            return
        
        # Agent selects action
        action = self.agent.select_action(self.current_state, valid_actions)
        
        # Execute action
        next_state, reward, done, info = self.env.step(action)
        
        # Sync GUI with environment
        self._sync_game_from_env()
        
        # Update state
        self.current_state = next_state
        
        # Check if done
        if done:
            self._finish_episode()
    
    def _finish_episode(self):
        """Finish current episode."""
        self.play_timer.stop()
        self.is_playing = False
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        if self.env:
            won = self.env.game.is_won()
            # Calculate total reward (simplified - actual reward would need tracking)
            reward = 100.0 if won else -10.0
            steps = self.env.game.revealed_count
            
            self.episode_finished.emit(won, reward, steps)
    
    def _toggle_play(self):
        """Toggle play/pause."""
        if not self.agent:
            return
        
        if self.is_playing:
            self._stop_play()
        else:
            # Start new episode
            self.play_episode(self.agent, self.game.difficulty, delay_ms=100)
    
    def _stop_play(self):
        """Stop playing."""
        self.play_timer.stop()
        self.is_playing = False
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(False)


```

```python
"""Main window for Minesweeper application."""

from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QLabel, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
import torch
from src.gui.game_board import GameBoard
from src.gui.menu_bar import MenuBar
from src.gui.rl_visualizer import RLVisualizer
from src.minesweeper.game import Game
from src.reinforcement_learning.dqn_agent import DQNAgent
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        """Initialize main window."""
        super().__init__()
        self.game = Game("medium")
        self.timer = QTimer()
        self.elapsed_time = 0
        self.rl_agent = None
        self.training_thread = None
        self._setup_ui()
        self._setup_timer()
    
    def _setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("Minesweeper mit RL")
        self.setMinimumSize(950, 650)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Menu bar
        menu_bar = MenuBar()
        menu_bar.new_game_easy.connect(lambda: self.new_game("easy"))
        menu_bar.new_game_medium.connect(lambda: self.new_game("medium"))
        menu_bar.new_game_hard.connect(lambda: self.new_game("hard"))
        menu_bar.change_board_size.connect(self._change_board_size)
        menu_bar.start_rl_training.connect(self._start_rl_training)
        menu_bar.load_rl_model.connect(self._load_rl_model)
        menu_bar.quit_requested.connect(self.close)
        self.setMenuBar(menu_bar)
        
        # Status bar
        self.status_label = QLabel("Minen: 0 | Zeit: 0s")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Game board
        self.game_board = GameBoard(self.game)
        self.game_board.game_won.connect(self._on_game_won)
        self.game_board.game_lost.connect(self._on_game_lost)
        layout.addWidget(self.game_board)
        
        # RL Visualizer
        self.rl_visualizer = RLVisualizer(self.game, self.game_board)
        self.rl_visualizer.episode_finished.connect(self._on_rl_episode_finished)
        layout.addWidget(self.rl_visualizer)
        
        self._update_status()
    
    def _setup_timer(self):
        """Setup game timer."""
        self.timer.timeout.connect(self._update_timer)
        self.timer.start(1000)  # Update every second
    
    def _update_timer(self):
        """Update timer display."""
        if not self.game.is_game_over():
            self.elapsed_time += 1
            self._update_status()
    
    def _update_status(self):
        """Update status bar."""
        remaining_mines = self.game.get_remaining_mines()
        self.status_label.setText(f"Minen: {remaining_mines} | Zeit: {self.elapsed_time}s")
    
    def _on_game_won(self):
        """Handle game won event."""
        self.timer.stop()
        QMessageBox.information(
            self,
            "Gewonnen!",
            f"Glückwunsch! Sie haben das Spiel in {self.elapsed_time} Sekunden gewonnen!"
        )
    
    def _on_game_lost(self):
        """Handle game lost event."""
        self.timer.stop()
        QMessageBox.information(
            self,
            "Verloren!",
            "Sie haben eine Mine getroffen. Spiel beendet!"
        )
        self.game_board._update_display()  # Show all mines
    
    def new_game(self, difficulty: str):
        """Start a new game with specified difficulty."""
        self.game.new_game(difficulty, self.game.width, self.game.height)
        self.elapsed_time = 0
        self.timer.start()
        self.game_board.reset_game(self.game)
        self._update_status()
        
        difficulty_names = {
            "easy": "Leicht",
            "medium": "Mittel",
            "hard": "Schwer"
        }
        self.setWindowTitle(f"Minesweeper mit RL - {difficulty_names.get(difficulty, 'Mittel')} ({self.game.width}x{self.game.height})")
    
    def _change_board_size(self):
        """Change board size."""
        from PySide6.QtWidgets import QInputDialog
        from src.utils.constants import BOARD_SIZES
        
        # Show size selection dialog
        sizes = list(BOARD_SIZES.keys())
        size_names = {
            "small": "Klein (15x10)",
            "medium": "Mittel (20x15)",
            "large": "Groß (30x20)",
            "xlarge": "Sehr groß (40x25)",
            "custom": "Benutzerdefiniert"
        }
        
        size_list = [size_names.get(s, s) for s in sizes]
        current_size_name = None
        for size_name, (w, h) in BOARD_SIZES.items():
            if w == self.game.width and h == self.game.height:
                current_size_name = size_names.get(size_name, size_name)
                break
        
        selected, ok = QInputDialog.getItem(
            self, "Spielfeldgröße ändern", "Größe wählen:",
            size_list, size_list.index(current_size_name) if current_size_name else 0, False
        )
        if not ok:
            return
        
        # Find selected size
        selected_size = None
        for size_name, display_name in size_names.items():
            if display_name == selected:
                selected_size = size_name
                break
        
        if selected_size == "custom":
            # Custom size
            width, ok1 = QInputDialog.getInt(
                self, "Benutzerdefinierte Größe", "Breite:", self.game.width, 5, 100, 1
            )
            if not ok1:
                return
            height, ok2 = QInputDialog.getInt(
                self, "Benutzerdefinierte Größe", "Höhe:", self.game.height, 5, 100, 1
            )
            if not ok2:
                return
        else:
            width, height = BOARD_SIZES[selected_size]
        
        # Create new game with new size
        self.game = Game(self.game.difficulty, width, height)
        self.elapsed_time = 0
        self.timer.start()
        self.game_board.reset_game(self.game)
        self._update_status()
        
        self.setWindowTitle(f"Minesweeper mit RL - {self.game.difficulty.title()} ({width}x{height})")
    
    def _start_rl_training(self):
        """Start RL training with visualization."""
        # Simple dialog for training parameters
        from PySide6.QtWidgets import QInputDialog
        
        episodes, ok = QInputDialog.getInt(
            self, "Training starten", "Anzahl Episoden:", 1000, 100, 100000, 100
        )
        if not ok:
            return
        
        difficulty, ok = QInputDialog.getItem(
            self, "Training starten", "Schwierigkeit:",
            ["Leicht", "Mittel", "Schwer"], 1, False
        )
        if not ok:
            return
        
        difficulty_map = {"Leicht": "easy", "Mittel": "medium", "Schwer": "hard"}
        difficulty = difficulty_map[difficulty]
        
        # Use current game size for training
        width = self.game.width
        height = self.game.height
        
        # Ask if user wants to use current size or different size
        use_current, ok = QInputDialog.getItem(
            self, "Training starten", 
            f"Verwende aktuelle Spielfeldgröße ({width}x{height})?\n\n"
            f"Ja: Verwendet aktuelle Größe\n"
            f"Nein: Wähle andere Größe",
            ["Ja", "Nein"], 0, False
        )
        if not ok:
            return
        
        if use_current == "Nein":
            from src.utils.constants import BOARD_SIZES
            sizes = list(BOARD_SIZES.keys())
            size_names = {
                "small": "Klein (15x10)",
                "medium": "Mittel (20x15)",
                "large": "Groß (30x20)",
                "xlarge": "Sehr groß (40x25)",
                "custom": "Benutzerdefiniert"
            }
            size_list = [size_names.get(s, s) for s in sizes]
            
            selected, ok = QInputDialog.getItem(
                self, "Spielfeldgröße", "Größe wählen:", size_list, 0, False
            )
            if not ok:
                return
            
            selected_size = None
            for size_name, display_name in size_names.items():
                if display_name == selected:
                    selected_size = size_name
                    break
            
            if selected_size == "custom":
                width, ok1 = QInputDialog.getInt(
                    self, "Benutzerdefinierte Größe", "Breite:", width, 5, 100, 1
                )
                if not ok1:
                    return
                height, ok2 = QInputDialog.getInt(
                    self, "Benutzerdefinierte Größe", "Höhe:", height, 5, 100, 1
                )
                if not ok2:
                    return
            else:
                width, height = BOARD_SIZES[selected_size]
        
        save_path = f"models/dqn_model_{width}x{height}.pth"
        
        # Create signal for visualization (must be created before thread)
        from PySide6.QtCore import QObject
        class VisualizationEmitter(QObject):
            """Emitter for visualization signals from training thread."""
            visualization_requested = Signal(object, int, str, int, int)  # agent, episode, difficulty, width, height
        
        visualization_emitter = VisualizationEmitter()
        visualization_emitter.visualization_requested.connect(
            lambda agent, ep, diff, w, h: self._handle_visualization(agent, ep, diff, w, h)
        )
        
        def visualization_callback(agent, episode):
            """Callback for visualization during training."""
            # Emit signal from training thread to GUI thread
            visualization_emitter.visualization_requested.emit(agent, episode, difficulty, width, height)
            return agent
        
        # Start training in thread
        from src.reinforcement_learning.trainer import train_with_visualization
        
        class TrainingWorker(QThread):
            finished = Signal()
            
            def run(self):
                train_with_visualization(
                    episodes=episodes,
                    difficulty=difficulty,
                    width=width,
                    height=height,
                    save_path=save_path,
                    log_interval=100,
                    visualization_callback=visualization_callback,
                    thread=None
                )
                self.finished.emit()
        
        self.training_thread = TrainingWorker()
        self.training_thread.finished.connect(lambda: QMessageBox.information(
            self, "Training", "Training abgeschlossen!"
        ))
        self.training_thread.start()
        
        QMessageBox.information(
            self,
            "Training gestartet",
            f"Training mit {episodes} Episoden gestartet.\n\n"
            f"Größe: {width}x{height}\n"
            f"Schwierigkeit: {difficulty}\n\n"
            f"Alle 100 Episoden wird eine Visualisierung angezeigt.\n"
            f"Am Ende folgt ein finaler Test-Lauf."
        )
    
    def _load_rl_model(self):
        """Load RL model and test it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Modell laden", "models/", "PyTorch Model (*.pth)"
        )
        if not path:
            return
        
        try:
            # Create environment to get action space size (use current game size)
            env = MinesweeperEnvironment(self.game.difficulty, self.game.width, self.game.height)
            
            # Create agent
            agent = DQNAgent(
                state_channels=1,
                action_space_size=env.action_space_size,
                board_height=self.game.height,
                board_width=self.game.width,
                lr=0.001,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay=0.995
            )
            
            # Load model
            agent.load(path)
            agent.epsilon = 0.0  # Set to greedy mode
            
            # Set agent for visualization
            self.rl_agent = agent
            self.rl_visualizer.set_agent(agent)
            
            QMessageBox.information(
                self,
                "Modell geladen",
                f"Modell erfolgreich geladen: {path}\n\nKlicken Sie auf 'RL-Agent spielen lassen' um den Agent zu testen."
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Fehler",
                f"Fehler beim Laden des Modells:\n{str(e)}"
            )
    
    def _handle_visualization(self, agent, episode, difficulty, width, height):
        """Handle visualization request from training thread (runs in GUI thread)."""
        self.rl_agent = agent
        self.rl_visualizer.set_agent(agent)
        
        if episode == -1:  # Final test
            # Final test episode - play with faster delay
            self.rl_visualizer.play_episode(agent, difficulty, delay_ms=50, width=width, height=height)
        else:
            # Every 100 episodes - show visualization
            self.rl_visualizer.play_episode(agent, difficulty, delay_ms=100, width=width, height=height)
    
    def _on_rl_episode_finished(self, won: bool, reward: float, steps: int):
        """Handle RL episode completion."""
        result_text = "Gewonnen!" if won else "Verloren!"
        QMessageBox.information(
            self,
            "RL-Episode beendet",
            f"{result_text}\n\nSchritte: {steps}\nReward: {reward:.2f}"
        )



```

```python
"""Main window for Minesweeper application."""

from PySide6.QtWidgets import (
    QMainWindow, QVBoxLayout, QWidget, QLabel, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
import torch
from src.gui.game_board import GameBoard
from src.gui.menu_bar import MenuBar
from src.gui.rl_visualizer import RLVisualizer
from src.minesweeper.game import Game
from src.reinforcement_learning.dqn_agent import DQNAgent
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        """Initialize main window."""
        super().__init__()
        self.game = Game("medium")
        self.timer = QTimer()
        self.elapsed_time = 0
        self.rl_agent = None
        self.training_thread = None
        self._setup_ui()
        self._setup_timer()
    
    def _setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("Minesweeper mit RL")
        self.setMinimumSize(950, 650)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Menu bar
        menu_bar = MenuBar()
        menu_bar.new_game_easy.connect(lambda: self.new_game("easy"))
        menu_bar.new_game_medium.connect(lambda: self.new_game("medium"))
        menu_bar.new_game_hard.connect(lambda: self.new_game("hard"))
        menu_bar.change_board_size.connect(self._change_board_size)
        menu_bar.start_rl_training.connect(self._start_rl_training)
        menu_bar.load_rl_model.connect(self._load_rl_model)
        menu_bar.quit_requested.connect(self.close)
        self.setMenuBar(menu_bar)
        
        # Status bar
        self.status_label = QLabel("Minen: 0 | Zeit: 0s")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Game board
        self.game_board = GameBoard(self.game)
        self.game_board.game_won.connect(self._on_game_won)
        self.game_board.game_lost.connect(self._on_game_lost)
        layout.addWidget(self.game_board)
        
        # RL Visualizer
        self.rl_visualizer = RLVisualizer(self.game, self.game_board)
        self.rl_visualizer.episode_finished.connect(self._on_rl_episode_finished)
        layout.addWidget(self.rl_visualizer)
        
        self._update_status()
    
    def _setup_timer(self):
        """Setup game timer."""
        self.timer.timeout.connect(self._update_timer)
        self.timer.start(1000)  # Update every second
    
    def _update_timer(self):
        """Update timer display."""
        if not self.game.is_game_over():
            self.elapsed_time += 1
            self._update_status()
    
    def _update_status(self):
        """Update status bar."""
        remaining_mines = self.game.get_remaining_mines()
        self.status_label.setText(f"Minen: {remaining_mines} | Zeit: {self.elapsed_time}s")
    
    def _on_game_won(self):
        """Handle game won event."""
        self.timer.stop()
        QMessageBox.information(
            self,
            "Gewonnen!",
            f"Glückwunsch! Sie haben das Spiel in {self.elapsed_time} Sekunden gewonnen!"
        )
    
    def _on_game_lost(self):
        """Handle game lost event."""
        self.timer.stop()
        QMessageBox.information(
            self,
            "Verloren!",
            "Sie haben eine Mine getroffen. Spiel beendet!"
        )
        self.game_board._update_display()  # Show all mines
    
    def new_game(self, difficulty: str):
        """Start a new game with specified difficulty."""
        self.game.new_game(difficulty, self.game.width, self.game.height)
        self.elapsed_time = 0
        self.timer.start()
        self.game_board.reset_game(self.game)
        self._update_status()
        
        difficulty_names = {
            "easy": "Leicht",
            "medium": "Mittel",
            "hard": "Schwer"
        }
        self.setWindowTitle(f"Minesweeper mit RL - {difficulty_names.get(difficulty, 'Mittel')} ({self.game.width}x{self.game.height})")
    
    def _change_board_size(self):
        """Change board size."""
        from PySide6.QtWidgets import QInputDialog
        from src.utils.constants import BOARD_SIZES
        
        # Show size selection dialog
        sizes = list(BOARD_SIZES.keys())
        size_names = {
            "small": "Klein (15x10)",
            "medium": "Mittel (20x15)",
            "large": "Groß (30x20)",
            "xlarge": "Sehr groß (40x25)",
            "custom": "Benutzerdefiniert"
        }
        
        size_list = [size_names.get(s, s) for s in sizes]
        current_size_name = None
        for size_name, (w, h) in BOARD_SIZES.items():
            if w == self.game.width and h == self.game.height:
                current_size_name = size_names.get(size_name, size_name)
                break
        
        selected, ok = QInputDialog.getItem(
            self, "Spielfeldgröße ändern", "Größe wählen:",
            size_list, size_list.index(current_size_name) if current_size_name else 0, False
        )
        if not ok:
            return
        
        # Find selected size
        selected_size = None
        for size_name, display_name in size_names.items():
            if display_name == selected:
                selected_size = size_name
                break
        
        if selected_size == "custom":
            # Custom size
            width, ok1 = QInputDialog.getInt(
                self, "Benutzerdefinierte Größe", "Breite:", self.game.width, 5, 100, 1
            )
            if not ok1:
                return
            height, ok2 = QInputDialog.getInt(
                self, "Benutzerdefinierte Größe", "Höhe:", self.game.height, 5, 100, 1
            )
            if not ok2:
                return
        else:
            width, height = BOARD_SIZES[selected_size]
        
        # Create new game with new size
        self.game = Game(self.game.difficulty, width, height)
        self.elapsed_time = 0
        self.timer.start()
        self.game_board.reset_game(self.game)
        self._update_status()
        
        self.setWindowTitle(f"Minesweeper mit RL - {self.game.difficulty.title()} ({width}x{height})")
    
    def _start_rl_training(self):
        """Start RL training with visualization."""
        # Simple dialog for training parameters
        from PySide6.QtWidgets import QInputDialog
        
        episodes, ok = QInputDialog.getInt(
            self, "Training starten", "Anzahl Episoden:", 1000, 100, 100000, 100
        )
        if not ok:
            return
        
        difficulty, ok = QInputDialog.getItem(
            self, "Training starten", "Schwierigkeit:",
            ["Leicht", "Mittel", "Schwer"], 1, False
        )
        if not ok:
            return
        
        difficulty_map = {"Leicht": "easy", "Mittel": "medium", "Schwer": "hard"}
        difficulty = difficulty_map[difficulty]
        
        # Use current game size for training
        width = self.game.width
        height = self.game.height
        
        # Ask if user wants to use current size or different size
        use_current, ok = QInputDialog.getItem(
            self, "Training starten", 
            f"Verwende aktuelle Spielfeldgröße ({width}x{height})?\n\n"
            f"Ja: Verwendet aktuelle Größe\n"
            f"Nein: Wähle andere Größe",
            ["Ja", "Nein"], 0, False
        )
        if not ok:
            return
        
        if use_current == "Nein":
            from src.utils.constants import BOARD_SIZES
            sizes = list(BOARD_SIZES.keys())
            size_names = {
                "small": "Klein (15x10)",
                "medium": "Mittel (20x15)",
                "large": "Groß (30x20)",
                "xlarge": "Sehr groß (40x25)",
                "custom": "Benutzerdefiniert"
            }
            size_list = [size_names.get(s, s) for s in sizes]
            
            selected, ok = QInputDialog.getItem(
                self, "Spielfeldgröße", "Größe wählen:", size_list, 0, False
            )
            if not ok:
                return
            
            selected_size = None
            for size_name, display_name in size_names.items():
                if display_name == selected:
                    selected_size = size_name
                    break
            
            if selected_size == "custom":
                width, ok1 = QInputDialog.getInt(
                    self, "Benutzerdefinierte Größe", "Breite:", width, 5, 100, 1
                )
                if not ok1:
                    return
                height, ok2 = QInputDialog.getInt(
                    self, "Benutzerdefinierte Größe", "Höhe:", height, 5, 100, 1
                )
                if not ok2:
                    return
            else:
                width, height = BOARD_SIZES[selected_size]
        
        save_path = f"models/dqn_model_{width}x{height}.pth"
        
        # Create signal for visualization (must be created before thread)
        from PySide6.QtCore import QObject
        class VisualizationEmitter(QObject):
            """Emitter for visualization signals from training thread."""
            visualization_requested = Signal(object, int, str, int, int)  # agent, episode, difficulty, width, height
        
        visualization_emitter = VisualizationEmitter()
        visualization_emitter.visualization_requested.connect(
            lambda agent, ep, diff, w, h: self._handle_visualization(agent, ep, diff, w, h)
        )
        
        def visualization_callback(agent, episode):
            """Callback for visualization during training."""
            # Emit signal from training thread to GUI thread
            visualization_emitter.visualization_requested.emit(agent, episode, difficulty, width, height)
            return agent
        
        # Start training in thread
        from src.reinforcement_learning.trainer import train_with_visualization
        
        class TrainingWorker(QThread):
            finished = Signal()
            
            def run(self):
                train_with_visualization(
                    episodes=episodes,
                    difficulty=difficulty,
                    width=width,
                    height=height,
                    save_path=save_path,
                    log_interval=100,
                    visualization_callback=visualization_callback,
                    thread=None
                )
                self.finished.emit()
        
        self.training_thread = TrainingWorker()
        self.training_thread.finished.connect(lambda: QMessageBox.information(
            self, "Training", "Training abgeschlossen!"
        ))
        self.training_thread.start()
        
        QMessageBox.information(
            self,
            "Training gestartet",
            f"Training mit {episodes} Episoden gestartet.\n\n"
            f"Größe: {width}x{height}\n"
            f"Schwierigkeit: {difficulty}\n\n"
            f"Alle 100 Episoden wird eine Visualisierung angezeigt.\n"
            f"Am Ende folgt ein finaler Test-Lauf."
        )
    
    def _load_rl_model(self):
        """Load RL model and test it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Modell laden", "models/", "PyTorch Model (*.pth)"
        )
        if not path:
            return
        
        try:
            checkpoint = torch.load(path, map_location=torch.device("cpu"))
            board_width = checkpoint.get("board_width", self.game.width)
            board_height = checkpoint.get("board_height", self.game.height)
            action_space_size = checkpoint.get("action_space_size", board_width * board_height * 2)
            
            agent = DQNAgent(
                state_channels=STATE_CHANNELS,
                action_space_size=action_space_size,
                board_height=board_height,
                board_width=board_width,
                lr=0.001,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay=0.995
            )
            
            # Load model
            agent.load(path)
            agent.epsilon = 0.0  # Set to greedy mode
            
            # Ensure GUI board matches model size
            if self.game.width != board_width or self.game.height != board_height:
                self.game.new_game(self.game.difficulty, board_width, board_height)
                self.game_board.reset_game(self.game)
            
            # Set agent for visualization
            self.rl_agent = agent
            self.rl_visualizer.set_agent(agent)
            
            QMessageBox.information(
                self,
                "Modell geladen",
                (
                    f"Modell erfolgreich geladen: {path}\n"
                    f"Boardgröße: {board_width}x{board_height}\n"
                    f"Flags im Aktionsraum: {'Ja' if action_space_size > board_width * board_height else 'Nein'}\n\n"
                    "Klicken Sie auf 'RL-Agent spielen lassen', um den Agent zu testen."
                )
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Fehler",
                f"Fehler beim Laden des Modells:\n{str(e)}"
            )
    
    def _handle_visualization(self, agent, episode, difficulty, width, height):
        """Handle visualization request from training thread (runs in GUI thread)."""
        self.rl_agent = agent
        self.rl_visualizer.set_agent(agent)
        
        if episode == -1:  # Final test
            # Final test episode - play with faster delay
            self.rl_visualizer.play_episode(agent, difficulty, delay_ms=50, width=width, height=height)
        else:
            # Every 100 episodes - show visualization
            self.rl_visualizer.play_episode(agent, difficulty, delay_ms=100, width=width, height=height)
    
    def _on_rl_episode_finished(self, won: bool, reward: float, steps: int):
        """Handle RL episode completion."""
        result_text = "Gewonnen!" if won else "Verloren!"
        QMessageBox.information(
            self,
            "RL-Episode beendet",
            f"{result_text}\n\nSchritte: {steps}\nReward: {reward:.2f}"
        )



```

```markdown
# Reinforcement Learning Implementierungs-Guide

Dieses Dokument erklärt detailliert, wie das Reinforcement Learning System für Minesweeper implementiert ist.

## Übersicht

Das RL-System verwendet **Deep Q-Network (DQN)** mit folgenden Komponenten:
- **Environment**: Wrapper für das Minesweeper-Spiel
- **DQN Network**: Convolutional Neural Network für Q-Value-Schätzung
- **DQN Agent**: Agent mit Experience Replay und Target Network
- **Training**: Episoden-basiertes Training mit Logging

---

## 1. Environment (`src/reinforcement_learning/environment.py`)

### Zweck
Wrappt das Minesweeper-Spiel in ein Gymnasium-ähnliches Interface für RL-Algorithmen.

### Implementierung

#### State Representation (Zustandsrepräsentation)

Der State besitzt **7 Kanäle** und liefert vollständigen Kontext über jede Zelle und ihre Nachbarschaft:

1. **Basis-Encoding** (`state[0]`): alter kombinierter Wert (-0.9 hidden, -0.5 Flag, -1 Mine, 0‑1 Zahl) inkl. Nachbarschaftsbonus/-strafe  
2. **Hidden-Maske** (`state[1]`): 1 für verdeckte Zelle, sonst 0  
3. **Flag-Maske** (`state[2]`): 1 für gesetzte Flagge, sonst 0  
4. **Aufgedeckte Zahl** (`state[3]`): normierte Zahl (0‑1) bzw. -1 bei Mine  
5. **Verdeckte Nachbarn** (`state[4]`): Anteil verdeckter Nachbarn (0‑1)  
6. **Flag-Nachbarn** (`state[5]`): Anteil geflaggter Nachbarn (0‑1)  
7. **Hinweis-Summe** (`state[6]`): normierte Summe aller bekannten Nachbarzahlen (0‑1)

Damit kennt der Agent sowohl den lokalen Status als auch die unmittelbare „Frontier“ und vorhandene Hinweise.

#### Action Space (Reveal vs. Flag)

- **Standard (GUI/Test)**: Reveal + Flag mit Größe `2 × width × height`
- **Training (Standard)**: `use_flag_actions=False` → nur Reveal-Aktionen (`width × height`), analog zum Ansatz aus [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning), damit der Agent sich wie ein menschlicher Spieler auf Progress-Züge konzentriert
- **CLI Flag**: `python -m ...trainer --use-flags` aktiviert wieder das alte Verhalten
- **Indexierung (mit Flags)**:
  - `0 … (N-1)`: Zelle aufdecken
  - `N … (2N-1)`: Flagge toggle
- **Validierung**:
  - Reveal nur für verdeckte, nicht geflaggte Felder
  - Flag nur für verdeckte oder bereits geflaggte Felder

#### Reward System

Orientiert sich an den Progress/Gues-Rewards aus [sdlee94s Minesweeper DQN](https://sdlee94.github.io/Minesweeper-AI-Reinforcement-Learning/):

- **Aufdecken**:
  - +0.25 pro sicher aufgedecktem Feld
  - +0.05 Bonus für Kettenreaktionen (mehrere Felder in einem Schritt)
  - +Frontier-Bonus bis +0.35, wenn mindestens ein Nachbar bereits bekannt ist
  - -0.2 Malus für reine Guess-Züge ohne Nachbarinformationen
  - -0.05 für wirkungslose Aktionen
- **Flaggen (sofort)**:
  - +0.2 bei korrekt gesetzter Flagge
  - -0.2 bei falscher Flagge
  - 0.0 beim Entfernen (kein Schwingen)
- **Flaggen (bei Verlust)**:
  - +0.5 pro korrekter Flagge
  - -0.5 pro falscher Flagge
- **Spiel verloren**: Moderate Strafe (~ -5 × Board-Skalierung), damit positive Rewards nicht erdrückt werden
- **Spiel gewonnen**: +10 × Board-Skalierung plus kleiner Progress-Bonus

#### Valid Actions Masking

Das Environment bietet zwei Methoden für gültige Aktionen:

1. **`get_valid_actions()`**: Boolean-Array (True = gültig)
2. **`get_action_mask()`**: Mask für Q-Values (-inf für ungültig, 0.0 für gültig)

**Warum Masking?**
- Verhindert, dass der Agent bereits aufgedeckte oder flagge Zellen auswählt
- Reduziert den Aktionsraum effektiv
- Verbessert Trainingseffizienz

---

## 2. DQN Network (`src/reinforcement_learning/network.py`)

### Architektur

**Convolutional Neural Network** nach sdlee94-Vorbild:

```
Input: (batch, 7, H, W)

Conv Stack (×4):
- Conv2d(in → 128, kernel=3, padding=1)
- BatchNorm2d(128)
- ReLU

AdaptiveAvgPool2d(8 × 8)  ->  128 × 8 × 8 Features

Fully Connected:
- Linear(8192 → 512) + Dropout(0.25)
- Linear(512 → 512) + Dropout(0.25)
- Linear(512 → num_actions)
```

- **4 × 128er Convs**: stärkere lokale Mustererkennung wie im Referenzprojekt  
- **Adaptive Pooling**: funktioniert auf allen Brettgrößen  
- **Doppelte 512er Dense-Layer**: entspricht `conv128x4_dense512x2` aus [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning)

---

## 3. DQN Agent (`src/reinforcement_learning/dqn_agent.py`)

### Komponenten

#### 3.1 ReplayBuffer

**Zweck**: Speichert Erfahrungen (State, Action, Reward, Next State, Done, Next-Action-Maske) für Experience Replay.

**Implementierung:**
- `deque` mit `maxlen` für automatische Größenbegrenzung
- Speichert zusätzlich die zulässigen Aktionen des Folgezustands zur Maskierung der Ziel-Q-Werte
- Zufälliges Sampling für Batch-Training
- Konvertiert NumPy-Arrays zu PyTorch-Tensoren

**Warum Experience Replay?**
- **Stabilisierung**: Bricht Korrelationen zwischen aufeinanderfolgenden Erfahrungen
- **Efficiency**: Nutzt Erfahrungen mehrfach
- **Diversität**: Batch enthält verschiedene Erfahrungen

#### 3.2 DQN Agent

**Zweck**: Implementiert DQN-Algorithmus mit allen notwendigen Komponenten.

**Hauptkomponenten:**

1. **Q-Network**: Haupt-Netzwerk für Q-Value-Schätzung
2. **Double Target Network**: Double DQN vermeidet Q-Value-Overestimation
3. **Epsilon-Greedy**: Exploration vs. Exploitation
4. **Experience Replay**: Speichert und nutzt vergangene Erfahrungen

**Hyperparameter:**

```python
- Learning Rate: 0.001 (Adam Optimizer)
- Gamma (Discount): 0.95 auf Standardbrettern (mehr Fokus auf Fortschritt)
- Epsilon Start: 1.0 (100% Exploration)
- Epsilon End: 0.03–0.10 je nach Schwierigkeitsgrad
- Epsilon Decay: dynamisch berechnet (~70 % der Episoden explorativ)
- Buffer Size: 10,000 Erfahrungen
- Batch Size: 32–96 (abhängig von Brettgröße)
- Target Update: Alle 100 Steps
- Action Space: Standardmäßig nur Reveal-Aktionen (`use_flag_actions=False`), Flags können per CLI aktiviert werden
```

**Gezielte Epsilon-Greedy Strategie:**

```python
if random() < epsilon:
    action = sample_frontier(valid_actions, state)  # Bevorzugt Zellen an bekannten Grenzen
else:
    action = argmax(Q(state, valid_only=True))
```

- **Frontier-Sampling**: Zufallsaktionen wählen bevorzugt verdeckte Felder, die an bereits aufgedeckte Zahlen grenzen  
- **Fallback auf Hinweise**: Wenn keine Frontier existiert, wird ein Feld mit dem größten Hinweis-Signal gewählt  
- **Episodenweiser Decay**: `epsilon` wird nach jeder Episode reduziert, nicht nach jedem Schritt

**Training Process (train_step):**

1. **Batch Sampling**: Zufällige Erfahrungen inkl. Aktionsmasken aus Replay Buffer
2. **Current Q-Values**: Q-Network schätzt Q(s, a)
3. **Target Q-Values**: Double DQN – Aktion via Online-Netz wählen, Q-Wert via Target-Netz bewerten
4. **TD Target**: `target = reward + gamma * Q_target(s', argmax_a Q_online(s'))`
5. **Loss**: Smooth L1 (Huber) zwischen Current und Target
6. **Backpropagation**: Gradienten werden berechnet und angewendet
7. **Gradient Clipping**: Verhindert Exploding Gradients (max norm = 1.0)
8. **Epsilon Decay**: Erfolgt episodenweise über `agent.decay_epsilon()`
9. **Target Update**: Alle N Steps wird Target Network aktualisiert

**Warum Target Network?**
- **Stabilität**: Verhindert instabile Q-Value-Updates
- **Convergence**: Hilft beim Konvergieren des Trainings
- **Delayed Updates**: Target Network wird nur periodisch aktualisiert

**Warum Gradient Clipping?**
- **Stabilität**: Verhindert sehr große Gradienten
- **Training**: Stabilisiert den Lernprozess

---

## 4. Training (`src/reinforcement_learning/trainer.py`)

### Training Loop

```python
for episode in range(episodes):
    state = env.reset()
    done = False
    
    while not done:
        # 1. Agent wählt Aktion
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions)
        
        # 2. Environment führt Aktion aus
        next_state, reward, done, info = env.step(action)
        next_valid_actions = env.get_valid_actions()
        
        # 3. Agent speichert Erfahrung
        agent.remember(state, action, reward, next_state, done, next_valid_actions)
        
        # 4. Agent trainiert (wenn genug Erfahrungen)
        agent.train_step()
        
        # 5. Update state
        state = next_state
    
    # 6. Exploration langsam reduzieren
    agent.decay_epsilon()
    
    # Logging alle N Episoden
    if episode % log_interval == 0:
        log_statistics()
```

### Logging

Alle N Episoden werden folgende Statistiken geloggt:
- **Average Reward**: Durchschnittliche Belohnung
- **Average Length**: Durchschnittliche Episodenlänge
- **Win Rate**: Prozentsatz gewonnener Spiele
- **Epsilon**: Aktueller Exploration-Faktor

### Model Saving

- Periodisches Speichern (alle N × log_interval Episoden)
- Finales Speichern nach Training
- Enthält: Q-Network, Target Network, Optimizer State, Epsilon

---

## 5. Design-Entscheidungen und Optimierungen

### Warum DQN?

1. **Discrete Action Space**: Minesweeper hat diskreten Aktionsraum (600 Aktionen)
2. **Value-Based**: Q-Learning passt gut für deterministische Umgebungen
3. **Bewährt**: DQN ist etabliert und gut verstanden

### Warum diese State-Encoding?

1. **Einfachheit**: Ein Kanal, einfache CNN-Architektur
2. **Normalisierung**: Werte zwischen -1 und 1 für besseres Training
3. **Informativ**: Unterscheidet alle wichtigen Zellzustände

### Warum diese Reward-Struktur?

1. **Sparse + Shaped**: Hauptrewards für wichtige Ereignisse, Shaped für Fortschritt
2. **Balance**: Nicht zu viele kleine Rewards (verhindert Overfitting)
3. **Klare Signale**: Gewinn/Verlust sind deutlich signalisiert

### Mögliche Verbesserungen

1. **Dueling DQN**: Trennt State-Value und Advantage
2. **Prioritized Replay**: Wichtige Erfahrungen werden öfter gesampelt
3. **Multi-Step Learning**: N-Step Returns statt 1-Step
4. **State-Encoding**: Mehr Features (z.B. Wahrscheinlichkeiten je Feld)
5. **Curriculum**: Transfer Learning zwischen Schwierigkeitsgraden

---

## 6. Verwendung

### Training starten:

```bash
python -m src.reinforcement_learning.trainer \
    --episodes 1000 \
    --difficulty medium \
    --save-path models/dqn_model.pth
```

### Modell laden:

```python
from src.reinforcement_learning.dqn_agent import DQNAgent
from src.reinforcement_learning.environment import MinesweeperEnvironment

env = MinesweeperEnvironment("medium")
agent = DQNAgent(state_channels=1, action_space_size=600)
agent.load("models/dqn_model.pth")

# Agent spielen lassen
state = env.reset()
done = False
while not done:
    valid_actions = env.get_valid_actions()
    action = agent.select_action(state, valid_actions)
    state, reward, done, info = env.step(action)
```

---

## 7. Zusammenfassung

Das RL-System implementiert:

✅ **Environment Wrapper**: Gymnasium-ähnliches Interface
✅ **State Encoding**: Normalisierte 2D-Repräsentation
✅ **DQN Network**: CNN für räumliche Features
✅ **Experience Replay**: Stabilisiert Training
✅ **Target Network**: Verhindert instabile Updates
✅ **Epsilon-Greedy**: Exploration/Exploitation Balance
✅ **Reward Shaping**: Sparse + Shaped Rewards
✅ **Action Masking**: Verhindert ungültige Aktionen

Das System ist vollständig funktionsfähig und bereit für Training!


```

```markdown
# RL Training & Visualisierung - Anleitung

## Übersicht

Diese Anleitung erklärt, wie Sie den DQN-Agenten trainieren und die Visualisierung verwenden.

## Training starten

### Über die GUI (Empfohlen)

1. **Anwendung starten:**
   ```bash
   python main.py
   ```

2. **Training starten:**
   - Klicken Sie auf: **Menü → Reinforcement Learning → Training starten (mit Visualisierung)**
   - Im ersten Dialog: Anzahl Episoden eingeben (z.B. `1000`)
   - Im zweiten Dialog: Schwierigkeit wählen:
     - **Leicht**: Weniger Minen, einfacher zum Lernen
     - **Mittel**: Standard-Schwierigkeit
     - **Schwer**: Mehr Minen, schwieriger

3. **Training läuft:**
   - Das Training läuft im Hintergrund
   - Alle **100 Episoden** wird automatisch eine Visualisierung gestartet
   - Am **Ende** folgt ein finaler Test-Lauf

### Über die Kommandozeile (Ohne GUI)

```bash
python -m src.reinforcement_learning.trainer --episodes 1000 --difficulty medium
```

**Optional: Flags deaktivieren/aktivieren**

- Standard (`main.py` & CLI): `use_flag_actions=False` → Agent konzentriert sich nur aufs Aufdecken (entspricht [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning))
- Falls Flaggen explizit trainiert werden sollen:

```bash
python -m src.reinforcement_learning.trainer --episodes 1000 --difficulty medium --use-flags
```

**Parameter:**
- `--episodes`: Anzahl Episoden (Standard: 1000)
- `--difficulty`: easy, medium, hard (Standard: medium)
- `--save-path`: Pfad zum Speichern (Standard: models/dqn_model.pth)
- `--log-interval`: Episoden zwischen Logging (Standard: 100)
- `--use-flags`: Flag- und Aufdeck-Aktionen trainieren (Standard deaktiviert)

## Visualisierung verstehen

### Automatische Visualisierungen (alle 100 Episoden)

Während des Trainings:
1. Nach jeder 100. Episode erscheint automatisch eine Visualisierung
2. Der Agent spielt eine komplette Episode
3. Sie sehen in Echtzeit, wie der Agent Zellen aufdeckt
4. Nach der Episode wird das Ergebnis angezeigt (Gewonnen/Verloren)

### Finaler Test-Lauf

Am Ende des Trainings:
- Ein finaler Test-Lauf wird automatisch gestartet
- Schnellere Visualisierung (50ms Delay statt 100ms)
- Zeigt die finale Leistung des trainierten Agents

## Geladenes Modell testen

### Modell laden

1. **Menü → Reinforcement Learning → Modell laden und testen**
2. Wählen Sie die Modell-Datei (z.B. `models/dqn_model.pth`)
3. Klicken Sie OK

### Agent manuell spielen lassen

1. Im **RL-Visualizer-Widget** (unterhalb des Spielfelds) finden Sie:
   - Button: **"RL-Agent spielen lassen"**
   - Button: **"Stoppen"**

2. **Agent starten:**
   - Klicken Sie auf "RL-Agent spielen lassen"
   - Der Agent spielt automatisch eine Episode
   - Sie sehen in Echtzeit die Entscheidungen

3. **Agent stoppen:**
   - Klicken Sie auf "Stoppen" um die aktuelle Episode zu beenden

## Tipps für effektives Training

### Training-Parameter

- **Episoden**: Starten Sie mit 1000 Episoden für erste Tests
- **Schwierigkeit**: Beginnen Sie mit "Leicht" für schnelleres Lernen
- **Wartezeit**: Die Visualisierung kann mit dem Delay-Parameter angepasst werden

### Was zu beachten ist

1. **Training dauert**: Je nach Episodenanzahl kann Training mehrere Minuten/Stunden dauern
2. **GUI bleibt aktiv**: Sie können während des Trainings weiterhin manuell spielen
3. **Modell wird gespeichert**: Das Modell wird automatisch gespeichert (Standard: `models/dqn_model.pth`)

## Troubleshooting

### Training startet nicht

- Überprüfen Sie, ob PyTorch korrekt installiert ist
- Prüfen Sie die Konsole auf Fehlermeldungen

### Visualisierung erscheint nicht

- Warten Sie bis Episode 100, 200, 300, etc. erreicht wird
- Prüfen Sie, ob das Training läuft (Konsole)

### Modell kann nicht geladen werden

- Stellen Sie sicher, dass das Modell bereits trainiert wurde
- Überprüfen Sie den Pfad zur Modell-Datei
- Prüfen Sie, ob die Modell-Datei existiert (`models/dqn_model.pth`)

## Fortgeschrittene Nutzung

### Training fortsetzen

Sie können ein bereits trainiertes Modell laden und weiter trainieren (zukünftige Erweiterung).

### Verschiedene Schwierigkeitsgrade

- Trainieren Sie verschiedene Modelle für verschiedene Schwierigkeitsgrade
- Speichern Sie sie mit unterschiedlichen Namen (z.B. `dqn_model_easy.pth`, `dqn_model_hard.pth`)

## Beispiel-Workflow

1. **Start**: `python main.py`
2. **Training starten**: Menü → RL → Training starten
   - 1000 Episoden, Mittel
3. **Warten**: Beobachten Sie die Visualisierungen alle 100 Episoden
4. **Modell laden**: Nach Training → Menü → RL → Modell laden
5. **Testen**: "RL-Agent spielen lassen" klicken
6. **Beobachten**: Sehen Sie, wie gut der Agent spielt!


```

```markdown
# KIP_Projekt - Minesweeper mit Reinforcement Learning

Ein vollständiges Minesweeper-Spiel mit GUI (PySide6) und Reinforcement Learning (DQN mit PyTorch).

## Projektübersicht

Dieses Projekt implementiert:
- **Minesweeper-Spiel** mit 20x30 Spielfeld
- **GUI** mit PySide6 für manuelles Spielen
- **Reinforcement Learning** mit Deep Q-Network (DQN) basierend auf PyTorch
- **3 Schwierigkeitsgrade**: Leicht (~10%), Mittel (~15%), Schwer (~20%)

## Projektstruktur

```
KIP_Projekt/
├── src/
│   ├── minesweeper/          # Spiellogik
│   ├── gui/                  # GUI-Komponenten
│   ├── reinforcement_learning/ # RL-Implementierung
│   └── utils/                # Hilfsfunktionen
├── models/                   # Gespeicherte RL-Modelle
├── data/                     # Training-Daten/Logs
├── requirements.txt
└── main.py                   # Hauptprogramm
```

## Installation

### Voraussetzungen

Python 3.8 oder höher muss installiert sein. Falls Python noch nicht installiert ist:

**Python von python.org**
1. Besuchen Sie https://www.python.org/downloads/
2. Laden Sie die neueste Python-Version herunter (3.8+)
3. Während der Installation: **Wichtig:** Aktivieren Sie "Add Python to PATH"
4. Installation abschließen

**Installation überprüfen:**
```powershell
python --version
# oder
py --version
```

### Projekt-Setup

1. Repository klonen oder herunterladen
2. Abhängigkeiten installieren:
```bash
# Wenn Python im PATH ist:
pip install -r requirements.txt

# Falls pip nicht gefunden wird, versuchen Sie:
python -m pip install -r requirements.txt
# oder
py -m pip install -r requirements.txt
```

**Hinweis für Windows:** Falls `pip` nicht erkannt wird, verwenden Sie `python -m pip` oder `py -m pip`.

### PyTorch Installation (Windows)

PyTorch benötigt die **Visual C++ Redistributable**. Falls beim Import von PyTorch ein Fehler auftritt:

**Visual C++ Redistributable installieren**
1. Laden Sie die Visual C++ Redistributable herunter:
   - https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Installieren Sie die Datei
3. PowerShell/Terminal neu starten

## Verwendung

### Manuelles Spielen

Starten Sie das GUI mit:
```bash
python main.py
```

**Bedienung:**
- Linksklick: Zelle aufdecken
- Rechtsklick: Flagge setzen/entfernen
- Menü: Neues Spiel mit verschiedenen Schwierigkeitsgraden starten
- Sichere Zellen werden automatisch aufgedeckt

### Reinforcement Learning Training

#### Training mit Visualisierung (GUI)

1. **Anwendung starten:**
   ```bash
   python main.py
   ```

2. **Training starten:**
   - Menü → **Reinforcement Learning** → **Training starten (mit Visualisierung)**
   - Anzahl Episoden eingeben (z.B. 1000)
   - Schwierigkeit wählen (Leicht/Mittel/Schwer)

3. **Visualisierung:**
   - Alle **100 Episoden** wird automatisch eine Visualisierung gestartet
   - Am **Ende** folgt ein finaler Test-Lauf
   - Sie sehen in Echtzeit, wie der Agent spielt!

#### Training über Kommandozeile (ohne GUI)

```bash
python -m src.reinforcement_learning.trainer --episodes 1000 --difficulty medium --save-path models/dqn_model.pth
```

**Parameter:**
- `--episodes`: Anzahl der Trainingsepisoden (Standard: 1000)
- `--difficulty`: Schwierigkeitsgrad - easy, medium, hard (Standard: medium)
- `--save-path`: Pfad zum Speichern des Modells (Standard: models/dqn_model.pth)
- `--log-interval`: Episoden zwischen Logging (Standard: 100)

### Geladenes Modell testen

1. **Modell laden:**
   - Menü → **Reinforcement Learning** → **Modell laden und testen**
   - Wählen Sie die Modell-Datei (`models/dqn_model.pth`)

2. **Agent spielen lassen:**
   - Klicken Sie auf **"RL-Agent spielen lassen"** im RL-Visualizer-Widget
   - Der Agent spielt automatisch eine Episode

**Detaillierte Anleitung:** Siehe [docs/RL_TRAINING_GUIDE.md](docs/RL_TRAINING_GUIDE.md)

## Technische Details

- **Programmiersprache**: Python 3.8+
- **GUI Framework**: PySide6
- **RL Framework**: PyTorch
- **Algorithmus**: Deep Q-Network (DQN) mit Experience Replay und Target Network

## Features

- ✅ Vollständiges Minesweeper-Spiel mit 20x30 Spielfeld
- ✅ 3 Schwierigkeitsgrade (Leicht, Mittel, Schwer)
- ✅ Manuelles Spielen mit intuitiver GUI
- ✅ Automatisches Aufdecken sicherer Zellen
- ✅ DQN-basiertes Reinforcement Learning
- ✅ Training-Script mit Logging
- ✅ Modell-Speicherung und -Laden
- ✅ **RL-Visualisierung während des Trainings (alle 100 Episoden)**
- ✅ **Finaler Test-Lauf am Ende des Trainings**
- ✅ **Manuelles Testen geladener Modelle in der GUI**
- ✅ **Reveal-only Action Space (Flags optional) – inspiriert durch [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning) für stabileres Training**

## Entwicklung

Das Projekt wurde mit Hilfe von Cursor und KI-Assistenz erstellt, um zu testen, wie effektiv KI bei der Programmierung helfen kann.

```

```python
"""Tests for MinesweeperEnvironment."""

import unittest
import numpy as np
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class TestMinesweeperEnvironment(unittest.TestCase):
    """Test cases for MinesweeperEnvironment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MinesweeperEnvironment("medium")
        self.env_no_flags = MinesweeperEnvironment("medium", use_flag_actions=False)
    
    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.difficulty, "medium")
        self.assertEqual(self.env.action_space_size, BOARD_WIDTH * BOARD_HEIGHT * 2)
        self.assertIsNotNone(self.env.game)
        self.assertEqual(self.env_no_flags.action_space_size, BOARD_WIDTH * BOARD_HEIGHT)
    
    def test_reset(self):
        """Test environment reset."""
        state = self.env.reset()
        
        # Check state shape
        self.assertEqual(state.shape, (STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH))
        self.assertEqual(state.dtype, np.float32)
        
        # Check that all cells are hidden initially (value should be -0.9)
        # Actually, first click hasn't happened yet, so state should be all hidden
        self.assertTrue(np.all(state[0] == -0.9))
        self.assertTrue(np.all(state[1] == 1.0))
    
    def test_step_valid_action(self):
        """Test step with valid action."""
        state = self.env.reset()
        
        # First action (should be valid)
        action = 0  # Top-left cell
        next_state, reward, done, info = self.env.step(action)
        
        # Check return types
        self.assertIsInstance(next_state, np.ndarray)
        self.assertIsInstance(reward, (float, np.floating))
        self.assertIsInstance(done, (bool, np.bool_))
        self.assertIsInstance(info, dict)
        
        # Check state shape
        self.assertEqual(next_state.shape, (STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH))
        
        # Check info keys
        self.assertIn("won", info)
        self.assertIn("lost", info)
        self.assertIn("valid_action", info)
    
    def test_step_invalid_action(self):
        """Test step with invalid action (already revealed cell)."""
        state = self.env.reset()
        
        # First action
        action1 = 0
        next_state, reward1, done1, info1 = self.env.step(action1)
        
        # Try same action again (should be invalid)
        next_state2, reward2, done2, info2 = self.env.step(action1)
        
        # Should get negative reward for invalid action
        self.assertLess(reward2, 0)
        self.assertFalse(info2["valid_action"])
    
    def test_state_representation(self):
        """Test state representation encoding."""
        state = self.env.reset()
        
        # Make some moves
        self.env.step(0)
        state = self.env._get_state()
        
        # Check state values are in expected range
        # -1.0 = mine, -0.9 = hidden, -0.5 = flagged, 0-1 = numbers
        self.assertTrue(np.all(state >= -1.0))
        self.assertTrue(np.all(state <= 1.0))
    
    def test_reward_calculation(self):
        """Test reward calculation."""
        state = self.env.reset()
        
        # First action should give reward if cells are revealed
        _, reward, _, _ = self.env.step(0)
        # Reward could be positive (cells revealed) or negative (mine hit)
        self.assertIsInstance(reward, (float, np.floating))
    
    def test_reward_on_win(self):
        """Test reward when game is won."""
        # This is hard to test deterministically, but we can check the structure
        state = self.env.reset()
        
        # Make moves until game ends (if possible)
        # For now, just check that reward calculation handles win case
        # (This would require solving the entire game, which is complex)
        pass
    
    def test_reward_on_loss(self):
        """Test reward when mine is hit."""
        state = self.env.reset()
        
        # Try to find and hit a mine (unlikely but possible)
        # If we hit a mine, reward should be -500.0 (updated reward structure)
        for _ in range(100):  # Try up to 100 actions
            valid_actions = self.env.get_valid_actions()
            if not np.any(valid_actions):
                break
            
            action = np.random.choice(np.where(valid_actions)[0])
            _, reward, done, info = self.env.step(action)
            
            if info["lost"]:
                # Reward is now scaled by board size, so check range instead of exact value
                # For medium difficulty (30x20), should be around -100
                self.assertLess(reward, 0)
                self.assertGreaterEqual(reward, -200.0)  # Allow some flexibility
                break
    
    def test_get_valid_actions(self):
        """Test getting valid actions."""
        state = self.env.reset()
        
        valid_actions = self.env.get_valid_actions()
        
        # Check shape
        self.assertEqual(valid_actions.shape, (BOARD_WIDTH * BOARD_HEIGHT * 2,))
        self.assertEqual(valid_actions.dtype, bool)
        
        cell_count = BOARD_WIDTH * BOARD_HEIGHT
        # Initially all reveal actions should be valid (all cells hidden)
        self.assertTrue(np.all(valid_actions[:cell_count]))
        # Flag actions should also be valid (can flag/unflag hidden cells)
        self.assertTrue(np.all(valid_actions[cell_count:]))
        
        # After first action, some should become invalid
        self.env.step(0)
        valid_actions = self.env.get_valid_actions()
        self.assertFalse(np.all(valid_actions[:cell_count]))  # Some reveals invalid now
        
        # No-flag variant
        self.env_no_flags.reset()
        valid_reveal_only = self.env_no_flags.get_valid_actions()
        self.assertEqual(valid_reveal_only.shape, (BOARD_WIDTH * BOARD_HEIGHT,))
        self.assertTrue(np.all(valid_reveal_only))
    
    def test_get_action_mask(self):
        """Test action masking."""
        state = self.env.reset()
        
        mask = self.env.get_action_mask()
        
        # Check shape
        self.assertEqual(mask.shape, (BOARD_WIDTH * BOARD_HEIGHT * 2,))
        
        # Valid actions should be 0.0, invalid should be -inf
        valid_actions = self.env.get_valid_actions()
        self.assertTrue(np.all(mask[valid_actions] == 0.0))
        self.assertTrue(np.all(np.isinf(mask[~valid_actions])))
        
        mask_no_flags = self.env_no_flags.get_action_mask()
        self.assertEqual(mask_no_flags.shape, (BOARD_WIDTH * BOARD_HEIGHT,))
        valid_nf = self.env_no_flags.get_valid_actions()
        self.assertTrue(np.all(mask_no_flags[valid_nf] == 0.0))
    
    def test_multiple_episodes(self):
        """Test multiple episodes."""
        # Run multiple episodes
        for _ in range(3):
            state = self.env.reset()
            self.assertFalse(self.env.game.is_game_over())
            
            # Make a few moves
            for _ in range(5):
                if self.env.game.is_game_over():
                    break
                valid = self.env.get_valid_actions()
                if np.any(valid):
                    action = np.random.choice(np.where(valid)[0])
                    self.env.step(action)
    
    def test_flag_rewards(self):
        """Test rewards for flagging correct/incorrect cells."""
        self.env.reset()
        # Trigger mine placement
        self.env.step(0)
        
        mine_idx = None
        safe_idx = None
        
        for row in range(self.env.height):
            for col in range(self.env.width):
                cell = self.env.game.board.get_cell(row, col)
                if cell.is_revealed():
                    continue
                idx = row * self.env.width + col
                if cell.is_mine and mine_idx is None:
                    mine_idx = idx
                elif not cell.is_mine and safe_idx is None:
                    safe_idx = idx
            if mine_idx is not None and safe_idx is not None:
                break
        
        self.assertIsNotNone(mine_idx)
        self.assertIsNotNone(safe_idx)
        
        # Flag a mine -> positive reward
        mine_flag_action = mine_idx + self.env.cell_count
        _, reward_correct, _, _ = self.env.step(mine_flag_action)
        self.assertGreater(reward_correct, 0)
        
        # Flag a safe cell -> penalty
        safe_flag_action = safe_idx + self.env.cell_count
        _, reward_wrong, _, _ = self.env.step(safe_flag_action)
        self.assertLess(reward_wrong, 0)


if __name__ == "__main__":
    unittest.main()


```

```python
"""Tests for DQNNetwork."""

import unittest
import torch
from src.reinforcement_learning.network import DQNNetwork
from src.reinforcement_learning.environment import STATE_CHANNELS
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class TestDQNNetwork(unittest.TestCase):
    """Test cases for DQNNetwork."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_channels = STATE_CHANNELS
        self.num_actions = BOARD_WIDTH * BOARD_HEIGHT
        self.network = DQNNetwork(self.input_channels, self.num_actions)
    
    def test_initialization(self):
        """Test network initialization."""
        self.assertEqual(self.network.num_actions, self.num_actions)
        self.assertIsNotNone(self.network.conv_stack)
        self.assertEqual(len(self.network.conv_stack), 12)  # 4 * (Conv+BN+ReLU)
        self.assertIsNotNone(self.network.fc1)
        self.assertIsNotNone(self.network.fc2)
        self.assertIsNotNone(self.network.fc_out)
    
    def test_forward_pass_single(self):
        """Test forward pass with single sample."""
        # Create dummy input (batch_size=1, channels=1, height, width)
        batch_size = 1
        x = torch.randn(batch_size, self.input_channels, BOARD_HEIGHT, BOARD_WIDTH)
        
        # Forward pass
        output = self.network(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.num_actions))
        
        # Check output values are finite
        self.assertTrue(torch.all(torch.isfinite(output)))
    
    def test_forward_pass_batch(self):
        """Test forward pass with batch."""
        batch_size = 32
        x = torch.randn(batch_size, self.input_channels, BOARD_HEIGHT, BOARD_WIDTH)
        
        output = self.network(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.num_actions))
        self.assertTrue(torch.all(torch.isfinite(output)))
    
    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 4, 16, 64]:
            x = torch.randn(batch_size, self.input_channels, BOARD_HEIGHT, BOARD_WIDTH)
            output = self.network(x)
            self.assertEqual(output.shape, (batch_size, self.num_actions))
    
    def test_gradient_flow(self):
        """Test that gradients can flow through the network."""
        x = torch.randn(1, self.input_channels, BOARD_HEIGHT, BOARD_WIDTH, requires_grad=True)
        output = self.network(x)
        
        # Create dummy loss
        loss = output.mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(x.grad)
        
        # Check that network parameters have gradients
        for param in self.network.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_network_parameters_count(self):
        """Test that network has reasonable number of parameters."""
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
        # Should have millions of parameters for this network
        self.assertGreater(total_params, 100000)
        self.assertEqual(total_params, trainable_params)
    
    def test_network_eval_mode(self):
        """Test network in eval mode."""
        self.network.eval()
        
        x = torch.randn(1, self.input_channels, BOARD_HEIGHT, BOARD_WIDTH)
        
        with torch.no_grad():
            output = self.network(x)
        
        self.assertEqual(output.shape, (1, self.num_actions))


if __name__ == "__main__":
    unittest.main()


```

## Ergebnis
- Angepasstes `DQNNetwork` folgt nun dem 4×128-Convolution-/512×512-Dense-Aufbau aus [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning); die Adaptive-Pooling-Stufe hält alle Brettgrößen konsistent und senkt Overfitting durch Dropout.  
```50:74:src/reinforcement_learning/network.py
        for _ in range(4):
            conv_layers.extend([
                nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ])
...
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc_out(x)
```
- Das Environment bietet jetzt optional einen reinen Reveal-Aktionsraum (`use_flag_actions=False`). Das reduziert den Aktionsraum standardmäßig auf `width × height` und repliziert damit den Progress/Gues-Ansatz aus [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning). Flags lassen sich für Experimente via `--use-flags` wieder zuschalten; GUI, Trainer und Visualizer synchronisieren diese Einstellung automatisch.  
```25:115:src/reinforcement_learning/environment.py
    def __init__(..., use_flag_actions: bool = True):
        ...
        self.use_flag_actions = use_flag_actions
        self.action_space_size = (
            self.cell_count if not use_flag_actions else self.cell_count * 2
        )
...
        if not self.use_flag_actions:
            return valid_reveal
        return np.concatenate([valid_reveal, valid_flag])
```
- `RLVisualizer` erkennt anhand der Modelldaten, ob Flags trainiert wurden, erzeugt das Environment entsprechend und vermeidet so Dimensionskonflikte. Das Laden externer Modelle liest nun Metadaten direkt aus dem Checkpoint (inkl. Boardgröße & Aktionsraum) und richtet das GUI-Board automatisch passend ein.
- Dokumentation/README erläutern den neuen Reveal-only-Default, die sdlee94-inspirierte Architektur und den neuen CLI-Schalter; Tests decken beide Aktionsraum-Varianten ab.

## Tests
- `python -m pytest tests/reinforcement_learning`

Gib Bescheid, falls wir weitere Hyperparameter ausloten oder Transfer-Learning wie im Referenzprojekt ausprobieren sollen.

```python
        for _ in range(4):
            conv_layers.extend([
                nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ])
...
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc_out(x)
```

```python
    def __init__(..., use_flag_actions: bool = True):
        ...
        self.use_flag_actions = use_flag_actions
        self.action_space_size = (
            self.cell_count if not use_flag_actions else self.cell_count * 2
        )
...
        if not self.use_flag_actions:
            return valid_reveal
        return np.concatenate([valid_reveal, valid_flag])
```

---

**User**

@powershell (703-776) Das sind die Resultate, es sieht alles mehr nach zufall aus, als das das modell tatsächlich lernt. Verbessere bitte den kompletten RL und schaue in jedes dokument, ob es irgendwo einen fehler geben könnte der geändert werden muss

---

**Cursor**

```python
"""RL Environment wrapper for Minesweeper."""

import numpy as np
from typing import Tuple, Optional
from src.minesweeper.game import Game
from src.utils.constants import (
    BOARD_WIDTH, BOARD_HEIGHT,
    CELL_HIDDEN, CELL_REVEALED, CELL_FLAGGED, CELL_MINE
)

STATE_CHANNELS = 7
NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]

FLAG_SET_REWARD = 0.2
FLAG_SET_PENALTY = 0.2
FLAG_FINAL_CORRECT_BONUS = 0.5
FLAG_FINAL_INCORRECT_PENALTY = 0.5


class MinesweeperEnvironment:
    """Environment wrapper for reinforcement learning."""
    
    def __init__(
        self,
        difficulty: str = "medium",
        width: int = BOARD_WIDTH,
        height: int = BOARD_HEIGHT,
        use_flag_actions: bool = False
    ):
        """
        Initialize the environment.
        
        Args:
            difficulty: Game difficulty level
            width: Board width
            height: Board height
        """
        self.difficulty = difficulty
        self.width = width
        self.height = height
        self.game = Game(difficulty, width, height)
        self.cell_count = width * height
        self.use_flag_actions = use_flag_actions
        self.action_space_size = (
            self.cell_count if not use_flag_actions else self.cell_count * 2
        )  # Reveal (+ optional Flag) per cell
        self.state_channels = STATE_CHANNELS
        self.reward_scale = max(1.0, (self.width * self.height) / 100.0)
        self.progress_scale = max(0.5, (self.width * self.height) / 300.0)
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment and return initial state.
        
        Returns:
            Initial state observation
        """
        self.game.new_game(self.difficulty, self.width, self.height)
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action index (0 to action_space_size - 1)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Convert action to row, col
        cell_index = action % self.cell_count if self.use_flag_actions and action >= self.cell_count else action
        row = cell_index // self.width
        col = cell_index % self.width
        action_type = "flag" if (self.use_flag_actions and action >= self.cell_count) else "reveal"
        
        # Get previous state
        prev_revealed = self.game.revealed_count
        reward = 0.0
        valid = False
        
        move_context = self._get_move_context(row, col)
        
        if action_type == "reveal":
            valid = self.game.reveal_cell(row, col)
            reward += self._calculate_reward(
                prev_revealed,
                valid,
                action_type=action_type,
                move_context=move_context
            )
        else:
            valid, flag_reward = self._handle_flag_action(row, col)
            reward += flag_reward
        
        # Check if done
        done = self.game.is_game_over()
        
        # Get next state
        next_state = self._get_state()
        
        # Info
        info = {
            "won": self.game.is_won(),
            "lost": self.game.is_lost(),
            "valid_action": valid
        }
        
        # Additional rewards/penalties when the game ends after this move
        if done and self.use_flag_actions and self.game.is_lost():
            reward += self._calculate_flag_end_bonus()
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation with better features.
        
        Returns:
            State array of shape (STATE_CHANNELS, height, width)
        """
        state = np.zeros((STATE_CHANNELS, self.height, self.width), dtype=np.float32)
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                adjacent_revealed = 0
                adjacent_hint_sum = 0
                hidden_neighbors = 0
                flagged_neighbors = 0
                
                for dr, dc in NEIGHBOR_OFFSETS:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        neighbor = self.game.board.get_cell(nr, nc)
                        if neighbor.is_hidden():
                            hidden_neighbors += 1
                        elif neighbor.is_flagged():
                            flagged_neighbors += 1
                        elif neighbor.is_revealed():
                            adjacent_revealed += 1
                            if not neighbor.is_mine:
                                adjacent_hint_sum += neighbor.adjacent_mines
                
                if cell.is_revealed():
                    if cell.is_mine:
                        state[0, row, col] = -1.0
                        state[3, row, col] = -1.0
                    else:
                        value = cell.adjacent_mines / 8.0
                        state[0, row, col] = value
                        state[3, row, col] = value
                elif cell.is_flagged():
                    state[0, row, col] = -0.5
                    state[2, row, col] = 1.0
                else:
                    neighbor_bonus = min(0.4, adjacent_revealed * 0.08)
                    hint_penalty = min(0.3, adjacent_hint_sum / 30.0)
                    cell_value = -0.9 + neighbor_bonus - hint_penalty
                    state[0, row, col] = max(-1.0, cell_value)
                    state[1, row, col] = 1.0
                
                state[4, row, col] = hidden_neighbors / 8.0
                state[5, row, col] = flagged_neighbors / 8.0
                state[6, row, col] = min(1.0, adjacent_hint_sum / 64.0)
        
        return state
    
    def _calculate_reward(
        self,
        prev_revealed: int,
        valid_action: bool,
        action_type: str,
        move_context: Optional[dict] = None
    ) -> float:
        """
        Calculate reward for the action.
        
        Args:
            prev_revealed: Number of revealed cells before action
            valid_action: Whether the action was valid
            
        Returns:
            Reward value
        """
        if action_type != "reveal":
            return 0.0
        
        if not valid_action:
            return -0.6 * self.reward_scale  # Penalty for invalid action
        
        board_scale = self.reward_scale
        cells_revealed = self.game.revealed_count - prev_revealed
        progress_ratio = self.game.revealed_count / max(1, self.game.total_safe_cells)
        is_guess = move_context.get("adjacent_revealed", 0) == 0 if move_context else True
        frontier_neighbors = move_context.get("adjacent_revealed", 0) if move_context else 0
        frontier_factor = frontier_neighbors / 8.0
        frontier_bonus = 0.0
        
        if move_context:
            if is_guess:
                frontier_bonus = -0.4 * board_scale
            else:
                frontier_bonus = 0.6 * (1.0 + frontier_factor)
        
        if self.game.is_lost():
            # Keep penalty strong enough to dominate random play
            penalty = -12.0 * board_scale
            return penalty + (frontier_bonus if is_guess else 0.0)
        
        if self.game.is_won():
            return 18.0 * board_scale + 6.0 * progress_ratio
        
        if cells_revealed > 0:
            base_reward = self.progress_scale * cells_revealed
            chain_bonus = 0.4 * self.progress_scale * max(0, cells_revealed - 1)
            stability_bonus = 0.3 * (0.0 if is_guess else 1.0)
            shaping = 0.4 * frontier_factor
            return base_reward + chain_bonus + stability_bonus + shaping
        
        # Slight penalty for ineffective move
        return -0.15 * self.progress_scale
    
    def _handle_flag_action(self, row: int, col: int) -> Tuple[bool, float]:
        """
        Handle flag toggling and return (valid, reward).
        """
        cell = self.game.board.get_cell(row, col)
        if not cell or cell.is_revealed():
            return False, -0.1
        
        toggled = self.game.toggle_flag(row, col)
        if not toggled:
            return False, -0.1
        
        if cell.is_flagged():
            if cell.is_mine:
                return True, FLAG_SET_REWARD
            return True, -FLAG_SET_PENALTY
        
        # Flag removed (no immediate reward to avoid oscillation)
        return True, 0.0
    
    def _calculate_flag_end_bonus(self) -> float:
        """Reward/punish based on flagged cells when the game ends."""
        correct_flags = 0
        incorrect_flags = 0
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                if cell and cell.is_flagged():
                    if cell.is_mine:
                        correct_flags += 1
                    else:
                        incorrect_flags += 1
        
        return (
            correct_flags * FLAG_FINAL_CORRECT_BONUS
            - incorrect_flags * FLAG_FINAL_INCORRECT_PENALTY
        )
    
    def get_valid_actions(self) -> np.ndarray:
        """
        Get mask of valid actions (cells that can be clicked).
        
        Returns:
            Boolean array of valid actions
        """
        valid_reveal = np.zeros(self.cell_count, dtype=bool)
        valid_flag = np.zeros(self.cell_count, dtype=bool) if self.use_flag_actions else None
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                action_idx = row * self.width + col
                
                # Reveal valid if cell hidden and not flagged
                if cell.is_hidden() and not cell.is_flagged():
                    valid_reveal[action_idx] = True
                
                # Flag valid if cell hidden or currently flagged (toggle)
                if self.use_flag_actions and not cell.is_revealed():
                    if cell.is_hidden() or cell.is_flagged():
                        valid_flag[action_idx] = True
        
        if not self.use_flag_actions:
            return valid_reveal
        
        return np.concatenate([valid_reveal, valid_flag])
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get action mask for masking invalid actions in Q-values.
        
        Returns:
            Mask array (1.0 for valid, -inf for invalid)
        """
        valid_actions = self.get_valid_actions()
        mask = np.where(valid_actions, 0.0, -np.inf)
        return mask

    def _get_move_context(self, row: int, col: int) -> dict:
        """Collect contextual information for the selected move."""
        context = {
            "adjacent_revealed": 0,
            "adjacent_hidden": 0,
            "adjacent_flagged": 0
        }
        
        for dr, dc in NEIGHBOR_OFFSETS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                neighbor = self.game.board.get_cell(nr, nc)
                if not neighbor:
                    continue
                if neighbor.is_revealed():
                    context["adjacent_revealed"] += 1
                elif neighbor.is_flagged():
                    context["adjacent_flagged"] += 1
                elif neighbor.is_hidden():
                    context["adjacent_hidden"] += 1
        
        return context



```

```python
"""RL Environment wrapper for Minesweeper."""

import numpy as np
from typing import Tuple, Optional
from src.minesweeper.game import Game
from src.utils.constants import (
    BOARD_WIDTH, BOARD_HEIGHT,
    CELL_HIDDEN, CELL_REVEALED, CELL_FLAGGED, CELL_MINE
)

STATE_CHANNELS = 7
NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]

FLAG_SET_REWARD = 0.2
FLAG_SET_PENALTY = 0.2
FLAG_FINAL_CORRECT_BONUS = 0.5
FLAG_FINAL_INCORRECT_PENALTY = 0.5


class MinesweeperEnvironment:
    """Environment wrapper for reinforcement learning."""
    
    def __init__(
        self,
        difficulty: str = "medium",
        width: int = BOARD_WIDTH,
        height: int = BOARD_HEIGHT,
        use_flag_actions: bool = False
    ):
        """
        Initialize the environment.
        
        Args:
            difficulty: Game difficulty level
            width: Board width
            height: Board height
        """
        self.difficulty = difficulty
        self.width = width
        self.height = height
        self.game = Game(difficulty, width, height)
        self.cell_count = width * height
        self.use_flag_actions = use_flag_actions
        self.action_space_size = (
            self.cell_count if not use_flag_actions else self.cell_count * 2
        )  # Reveal (+ optional Flag) per cell
        self.state_channels = STATE_CHANNELS
        self.reward_scale = max(1.0, (self.width * self.height) / 100.0)
        self.progress_scale = max(0.5, (self.width * self.height) / 300.0)
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment and return initial state.
        
        Returns:
            Initial state observation
        """
        self.game.new_game(self.difficulty, self.width, self.height)
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action index (0 to action_space_size - 1)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Convert action to row, col
        cell_index = action % self.cell_count if self.use_flag_actions and action >= self.cell_count else action
        row = cell_index // self.width
        col = cell_index % self.width
        action_type = "flag" if (self.use_flag_actions and action >= self.cell_count) else "reveal"
        
        # Get previous state
        prev_revealed = self.game.revealed_count
        reward = 0.0
        valid = False
        
        move_context = self._get_move_context(row, col)
        
        if action_type == "reveal":
            valid = self.game.reveal_cell(row, col)
            reward += self._calculate_reward(
                prev_revealed,
                valid,
                action_type=action_type,
                move_context=move_context
            )
        else:
            valid, flag_reward = self._handle_flag_action(row, col)
            reward += flag_reward
        
        # Check if done
        done = self.game.is_game_over()
        
        # Get next state
        next_state = self._get_state()
        
        # Info
        info = {
            "won": self.game.is_won(),
            "lost": self.game.is_lost(),
            "valid_action": valid
        }
        
        # Additional rewards/penalties when the game ends after this move
        if done and self.use_flag_actions and self.game.is_lost():
            reward += self._calculate_flag_end_bonus()
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation with better features.
        
        Returns:
            State array of shape (STATE_CHANNELS, height, width)
        """
        state = np.zeros((STATE_CHANNELS, self.height, self.width), dtype=np.float32)
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                adjacent_revealed = 0
                adjacent_hint_sum = 0
                hidden_neighbors = 0
                flagged_neighbors = 0
                
                for dr, dc in NEIGHBOR_OFFSETS:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        neighbor = self.game.board.get_cell(nr, nc)
                        if neighbor.is_hidden():
                            hidden_neighbors += 1
                        elif neighbor.is_flagged():
                            flagged_neighbors += 1
                        elif neighbor.is_revealed():
                            adjacent_revealed += 1
                            if not neighbor.is_mine:
                                adjacent_hint_sum += neighbor.adjacent_mines
                
                if cell.is_revealed():
                    if cell.is_mine:
                        state[0, row, col] = -1.0
                        state[3, row, col] = -1.0
                    else:
                        value = cell.adjacent_mines / 8.0
                        state[0, row, col] = value
                        state[3, row, col] = value
                elif cell.is_flagged():
                    state[0, row, col] = -0.5
                    state[2, row, col] = 1.0
                else:
                    neighbor_bonus = min(0.4, adjacent_revealed * 0.08)
                    hint_penalty = min(0.3, adjacent_hint_sum / 30.0)
                    cell_value = -0.9 + neighbor_bonus - hint_penalty
                    state[0, row, col] = max(-1.0, cell_value)
                    state[1, row, col] = 1.0
                
                state[4, row, col] = hidden_neighbors / 8.0
                state[5, row, col] = flagged_neighbors / 8.0
                state[6, row, col] = min(1.0, adjacent_hint_sum / 64.0)
        
        return state
    
    def _calculate_reward(
        self,
        prev_revealed: int,
        valid_action: bool,
        action_type: str,
        move_context: Optional[dict] = None
    ) -> float:
        """
        Calculate reward for the action.
        
        Args:
            prev_revealed: Number of revealed cells before action
            valid_action: Whether the action was valid
            
        Returns:
            Reward value
        """
        if action_type != "reveal":
            return 0.0
        
        if not valid_action:
            return -0.6 * self.reward_scale  # Penalty for invalid action
        
        board_scale = self.reward_scale
        cells_revealed = self.game.revealed_count - prev_revealed
        progress_ratio = self.game.revealed_count / max(1, self.game.total_safe_cells)
        is_guess = move_context.get("adjacent_revealed", 0) == 0 if move_context else True
        frontier_neighbors = move_context.get("adjacent_revealed", 0) if move_context else 0
        frontier_factor = frontier_neighbors / 8.0
        frontier_bonus = 0.0
        
        if move_context:
            if is_guess:
                frontier_bonus = -0.4 * board_scale
            else:
                frontier_bonus = 0.6 * (1.0 + frontier_factor)
        
        if self.game.is_lost():
            # Keep penalty strong enough to dominate random play
            penalty = -12.0 * board_scale
            return penalty + (frontier_bonus if is_guess else 0.0)
        
        if self.game.is_won():
            return 18.0 * board_scale + 6.0 * progress_ratio
        
        if cells_revealed > 0:
            base_reward = self.progress_scale * cells_revealed
            chain_bonus = 0.4 * self.progress_scale * max(0, cells_revealed - 1)
            stability_bonus = 0.3 * (0.0 if is_guess else 1.0)
            shaping = 0.4 * frontier_factor
            return base_reward + chain_bonus + stability_bonus + shaping + 0.25 * frontier_bonus
        
        # Slight penalty for ineffective move
        return -0.15 * self.progress_scale
    
    def _handle_flag_action(self, row: int, col: int) -> Tuple[bool, float]:
        """
        Handle flag toggling and return (valid, reward).
        """
        cell = self.game.board.get_cell(row, col)
        if not cell or cell.is_revealed():
            return False, -0.1
        
        toggled = self.game.toggle_flag(row, col)
        if not toggled:
            return False, -0.1
        
        if cell.is_flagged():
            if cell.is_mine:
                return True, FLAG_SET_REWARD
            return True, -FLAG_SET_PENALTY
        
        # Flag removed (no immediate reward to avoid oscillation)
        return True, 0.0
    
    def _calculate_flag_end_bonus(self) -> float:
        """Reward/punish based on flagged cells when the game ends."""
        correct_flags = 0
        incorrect_flags = 0
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                if cell and cell.is_flagged():
                    if cell.is_mine:
                        correct_flags += 1
                    else:
                        incorrect_flags += 1
        
        return (
            correct_flags * FLAG_FINAL_CORRECT_BONUS
            - incorrect_flags * FLAG_FINAL_INCORRECT_PENALTY
        )
    
    def get_valid_actions(self) -> np.ndarray:
        """
        Get mask of valid actions (cells that can be clicked).
        
        Returns:
            Boolean array of valid actions
        """
        valid_reveal = np.zeros(self.cell_count, dtype=bool)
        valid_flag = np.zeros(self.cell_count, dtype=bool) if self.use_flag_actions else None
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                action_idx = row * self.width + col
                
                # Reveal valid if cell hidden and not flagged
                if cell.is_hidden() and not cell.is_flagged():
                    valid_reveal[action_idx] = True
                
                # Flag valid if cell hidden or currently flagged (toggle)
                if self.use_flag_actions and not cell.is_revealed():
                    if cell.is_hidden() or cell.is_flagged():
                        valid_flag[action_idx] = True
        
        if not self.use_flag_actions:
            return valid_reveal
        
        return np.concatenate([valid_reveal, valid_flag])
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get action mask for masking invalid actions in Q-values.
        
        Returns:
            Mask array (1.0 for valid, -inf for invalid)
        """
        valid_actions = self.get_valid_actions()
        mask = np.where(valid_actions, 0.0, -np.inf)
        return mask

    def _get_move_context(self, row: int, col: int) -> dict:
        """Collect contextual information for the selected move."""
        context = {
            "adjacent_revealed": 0,
            "adjacent_hidden": 0,
            "adjacent_flagged": 0
        }
        
        for dr, dc in NEIGHBOR_OFFSETS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                neighbor = self.game.board.get_cell(nr, nc)
                if not neighbor:
                    continue
                if neighbor.is_revealed():
                    context["adjacent_revealed"] += 1
                elif neighbor.is_flagged():
                    context["adjacent_flagged"] += 1
                elif neighbor.is_hidden():
                    context["adjacent_hidden"] += 1
        
        return context



```

```python
"""DQN Agent for Minesweeper."""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Tuple, Optional
from src.reinforcement_learning.network import DQNNetwork
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_valid_actions: np.ndarray
    ):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done, next_valid_actions))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_valid_actions = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        next_valid_actions = torch.BoolTensor(np.array(next_valid_actions))
        
        return states, actions, rewards, next_states, dones, next_valid_actions
    
    def __len__(self):
        """Get buffer size."""
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network Agent."""
    
    def __init__(
        self,
        state_channels: int = STATE_CHANNELS,
        action_space_size: int = 600,
        board_height: int = 20,
        board_width: int = 30,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update: int = 100,
        device: Optional[torch.device] = None
    ):
        """
        Initialize DQN Agent.
        
        Args:
            state_channels: Number of input channels
            action_space_size: Size of action space
            board_height: Height of the board
            board_width: Width of the board
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Initial epsilon for epsilon-greedy
            epsilon_end: Final epsilon
            epsilon_decay: Epsilon decay rate
            buffer_size: Replay buffer size
            batch_size: Batch size for training
            target_update: Steps between target network updates
            device: PyTorch device
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space_size = action_space_size
        self.board_height = board_height
        self.board_width = board_width
        self.cell_count = board_height * board_width
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        
        # Networks
        self.q_network = DQNNetwork(state_channels, action_space_size, board_height, board_width).to(self.device)
        self.target_network = DQNNetwork(state_channels, action_space_size, board_height, board_width).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.q_network.train()  # Start in training mode
        self.target_network.eval()  # Target network always in eval mode
        
        # Optimizer & loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        self._neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                                  (0, -1),           (0, 1),
                                  (1, -1),  (1, 0),  (1, 1)]
    
    def select_action(self, state: np.ndarray, valid_actions: Optional[np.ndarray] = None) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            valid_actions: Boolean array of valid actions
            
        Returns:
            Selected action index
        """
        if random.random() < self.epsilon:
            if valid_actions is not None:
                frontier_action = self._sample_frontier_action(state, valid_actions)
                if frontier_action is not None:
                    return frontier_action
                
                # Prefer reveal actions during random exploration
                reveal_mask = valid_actions[:self.cell_count]
                reveal_indices = np.where(reveal_mask)[0]
                if reveal_indices.size > 0:
                    return int(random.choice(reveal_indices.tolist()))
                
                valid_indices = np.where(valid_actions)[0]
                if len(valid_indices) > 0:
                    return int(random.choice(valid_indices.tolist()))
            return random.randint(0, self.action_space_size - 1)
        
        # Greedy action
        self.q_network.eval()  # Set to eval mode for inference
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            
            # Mask invalid actions
            if valid_actions is not None:
                mask = torch.FloatTensor(valid_actions).to(self.device)
                q_values = q_values + (1 - mask) * -1e9
            
            action = q_values.argmax().item()
        
        return action
    
    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_valid_actions: np.ndarray
    ):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done, next_valid_actions)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Set network to training mode
        self.q_network.train()
        
        # Sample batch
        states, actions, rewards, next_states, dones, next_valid_actions = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        next_valid_actions = next_valid_actions.to(self.device)
        
        # Current Q values
        q_values = self.q_network(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN target calculation
        with torch.no_grad():
            next_q_online = self.q_network(next_states)
            next_q_online = next_q_online.masked_fill(~next_valid_actions, -1e9)
            best_next_actions = next_q_online.argmax(dim=1, keepdim=True)
            
            next_q_target = self.target_network(next_states)
            target_q_values = next_q_target.gather(1, best_next_actions).squeeze(1)
            target_q = rewards + (1 - dones.float()) * self.gamma * target_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def _sample_frontier_action(self, state: np.ndarray, valid_actions: np.ndarray) -> Optional[int]:
        """Sample an action near the information frontier for smarter exploration."""
        if valid_actions is None or not np.any(valid_actions):
            return None
        
        if state is None:
            return None
        
        if state.ndim == 4:
            # Remove batch dimension if present
            state = state.squeeze(0)
        
        if state.ndim != 3 or state.shape[0] < 2:
            return None
        
        height, width = state.shape[1], state.shape[2]
        cell_count = self.cell_count
        if valid_actions.shape[0] < cell_count:
            return None
        
        reveal_valid = valid_actions[:cell_count]
        reveal_matrix = reveal_valid.reshape(height, width)
        hidden_mask = state[1] > 0.5
        flagged_mask = state[2] > 0.5 if state.shape[0] > 2 else np.zeros_like(hidden_mask, dtype=bool)
        revealed_mask = ~(hidden_mask | flagged_mask)
        
        frontier_mask = np.zeros((height, width), dtype=bool)
        
        for row in range(height):
            for col in range(width):
                if not hidden_mask[row, col] or not reveal_matrix[row, col]:
                    continue
                for dr, dc in self._neighbor_offsets:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < height and 0 <= nc < width and revealed_mask[nr, nc]:
                        frontier_mask[row, col] = True
                        break
        
        frontier_indices = np.where(frontier_mask.reshape(-1))[0]
        if frontier_indices.size > 0:
            return int(random.choice(frontier_indices.tolist()))
        
        # Fall back to cells with strongest hint information if available
        if state.shape[0] > 6:
            hint_channel = state[6].reshape(-1)
            valid_indices = np.where(reveal_valid)[0]
            if valid_indices.size > 0:
                best_idx = valid_indices[np.argmax(hint_channel[valid_indices])]
                return int(best_idx)
        
        return None
    
    def save(self, filepath: str):
        """Save model to file."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'board_height': self.board_height,
            'board_width': self.board_width,
            'action_space_size': self.action_space_size
        }, filepath)
    
    def load(self, filepath: str):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Check if board size matches
        saved_height = checkpoint.get('board_height', None)
        saved_width = checkpoint.get('board_width', None)
        saved_action_size = checkpoint.get('action_space_size', None)
        
        if saved_height is not None and saved_width is not None:
            if saved_height != self.board_height or saved_width != self.board_width:
                raise ValueError(
                    f"Model was trained for board size {saved_width}x{saved_height}, "
                    f"but current size is {self.board_width}x{self.board_height}. "
                    f"Please load the model with matching board size."
                )
        
        if saved_action_size is not None and saved_action_size != self.action_space_size:
            raise ValueError(
                f"Model was trained for action space size {saved_action_size}, "
                f"but current size is {self.action_space_size}. "
                f"Please load the model with matching board size."
            )
        
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.q_network.eval()
        self.target_network.eval()



```

```python
"""Training script for DQN agent."""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from src.reinforcement_learning.dqn_agent import DQNAgent
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
    agent: DQNAgent,
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
            action = agent.select_action(state, valid_actions)
            state, reward, done, info = eval_env.step(action)
            reward_sum += reward
            steps += 1
        
        if info["won"]:
            wins += 1
        total_rewards.append(reward_sum)
        total_lengths.append(steps)
    
    agent.epsilon = original_epsilon
    
    return {
        "episodes": episodes,
        "win_rate": (wins / episodes) * 100.0,
        "avg_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "avg_length": float(np.mean(total_lengths)) if total_lengths else 0.0
    }


def train_with_visualization(
    episodes: int = 1000,
    difficulty: str = "medium",
    width: int = None,
    height: int = None,
    save_path: str = "models/dqn_model.pth",
    log_interval: int = 100,
    visualization_callback: callable = None,
    thread=None,
    use_flag_actions: bool = False
):
    """
    Train DQN agent on Minesweeper with visualization support.
    
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
    
    # Adjust hyperparameters based on board size
    # Smaller boards need faster learning, larger boards need more exploration
    board_size_factor = (width * height) / 600.0  # Normalize to default 30x20
    
    # Learning rate scaled by board size (smaller boards learn faster)
    base_lr = 0.001
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
    
    # Batch size tuned per board size
    batch_size = min(96, max(32, int(48 * (1.0 / max(board_size_factor, 0.6)))))
    
    agent = DQNAgent(
        state_channels=env.state_channels,
        action_space_size=env.action_space_size,
        board_height=height,
        board_width=width,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_floor,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    episode_wins = []  # Track wins for each episode
    
    print(f"Starting training for {episodes} episodes...")
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
            
            # Select action
            action = agent.select_action(state, valid_actions)
            
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
        
        # Episode finished: decay epsilon once
        agent.decay_epsilon()
        
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
    use_flag_actions: bool = False
):
    """
    Train DQN agent on Minesweeper (without visualization).
    
    Args:
        episodes: Number of training episodes
        difficulty: Game difficulty
        save_path: Path to save model
        log_interval: Episodes between logging
    """
    train_with_visualization(
        episodes=episodes,
        difficulty=difficulty,
        save_path=save_path,
        log_interval=log_interval,
        visualization_callback=None,
        thread=None,
        use_flag_actions=use_flag_actions
    )


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train DQN agent on Minesweeper")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--difficulty", type=str, default="medium", 
                       choices=["easy", "medium", "hard"], help="Game difficulty")
    parser.add_argument("--save-path", type=str, default="models/dqn_model.pth",
                       help="Path to save model")
    parser.add_argument("--log-interval", type=int, default=100,
                       help="Episodes between logging")
    parser.add_argument("--use-flags", action="store_true",
                       help="Enable flagging actions in the action space")
    
    args = parser.parse_args()
    
    train(
        episodes=args.episodes,
        difficulty=args.difficulty,
        save_path=args.save_path,
        log_interval=args.log_interval,
        use_flag_actions=args.use_flags
    )


if __name__ == "__main__":
    main()

```

```python
"""Training script for DQN agent."""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from src.reinforcement_learning.dqn_agent import DQNAgent
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
    agent: DQNAgent,
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
            action = agent.select_action(state, valid_actions)
            state, reward, done, info = eval_env.step(action)
            reward_sum += reward
            steps += 1
        
        if info["won"]:
            wins += 1
        total_rewards.append(reward_sum)
        total_lengths.append(steps)
    
    agent.epsilon = original_epsilon
    
    return {
        "episodes": episodes,
        "win_rate": (wins / episodes) * 100.0,
        "avg_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "avg_length": float(np.mean(total_lengths)) if total_lengths else 0.0
    }


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
    eval_episodes: int = 20
):
    """
    Train DQN agent on Minesweeper with visualization support.
    
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
    
    # Adjust hyperparameters based on board size
    # Smaller boards need faster learning, larger boards need more exploration
    board_size_factor = (width * height) / 600.0  # Normalize to default 30x20
    
    # Learning rate scaled by board size (smaller boards learn faster)
    base_lr = 0.001
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
    epsilon_schedule = LinearSchedule(epsilon_start, epsilon_floor, decay_span)
    
    # Batch size tuned per board size
    batch_size = min(96, max(32, int(48 * (1.0 / max(board_size_factor, 0.6)))))
    
    agent = DQNAgent(
        state_channels=env.state_channels,
        action_space_size=env.action_space_size,
        board_height=height,
        board_width=width,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_floor,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    episode_wins = []  # Track wins for each episode
    
    print(f"Starting training for {episodes} episodes...")
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
            
            # Select action
            action = agent.select_action(state, valid_actions)
            
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
    use_flag_actions: bool = False
):
    """
    Train DQN agent on Minesweeper (without visualization).
    
    Args:
        episodes: Number of training episodes
        difficulty: Game difficulty
        save_path: Path to save model
        log_interval: Episodes between logging
    """
    train_with_visualization(
        episodes=episodes,
        difficulty=difficulty,
        save_path=save_path,
        log_interval=log_interval,
        visualization_callback=None,
        thread=None,
        use_flag_actions=use_flag_actions
    )


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train DQN agent on Minesweeper")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--difficulty", type=str, default="medium", 
                       choices=["easy", "medium", "hard"], help="Game difficulty")
    parser.add_argument("--save-path", type=str, default="models/dqn_model.pth",
                       help="Path to save model")
    parser.add_argument("--log-interval", type=int, default=100,
                       help="Episodes between logging")
    parser.add_argument("--use-flags", action="store_true",
                       help="Enable flagging actions in the action space")
    
    args = parser.parse_args()
    
    train(
        episodes=args.episodes,
        difficulty=args.difficulty,
        save_path=args.save_path,
        log_interval=args.log_interval,
        use_flag_actions=args.use_flags
    )


if __name__ == "__main__":
    main()

```

```python
"""Training script for DQN agent."""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from src.reinforcement_learning.dqn_agent import DQNAgent
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
    agent: DQNAgent,
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
            action = agent.select_action(state, valid_actions)
            state, reward, done, info = eval_env.step(action)
            reward_sum += reward
            steps += 1
        
        if info["won"]:
            wins += 1
        total_rewards.append(reward_sum)
        total_lengths.append(steps)
    
    agent.epsilon = original_epsilon
    
    return {
        "episodes": episodes,
        "win_rate": (wins / episodes) * 100.0,
        "avg_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "avg_length": float(np.mean(total_lengths)) if total_lengths else 0.0
    }


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
    eval_episodes: int = 20
):
    """
    Train DQN agent on Minesweeper with visualization support.
    
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
    
    # Adjust hyperparameters based on board size
    # Smaller boards need faster learning, larger boards need more exploration
    board_size_factor = (width * height) / 600.0  # Normalize to default 30x20
    
    # Learning rate scaled by board size (smaller boards learn faster)
    base_lr = 0.001
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
    epsilon_schedule = LinearSchedule(epsilon_start, epsilon_floor, decay_span)
    
    # Batch size tuned per board size
    batch_size = min(96, max(32, int(48 * (1.0 / max(board_size_factor, 0.6)))))
    
    agent = DQNAgent(
        state_channels=env.state_channels,
        action_space_size=env.action_space_size,
        board_height=height,
        board_width=width,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_floor,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    episode_wins = []  # Track wins for each episode
    
    print(f"Starting training for {episodes} episodes...")
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
            
            # Select action
            action = agent.select_action(state, valid_actions)
            
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
    eval_episodes: int = 20
):
    """
    Train DQN agent on Minesweeper (without visualization).
    
    Args:
        episodes: Number of training episodes
        difficulty: Game difficulty
        save_path: Path to save model
        log_interval: Episodes between logging
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
        eval_episodes=eval_episodes
    )


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train DQN agent on Minesweeper")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--difficulty", type=str, default="medium", 
                       choices=["easy", "medium", "hard"], help="Game difficulty")
    parser.add_argument("--save-path", type=str, default="models/dqn_model.pth",
                       help="Path to save model")
    parser.add_argument("--log-interval", type=int, default=100,
                       help="Episodes between logging")
    parser.add_argument("--use-flags", action="store_true",
                       help="Enable flagging actions in the action space")
    
    args = parser.parse_args()
    
    train(
        episodes=args.episodes,
        difficulty=args.difficulty,
        save_path=args.save_path,
        log_interval=args.log_interval,
        use_flag_actions=args.use_flags
    )


if __name__ == "__main__":
    main()

```

```python
"""Training script for DQN agent."""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from src.reinforcement_learning.dqn_agent import DQNAgent
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
    agent: DQNAgent,
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
            action = agent.select_action(state, valid_actions)
            state, reward, done, info = eval_env.step(action)
            reward_sum += reward
            steps += 1
        
        if info["won"]:
            wins += 1
        total_rewards.append(reward_sum)
        total_lengths.append(steps)
    
    agent.epsilon = original_epsilon
    
    return {
        "episodes": episodes,
        "win_rate": (wins / episodes) * 100.0,
        "avg_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "avg_length": float(np.mean(total_lengths)) if total_lengths else 0.0
    }


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
    eval_episodes: int = 20
):
    """
    Train DQN agent on Minesweeper with visualization support.
    
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
    
    # Adjust hyperparameters based on board size
    # Smaller boards need faster learning, larger boards need more exploration
    board_size_factor = (width * height) / 600.0  # Normalize to default 30x20
    
    # Learning rate scaled by board size (smaller boards learn faster)
    base_lr = 0.001
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
    epsilon_schedule = LinearSchedule(epsilon_start, epsilon_floor, decay_span)
    
    # Batch size tuned per board size
    batch_size = min(96, max(32, int(48 * (1.0 / max(board_size_factor, 0.6)))))
    
    agent = DQNAgent(
        state_channels=env.state_channels,
        action_space_size=env.action_space_size,
        board_height=height,
        board_width=width,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_floor,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    episode_wins = []  # Track wins for each episode
    
    print(f"Starting training for {episodes} episodes...")
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
            
            # Select action
            action = agent.select_action(state, valid_actions)
            
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
    eval_episodes: int = 20
):
    """
    Train DQN agent on Minesweeper (without visualization).
    
    Args:
        episodes: Number of training episodes
        difficulty: Game difficulty
        save_path: Path to save model
        log_interval: Episodes between logging
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
        eval_episodes=eval_episodes
    )


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train DQN agent on Minesweeper")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--difficulty", type=str, default="medium", 
                       choices=["easy", "medium", "hard"], help="Game difficulty")
    parser.add_argument("--save-path", type=str, default="models/dqn_model.pth",
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
    
    args = parser.parse_args()
    
    train(
        episodes=args.episodes,
        difficulty=args.difficulty,
        save_path=args.save_path,
        log_interval=args.log_interval,
        use_flag_actions=args.use_flags,
        width=args.width,
        height=args.height,
        eval_episodes=args.eval_episodes
    )


if __name__ == "__main__":
    main()

```

```python
"""Training script for DQN agent."""

import os
import argparse
import numpy as np
from src.reinforcement_learning.dqn_agent import DQNAgent
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
    agent: DQNAgent,
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
            action = agent.select_action(state, valid_actions)
            state, reward, done, info = eval_env.step(action)
            reward_sum += reward
            steps += 1
        
        if info["won"]:
            wins += 1
        total_rewards.append(reward_sum)
        total_lengths.append(steps)
    
    agent.epsilon = original_epsilon
    
    return {
        "episodes": episodes,
        "win_rate": (wins / episodes) * 100.0,
        "avg_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "avg_length": float(np.mean(total_lengths)) if total_lengths else 0.0
    }


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
    eval_episodes: int = 20
):
    """
    Train DQN agent on Minesweeper with visualization support.
    
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
    
    # Adjust hyperparameters based on board size
    # Smaller boards need faster learning, larger boards need more exploration
    board_size_factor = (width * height) / 600.0  # Normalize to default 30x20
    
    # Learning rate scaled by board size (smaller boards learn faster)
    base_lr = 0.001
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
    epsilon_schedule = LinearSchedule(epsilon_start, epsilon_floor, decay_span)
    
    # Batch size tuned per board size
    batch_size = min(96, max(32, int(48 * (1.0 / max(board_size_factor, 0.6)))))
    
    agent = DQNAgent(
        state_channels=env.state_channels,
        action_space_size=env.action_space_size,
        board_height=height,
        board_width=width,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_floor,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    episode_wins = []  # Track wins for each episode
    
    print(f"Starting training for {episodes} episodes...")
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
            
            # Select action
            action = agent.select_action(state, valid_actions)
            
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
    eval_episodes: int = 20
):
    """
    Train DQN agent on Minesweeper (without visualization).
    
    Args:
        episodes: Number of training episodes
        difficulty: Game difficulty
        save_path: Path to save model
        log_interval: Episodes between logging
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
        eval_episodes=eval_episodes
    )


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train DQN agent on Minesweeper")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--difficulty", type=str, default="medium", 
                       choices=["easy", "medium", "hard"], help="Game difficulty")
    parser.add_argument("--save-path", type=str, default="models/dqn_model.pth",
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
    
    args = parser.parse_args()
    
    train(
        episodes=args.episodes,
        difficulty=args.difficulty,
        save_path=args.save_path,
        log_interval=args.log_interval,
        use_flag_actions=args.use_flags,
        width=args.width,
        height=args.height,
        eval_episodes=args.eval_episodes
    )


if __name__ == "__main__":
    main()

```

```python
"""Tests for MinesweeperEnvironment."""

import unittest
import numpy as np
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class TestMinesweeperEnvironment(unittest.TestCase):
    """Test cases for MinesweeperEnvironment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MinesweeperEnvironment("medium", use_flag_actions=False)
        self.env_with_flags = MinesweeperEnvironment("medium", use_flag_actions=True)
    
    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.difficulty, "medium")
        self.assertEqual(self.env.action_space_size, BOARD_WIDTH * BOARD_HEIGHT)
        self.assertIsNotNone(self.env.game)
        self.assertEqual(self.env_with_flags.action_space_size, BOARD_WIDTH * BOARD_HEIGHT * 2)
    
    def test_reset(self):
        """Test environment reset."""
        state = self.env.reset()
        
        # Check state shape
        self.assertEqual(state.shape, (STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH))
        self.assertEqual(state.dtype, np.float32)
        
        # Check that all cells are hidden initially (value should be -0.9)
        # Actually, first click hasn't happened yet, so state should be all hidden
        self.assertTrue(np.all(state[0] == -0.9))
        self.assertTrue(np.all(state[1] == 1.0))
    
    def test_step_valid_action(self):
        """Test step with valid action."""
        state = self.env.reset()
        
        # First action (should be valid)
        action = 0  # Top-left cell
        next_state, reward, done, info = self.env.step(action)
        
        # Check return types
        self.assertIsInstance(next_state, np.ndarray)
        self.assertIsInstance(reward, (float, np.floating))
        self.assertIsInstance(done, (bool, np.bool_))
        self.assertIsInstance(info, dict)
        
        # Check state shape
        self.assertEqual(next_state.shape, (STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH))
        
        # Check info keys
        self.assertIn("won", info)
        self.assertIn("lost", info)
        self.assertIn("valid_action", info)
    
    def test_step_invalid_action(self):
        """Test step with invalid action (already revealed cell)."""
        state = self.env.reset()
        
        # First action
        action1 = 0
        next_state, reward1, done1, info1 = self.env.step(action1)
        
        # Try same action again (should be invalid)
        next_state2, reward2, done2, info2 = self.env.step(action1)
        
        # Should get negative reward for invalid action
        self.assertLess(reward2, 0)
        self.assertFalse(info2["valid_action"])
    
    def test_state_representation(self):
        """Test state representation encoding."""
        state = self.env.reset()
        
        # Make some moves
        self.env.step(0)
        state = self.env._get_state()
        
        # Check state values are in expected range
        # -1.0 = mine, -0.9 = hidden, -0.5 = flagged, 0-1 = numbers
        self.assertTrue(np.all(state >= -1.0))
        self.assertTrue(np.all(state <= 1.0))
    
    def test_reward_calculation(self):
        """Test reward calculation."""
        state = self.env.reset()
        
        # First action should give reward if cells are revealed
        _, reward, _, _ = self.env.step(0)
        # Reward could be positive (cells revealed) or negative (mine hit)
        self.assertIsInstance(reward, (float, np.floating))
    
    def test_reward_on_win(self):
        """Test reward when game is won."""
        # This is hard to test deterministically, but we can check the structure
        state = self.env.reset()
        
        # Make moves until game ends (if possible)
        # For now, just check that reward calculation handles win case
        # (This would require solving the entire game, which is complex)
        pass
    
    def test_reward_on_loss(self):
        """Test reward when mine is hit."""
        state = self.env.reset()
        
        # Try to find and hit a mine (unlikely but possible)
        # If we hit a mine, reward should be -500.0 (updated reward structure)
        for _ in range(100):  # Try up to 100 actions
            valid_actions = self.env.get_valid_actions()
            if not np.any(valid_actions):
                break
            
            action = np.random.choice(np.where(valid_actions)[0])
            _, reward, done, info = self.env.step(action)
            
            if info["lost"]:
                # Reward is now scaled by board size, so check range instead of exact value
                # For medium difficulty (30x20), should be around -100
                self.assertLess(reward, 0)
                self.assertGreaterEqual(reward, -200.0)  # Allow some flexibility
                break
    
    def test_get_valid_actions(self):
        """Test getting valid actions."""
        state = self.env.reset()
        
        cell_count = BOARD_WIDTH * BOARD_HEIGHT
        valid_reveal_only = self.env.get_valid_actions()
        self.assertEqual(valid_reveal_only.shape, (cell_count,))
        self.assertEqual(valid_reveal_only.dtype, bool)
        self.assertTrue(np.all(valid_reveal_only))
        
        # After first action, some should become invalid
        self.env.step(0)
        valid_reveal_only = self.env.get_valid_actions()
        self.assertFalse(np.all(valid_reveal_only))  # Some reveals invalid now
        
        # With flags enabled
        self.env_with_flags.reset()
        valid_with_flags = self.env_with_flags.get_valid_actions()
        self.assertEqual(valid_with_flags.shape, (cell_count * 2,))
        self.assertTrue(np.all(valid_with_flags[:cell_count]))
        self.assertTrue(np.all(valid_with_flags[cell_count:]))
    
    def test_get_action_mask(self):
        """Test action masking."""
        state = self.env.reset()
        
        mask_no_flags = self.env.get_action_mask()
        self.assertEqual(mask_no_flags.shape, (BOARD_WIDTH * BOARD_HEIGHT,))
        valid_nf = self.env.get_valid_actions()
        self.assertTrue(np.all(mask_no_flags[valid_nf] == 0.0))
        
        mask_flags = self.env_with_flags.get_action_mask()
        self.assertEqual(mask_flags.shape, (BOARD_WIDTH * BOARD_HEIGHT * 2,))
        valid_flags = self.env_with_flags.get_valid_actions()
        self.assertTrue(np.all(mask_flags[valid_flags] == 0.0))
        self.assertTrue(np.all(np.isinf(mask_flags[~valid_flags])))
    
    def test_multiple_episodes(self):
        """Test multiple episodes."""
        # Run multiple episodes
        for _ in range(3):
            state = self.env.reset()
            self.assertFalse(self.env.game.is_game_over())
            
            # Make a few moves
            for _ in range(5):
                if self.env.game.is_game_over():
                    break
                valid = self.env.get_valid_actions()
                if np.any(valid):
                    action = np.random.choice(np.where(valid)[0])
                    self.env.step(action)
    
    def test_flag_rewards(self):
        """Test rewards for flagging correct/incorrect cells."""
        env = self.env_with_flags
        env.reset()
        # Trigger mine placement
        env.step(0)
        
        mine_idx = None
        safe_idx = None
        
        for row in range(env.height):
            for col in range(env.width):
                cell = env.game.board.get_cell(row, col)
                if cell.is_revealed():
                    continue
                idx = row * env.width + col
                if cell.is_mine and mine_idx is None:
                    mine_idx = idx
                elif not cell.is_mine and safe_idx is None:
                    safe_idx = idx
            if mine_idx is not None and safe_idx is not None:
                break
        
        self.assertIsNotNone(mine_idx)
        self.assertIsNotNone(safe_idx)
        
        # Flag a mine -> positive reward
        mine_flag_action = mine_idx + env.cell_count
        _, reward_correct, _, _ = env.step(mine_flag_action)
        self.assertGreater(reward_correct, 0)
        
        # Flag a safe cell -> penalty
        safe_flag_action = safe_idx + env.cell_count
        _, reward_wrong, _, _ = env.step(safe_flag_action)
        self.assertLess(reward_wrong, 0)


if __name__ == "__main__":
    unittest.main()


```

```python
"""Tests for MinesweeperEnvironment."""

import unittest
import numpy as np
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class TestMinesweeperEnvironment(unittest.TestCase):
    """Test cases for MinesweeperEnvironment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MinesweeperEnvironment("medium", use_flag_actions=False)
        self.env_with_flags = MinesweeperEnvironment("medium", use_flag_actions=True)
    
    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.difficulty, "medium")
        self.assertEqual(self.env.action_space_size, BOARD_WIDTH * BOARD_HEIGHT)
        self.assertIsNotNone(self.env.game)
        self.assertEqual(self.env_with_flags.action_space_size, BOARD_WIDTH * BOARD_HEIGHT * 2)
    
    def test_reset(self):
        """Test environment reset."""
        state = self.env.reset()
        
        # Check state shape
        self.assertEqual(state.shape, (STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH))
        self.assertEqual(state.dtype, np.float32)
        
        # Check that all cells are hidden initially (value should be -0.9)
        # Actually, first click hasn't happened yet, so state should be all hidden
        self.assertTrue(np.all(state[0] == -0.9))
        self.assertTrue(np.all(state[1] == 1.0))
    
    def test_step_valid_action(self):
        """Test step with valid action."""
        state = self.env.reset()
        
        # First action (should be valid)
        action = 0  # Top-left cell
        next_state, reward, done, info = self.env.step(action)
        
        # Check return types
        self.assertIsInstance(next_state, np.ndarray)
        self.assertIsInstance(reward, (float, np.floating))
        self.assertIsInstance(done, (bool, np.bool_))
        self.assertIsInstance(info, dict)
        
        # Check state shape
        self.assertEqual(next_state.shape, (STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH))
        
        # Check info keys
        self.assertIn("won", info)
        self.assertIn("lost", info)
        self.assertIn("valid_action", info)
    
    def test_step_invalid_action(self):
        """Test step with invalid action (already revealed cell)."""
        state = self.env.reset()
        
        # First action
        action1 = 0
        next_state, reward1, done1, info1 = self.env.step(action1)
        
        # Try same action again (should be invalid)
        next_state2, reward2, done2, info2 = self.env.step(action1)
        
        # Should get negative reward for invalid action
        self.assertLess(reward2, 0)
        self.assertFalse(info2["valid_action"])
    
    def test_state_representation(self):
        """Test state representation encoding."""
        state = self.env.reset()
        
        # Make some moves
        self.env.step(0)
        state = self.env._get_state()
        
        # Check state values are in expected range
        # -1.0 = mine, -0.9 = hidden, -0.5 = flagged, 0-1 = numbers
        self.assertTrue(np.all(state >= -1.0))
        self.assertTrue(np.all(state <= 1.0))
    
    def test_reward_calculation(self):
        """Test reward calculation."""
        state = self.env.reset()
        
        # First action should give reward if cells are revealed
        _, reward, _, _ = self.env.step(0)
        # Reward could be positive (cells revealed) or negative (mine hit)
        self.assertIsInstance(reward, (float, np.floating))
    
    def test_reward_on_win(self):
        """Test reward when game is won."""
        # This is hard to test deterministically, but we can check the structure
        state = self.env.reset()
        
        # Make moves until game ends (if possible)
        # For now, just check that reward calculation handles win case
        # (This would require solving the entire game, which is complex)
        pass
    
    def test_reward_on_loss(self):
        """Test reward when mine is hit."""
        state = self.env.reset()
        
        # Try to find and hit a mine (unlikely but possible)
        # Loss penalty scales with board size (approx -12 * scale)
        for _ in range(100):  # Try up to 100 actions
            valid_actions = self.env.get_valid_actions()
            if not np.any(valid_actions):
                break
            
            action = np.random.choice(np.where(valid_actions)[0])
            _, reward, done, info = self.env.step(action)
            
            if info["lost"]:
                # Reward is now scaled by board size, so check range instead of exact value
                # For medium difficulty (30x20), should be around -100
                self.assertLess(reward, 0)
                self.assertGreaterEqual(reward, -200.0)  # Allow some flexibility
                break
    
    def test_get_valid_actions(self):
        """Test getting valid actions."""
        state = self.env.reset()
        
        cell_count = BOARD_WIDTH * BOARD_HEIGHT
        valid_reveal_only = self.env.get_valid_actions()
        self.assertEqual(valid_reveal_only.shape, (cell_count,))
        self.assertEqual(valid_reveal_only.dtype, bool)
        self.assertTrue(np.all(valid_reveal_only))
        
        # After first action, some should become invalid
        self.env.step(0)
        valid_reveal_only = self.env.get_valid_actions()
        self.assertFalse(np.all(valid_reveal_only))  # Some reveals invalid now
        
        # With flags enabled
        self.env_with_flags.reset()
        valid_with_flags = self.env_with_flags.get_valid_actions()
        self.assertEqual(valid_with_flags.shape, (cell_count * 2,))
        self.assertTrue(np.all(valid_with_flags[:cell_count]))
        self.assertTrue(np.all(valid_with_flags[cell_count:]))
    
    def test_get_action_mask(self):
        """Test action masking."""
        state = self.env.reset()
        
        mask_no_flags = self.env.get_action_mask()
        self.assertEqual(mask_no_flags.shape, (BOARD_WIDTH * BOARD_HEIGHT,))
        valid_nf = self.env.get_valid_actions()
        self.assertTrue(np.all(mask_no_flags[valid_nf] == 0.0))
        
        mask_flags = self.env_with_flags.get_action_mask()
        self.assertEqual(mask_flags.shape, (BOARD_WIDTH * BOARD_HEIGHT * 2,))
        valid_flags = self.env_with_flags.get_valid_actions()
        self.assertTrue(np.all(mask_flags[valid_flags] == 0.0))
        self.assertTrue(np.all(np.isinf(mask_flags[~valid_flags])))
    
    def test_multiple_episodes(self):
        """Test multiple episodes."""
        # Run multiple episodes
        for _ in range(3):
            state = self.env.reset()
            self.assertFalse(self.env.game.is_game_over())
            
            # Make a few moves
            for _ in range(5):
                if self.env.game.is_game_over():
                    break
                valid = self.env.get_valid_actions()
                if np.any(valid):
                    action = np.random.choice(np.where(valid)[0])
                    self.env.step(action)
    
    def test_flag_rewards(self):
        """Test rewards for flagging correct/incorrect cells."""
        env = self.env_with_flags
        env.reset()
        # Trigger mine placement
        env.step(0)
        
        mine_idx = None
        safe_idx = None
        
        for row in range(env.height):
            for col in range(env.width):
                cell = env.game.board.get_cell(row, col)
                if cell.is_revealed():
                    continue
                idx = row * env.width + col
                if cell.is_mine and mine_idx is None:
                    mine_idx = idx
                elif not cell.is_mine and safe_idx is None:
                    safe_idx = idx
            if mine_idx is not None and safe_idx is not None:
                break
        
        self.assertIsNotNone(mine_idx)
        self.assertIsNotNone(safe_idx)
        
        # Flag a mine -> positive reward
        mine_flag_action = mine_idx + env.cell_count
        _, reward_correct, _, _ = env.step(mine_flag_action)
        self.assertGreater(reward_correct, 0)
        
        # Flag a safe cell -> penalty
        safe_flag_action = safe_idx + env.cell_count
        _, reward_wrong, _, _ = env.step(safe_flag_action)
        self.assertLess(reward_wrong, 0)


if __name__ == "__main__":
    unittest.main()


```

```markdown
# KIP_Projekt - Minesweeper mit Reinforcement Learning

Ein vollständiges Minesweeper-Spiel mit GUI (PySide6) und Reinforcement Learning (DQN mit PyTorch).

## Projektübersicht

Dieses Projekt implementiert:
- **Minesweeper-Spiel** mit 20x30 Spielfeld
- **GUI** mit PySide6 für manuelles Spielen
- **Reinforcement Learning** mit Deep Q-Network (DQN) basierend auf PyTorch
- **3 Schwierigkeitsgrade**: Leicht (~10%), Mittel (~15%), Schwer (~20%)

## Projektstruktur

```
KIP_Projekt/
├── src/
│   ├── minesweeper/          # Spiellogik
│   ├── gui/                  # GUI-Komponenten
│   ├── reinforcement_learning/ # RL-Implementierung
│   └── utils/                # Hilfsfunktionen
├── models/                   # Gespeicherte RL-Modelle
├── data/                     # Training-Daten/Logs
├── requirements.txt
└── main.py                   # Hauptprogramm
```

## Installation

### Voraussetzungen

Python 3.8 oder höher muss installiert sein. Falls Python noch nicht installiert ist:

**Python von python.org**
1. Besuchen Sie https://www.python.org/downloads/
2. Laden Sie die neueste Python-Version herunter (3.8+)
3. Während der Installation: **Wichtig:** Aktivieren Sie "Add Python to PATH"
4. Installation abschließen

**Installation überprüfen:**
```powershell
python --version
# oder
py --version
```

### Projekt-Setup

1. Repository klonen oder herunterladen
2. Abhängigkeiten installieren:
```bash
# Wenn Python im PATH ist:
pip install -r requirements.txt

# Falls pip nicht gefunden wird, versuchen Sie:
python -m pip install -r requirements.txt
# oder
py -m pip install -r requirements.txt
```

**Hinweis für Windows:** Falls `pip` nicht erkannt wird, verwenden Sie `python -m pip` oder `py -m pip`.

### PyTorch Installation (Windows)

PyTorch benötigt die **Visual C++ Redistributable**. Falls beim Import von PyTorch ein Fehler auftritt:

**Visual C++ Redistributable installieren**
1. Laden Sie die Visual C++ Redistributable herunter:
   - https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Installieren Sie die Datei
3. PowerShell/Terminal neu starten

## Verwendung

### Manuelles Spielen

Starten Sie das GUI mit:
```bash
python main.py
```

**Bedienung:**
- Linksklick: Zelle aufdecken
- Rechtsklick: Flagge setzen/entfernen
- Menü: Neues Spiel mit verschiedenen Schwierigkeitsgraden starten
- Sichere Zellen werden automatisch aufgedeckt

### Reinforcement Learning Training

#### Training mit Visualisierung (GUI)

1. **Anwendung starten:**
   ```bash
   python main.py
   ```

2. **Training starten:**
   - Menü → **Reinforcement Learning** → **Training starten (mit Visualisierung)**
   - Anzahl Episoden eingeben (z.B. 1000)
   - Schwierigkeit wählen (Leicht/Mittel/Schwer)

3. **Visualisierung:**
   - Alle **100 Episoden** wird automatisch eine Visualisierung gestartet
   - Am **Ende** folgt ein finaler Test-Lauf
   - Sie sehen in Echtzeit, wie der Agent spielt!

#### Training über Kommandozeile (ohne GUI)

```bash
python -m src.reinforcement_learning.trainer ^
  --episodes 1200 ^
  --difficulty medium ^
  --width 15 --height 10 ^
  --eval-episodes 25 ^
  --save-path models/dqn_model_15x10.pth
```

**Parameter (Auszug):**
- `--episodes`: Anzahl der Trainingsepisoden (Standard: 1000)
- `--difficulty`: Schwierigkeitsgrad - easy, medium, hard (Standard: medium)
- `--width / --height`: Optionales Spielfeld-Override (Standard: Werte aus `constants.py`)
- `--save-path`: Pfad zum Speichern des Modells (Standard: models/dqn_model.pth)
- `--log-interval`: Episoden zwischen Logging (Standard: 100)
- `--use-flags`: Flag- und Aufdeck-Aktionen trainieren (Standard: ausgeschaltet → Reveal-only wie im Referenzprojekt)
- `--eval-episodes`: Anzahl greedy Evaluations-Episoden pro Log-Block (zeigt echte Gewinnrate ohne Exploration)

Jeder Log-Eintrag enthält nun zusätzlich eine **greedy Evaluation (ε=0)**, sodass sofort sichtbar ist, wie gut das Modell ohne Zufallszüge abschneidet. Am Ende des Trainings wird automatisch eine größere Evaluationsserie (mind. 25 Episoden) ausgeführt und zusammengefasst.

### Geladenes Modell testen

1. **Modell laden:**
   - Menü → **Reinforcement Learning** → **Modell laden und testen**
   - Wählen Sie die Modell-Datei (`models/dqn_model.pth`)

2. **Agent spielen lassen:**
   - Klicken Sie auf **"RL-Agent spielen lassen"** im RL-Visualizer-Widget
   - Der Agent spielt automatisch eine Episode

**Detaillierte Anleitung:** Siehe [docs/RL_TRAINING_GUIDE.md](docs/RL_TRAINING_GUIDE.md)

## Technische Details

- **Programmiersprache**: Python 3.8+
- **GUI Framework**: PySide6
- **RL Framework**: PyTorch
- **Algorithmus**: Deep Q-Network (DQN) mit Experience Replay und Target Network

## Features

- ✅ Vollständiges Minesweeper-Spiel mit 20x30 Spielfeld
- ✅ 3 Schwierigkeitsgrade (Leicht, Mittel, Schwer)
- ✅ Manuelles Spielen mit intuitiver GUI
- ✅ Automatisches Aufdecken sicherer Zellen
- ✅ DQN-basiertes Reinforcement Learning
- ✅ Training-Script mit Logging
- ✅ Modell-Speicherung und -Laden
- ✅ **RL-Visualisierung während des Trainings (alle 100 Episoden)**
- ✅ **Finaler Test-Lauf am Ende des Trainings**
- ✅ **Manuelles Testen geladener Modelle in der GUI**
- ✅ **Reveal-only Action Space (Flags optional) – inspiriert durch [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning) für stabileres Training**
- ✅ **Lineare Epsilon-Schedules & automatische greedy Evaluationsläufe (Win-Rate ohne Zufall)**
- ✅ **CLI-Training mit frei wählbarer Brettgröße (`--width/--height`) und Flag-Modus**

## Entwicklung

Das Projekt wurde mit Hilfe von Cursor und KI-Assistenz erstellt, um zu testen, wie effektiv KI bei der Programmierung helfen kann.

```

```markdown
# RL Training & Visualisierung - Anleitung

## Übersicht

Diese Anleitung erklärt, wie Sie den DQN-Agenten trainieren und die Visualisierung verwenden.

## Training starten

### Über die GUI (Empfohlen)

1. **Anwendung starten:**
   ```bash
   python main.py
   ```

2. **Training starten:**
   - Klicken Sie auf: **Menü → Reinforcement Learning → Training starten (mit Visualisierung)**
   - Im ersten Dialog: Anzahl Episoden eingeben (z.B. `1000`)
   - Im zweiten Dialog: Schwierigkeit wählen:
     - **Leicht**: Weniger Minen, einfacher zum Lernen
     - **Mittel**: Standard-Schwierigkeit
     - **Schwer**: Mehr Minen, schwieriger

3. **Training läuft:**
   - Das Training läuft im Hintergrund
   - Alle **100 Episoden** wird automatisch eine Visualisierung gestartet
   - Am **Ende** folgt ein finaler Test-Lauf

### Über die Kommandozeile (Ohne GUI)

```bash
python -m src.reinforcement_learning.trainer ^
  --episodes 1500 ^
  --difficulty easy ^
  --width 9 --height 9 ^
  --eval-episodes 20
```

**Wichtige Parameter:**
- `--episodes`: Anzahl Episoden (Standard: 1000)
- `--difficulty`: easy, medium, hard (Standard: medium)
- `--width / --height`: Override für Brettgröße (Standard: Werte aus `src/utils/constants.py`)
- `--save-path`: Pfad zum Speichern (Standard: models/dqn_model.pth)
- `--log-interval`: Episoden zwischen Logging (Standard: 100)
- `--use-flags`: Flag- UND Reveal-Aktionen trainieren (Standard deaktiviert → Reveal-only wie im Referenzprojekt von [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning))
- `--eval-episodes`: Anzahl greedy Evaluations-Episoden pro Log-Block (ε=0). Setze `0`, falls nur Trainingsstatistiken benötigt werden.

Nach jedem Log-Block werden nun zwei Werte ausgegeben:
1. **Trainings-Winrate** (inkl. Exploration, d.h. mit aktuellem ε)
2. **Greedy Evaluation** (ε=0) → echte Gewinnrate ohne Zufallszüge, plus durchschnittliche Zuganzahl.

## Visualisierung verstehen

### Automatische Visualisierungen (alle 100 Episoden)

Während des Trainings:
1. Nach jeder 100. Episode erscheint automatisch eine Visualisierung
2. Der Agent spielt eine komplette Episode
3. Sie sehen in Echtzeit, wie der Agent Zellen aufdeckt
4. Nach der Episode wird das Ergebnis angezeigt (Gewonnen/Verloren)

### Finaler Test-Lauf

Am Ende des Trainings:
- Ein finaler Test-Lauf wird automatisch gestartet
- Schnellere Visualisierung (50ms Delay statt 100ms)
- Zeigt die finale Leistung des trainierten Agents

## Geladenes Modell testen

### Modell laden

1. **Menü → Reinforcement Learning → Modell laden und testen**
2. Wählen Sie die Modell-Datei (z.B. `models/dqn_model.pth`)
3. Klicken Sie OK

### Agent manuell spielen lassen

1. Im **RL-Visualizer-Widget** (unterhalb des Spielfelds) finden Sie:
   - Button: **"RL-Agent spielen lassen"**
   - Button: **"Stoppen"**

2. **Agent starten:**
   - Klicken Sie auf "RL-Agent spielen lassen"
   - Der Agent spielt automatisch eine Episode
   - Sie sehen in Echtzeit die Entscheidungen

3. **Agent stoppen:**
   - Klicken Sie auf "Stoppen" um die aktuelle Episode zu beenden

## Tipps für effektives Training

### Training-Parameter

- **Episoden**: Starten Sie mit 1000 Episoden für erste Tests
- **Schwierigkeit**: Beginnen Sie mit "Leicht" für schnelleres Lernen
- **Wartezeit**: Die Visualisierung kann mit dem Delay-Parameter angepasst werden

### Was zu beachten ist

1. **Training dauert**: Je nach Episodenanzahl kann Training mehrere Minuten/Stunden dauern
2. **GUI bleibt aktiv**: Sie können während des Trainings weiterhin manuell spielen
3. **Modell wird gespeichert**: Das Modell wird automatisch gespeichert (Standard: `models/dqn_model.pth`)

## Troubleshooting

### Training startet nicht

- Überprüfen Sie, ob PyTorch korrekt installiert ist
- Prüfen Sie die Konsole auf Fehlermeldungen

### Visualisierung erscheint nicht

- Warten Sie bis Episode 100, 200, 300, etc. erreicht wird
- Prüfen Sie, ob das Training läuft (Konsole)

### Modell kann nicht geladen werden

- Stellen Sie sicher, dass das Modell bereits trainiert wurde
- Überprüfen Sie den Pfad zur Modell-Datei
- Prüfen Sie, ob die Modell-Datei existiert (`models/dqn_model.pth`)

## Fortgeschrittene Nutzung

### Training fortsetzen

Sie können ein bereits trainiertes Modell laden und weiter trainieren (zukünftige Erweiterung).

### Verschiedene Schwierigkeitsgrade

- Trainieren Sie verschiedene Modelle für verschiedene Schwierigkeitsgrade
- Speichern Sie sie mit unterschiedlichen Namen (z.B. `dqn_model_easy.pth`, `dqn_model_hard.pth`)

## Beispiel-Workflow

1. **Start**: `python main.py`
2. **Training starten**: Menü → RL → Training starten
   - 1000 Episoden, Mittel
3. **Warten**: Beobachten Sie die Visualisierungen alle 100 Episoden
4. **Modell laden**: Nach Training → Menü → RL → Modell laden
5. **Testen**: "RL-Agent spielen lassen" klicken
6. **Beobachten**: Sehen Sie, wie gut der Agent spielt!


```

```markdown
# Reinforcement Learning Implementierungs-Guide

Dieses Dokument erklärt detailliert, wie das Reinforcement Learning System für Minesweeper implementiert ist.

## Übersicht

Das RL-System verwendet **Deep Q-Network (DQN)** mit folgenden Komponenten:
- **Environment**: Wrapper für das Minesweeper-Spiel
- **DQN Network**: Convolutional Neural Network für Q-Value-Schätzung
- **DQN Agent**: Agent mit Experience Replay und Target Network
- **Training**: Episoden-basiertes Training mit Logging

---

## 1. Environment (`src/reinforcement_learning/environment.py`)

### Zweck
Wrappt das Minesweeper-Spiel in ein Gymnasium-ähnliches Interface für RL-Algorithmen.

### Implementierung

#### State Representation (Zustandsrepräsentation)

Der State besitzt **7 Kanäle** und liefert vollständigen Kontext über jede Zelle und ihre Nachbarschaft:

1. **Basis-Encoding** (`state[0]`): alter kombinierter Wert (-0.9 hidden, -0.5 Flag, -1 Mine, 0‑1 Zahl) inkl. Nachbarschaftsbonus/-strafe  
2. **Hidden-Maske** (`state[1]`): 1 für verdeckte Zelle, sonst 0  
3. **Flag-Maske** (`state[2]`): 1 für gesetzte Flagge, sonst 0  
4. **Aufgedeckte Zahl** (`state[3]`): normierte Zahl (0‑1) bzw. -1 bei Mine  
5. **Verdeckte Nachbarn** (`state[4]`): Anteil verdeckter Nachbarn (0‑1)  
6. **Flag-Nachbarn** (`state[5]`): Anteil geflaggter Nachbarn (0‑1)  
7. **Hinweis-Summe** (`state[6]`): normierte Summe aller bekannten Nachbarzahlen (0‑1)

Damit kennt der Agent sowohl den lokalen Status als auch die unmittelbare „Frontier“ und vorhandene Hinweise.

#### Action Space (Reveal vs. Flag)

- **Standard (GUI/Test)**: Reveal + Flag mit Größe `2 × width × height`
- **Training (Standard)**: `use_flag_actions=False` → nur Reveal-Aktionen (`width × height`), analog zum Ansatz aus [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning), damit der Agent sich wie ein menschlicher Spieler auf Progress-Züge konzentriert
- **CLI Flag**: `python -m ...trainer --use-flags` aktiviert wieder das alte Verhalten
- **Indexierung (mit Flags)**:
  - `0 … (N-1)`: Zelle aufdecken
  - `N … (2N-1)`: Flagge toggle
- **Validierung**:
  - Reveal nur für verdeckte, nicht geflaggte Felder
  - Flag nur für verdeckte oder bereits geflaggte Felder

#### Reward System

Skaliert automatisch mit der Brettgröße (`reward_scale = max(1, width*height/100)`):

- **Aufdecken**:
  - `+progress_scale` (mind. 0.5) pro neu aufgedecktem Feld
  - Kettenbonus für Flood-Fill (`+0.4 * progress_scale` pro zusätzlichem Feld)
  - Frontier-Bonus (`+0.6`) sobald angrenzende Hinweise existieren
  - Guess-Penalty (`-0.4 * reward_scale`), falls komplett isoliert geklickt wird
- **Keine Fortschritte**: `-0.15 * progress_scale`
- **Flaggen (optional)**:
  - `+0.2` bei korrekt gesetzter Flagge, `-0.2` bei Fehlflag
  - End-Bonus/-Malus (`±0.5`) für korrekte/inkorrekte Flags bei Spielende
- **Spiel verloren**: `-12 * reward_scale` (starke negative Verstärkung)
- **Spiel gewonnen**: `+18 * reward_scale + 6 * progress_ratio`

#### Valid Actions Masking

Das Environment bietet zwei Methoden für gültige Aktionen:

1. **`get_valid_actions()`**: Boolean-Array (True = gültig)
2. **`get_action_mask()`**: Mask für Q-Values (-inf für ungültig, 0.0 für gültig)

**Warum Masking?**
- Verhindert, dass der Agent bereits aufgedeckte oder flagge Zellen auswählt
- Reduziert den Aktionsraum effektiv
- Verbessert Trainingseffizienz

---

## 2. DQN Network (`src/reinforcement_learning/network.py`)

### Architektur

**Convolutional Neural Network** nach sdlee94-Vorbild:

```
Input: (batch, 7, H, W)

Conv Stack (×4):
- Conv2d(in → 128, kernel=3, padding=1)
- BatchNorm2d(128)
- ReLU

AdaptiveAvgPool2d(8 × 8)  ->  128 × 8 × 8 Features

Fully Connected:
- Linear(8192 → 512) + Dropout(0.25)
- Linear(512 → 512) + Dropout(0.25)
- Linear(512 → num_actions)
```

- **4 × 128er Convs**: stärkere lokale Mustererkennung wie im Referenzprojekt  
- **Adaptive Pooling**: funktioniert auf allen Brettgrößen  
- **Doppelte 512er Dense-Layer**: entspricht `conv128x4_dense512x2` aus [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning)

---

## 3. DQN Agent (`src/reinforcement_learning/dqn_agent.py`)

### Komponenten

#### 3.1 ReplayBuffer

**Zweck**: Speichert Erfahrungen (State, Action, Reward, Next State, Done, Next-Action-Maske) für Experience Replay.

**Implementierung:**
- `deque` mit `maxlen` für automatische Größenbegrenzung
- Speichert zusätzlich die zulässigen Aktionen des Folgezustands zur Maskierung der Ziel-Q-Werte
- Zufälliges Sampling für Batch-Training
- Konvertiert NumPy-Arrays zu PyTorch-Tensoren

**Warum Experience Replay?**
- **Stabilisierung**: Bricht Korrelationen zwischen aufeinanderfolgenden Erfahrungen
- **Efficiency**: Nutzt Erfahrungen mehrfach
- **Diversität**: Batch enthält verschiedene Erfahrungen

#### 3.2 DQN Agent

**Zweck**: Implementiert DQN-Algorithmus mit allen notwendigen Komponenten.

**Hauptkomponenten:**

1. **Q-Network**: Haupt-Netzwerk für Q-Value-Schätzung
2. **Double Target Network**: Double DQN vermeidet Q-Value-Overestimation
3. **Epsilon-Greedy**: Exploration vs. Exploitation
4. **Experience Replay**: Speichert und nutzt vergangene Erfahrungen

**Hyperparameter:**

```python
- Learning Rate: 0.001 (Adam Optimizer)
- Gamma (Discount): 0.95 auf Standardbrettern (mehr Fokus auf Fortschritt)
- Epsilon Start: 1.0 (100% Exploration)
- Epsilon End: 0.03–0.10 je nach Schwierigkeitsgrad
- Epsilon Decay: dynamisch berechnet (~70 % der Episoden explorativ)
- Buffer Size: 10,000 Erfahrungen
- Batch Size: 32–96 (abhängig von Brettgröße)
- Target Update: Alle 100 Steps
- Action Space: Standardmäßig nur Reveal-Aktionen (`use_flag_actions=False`), Flags können per CLI aktiviert werden
```

**Gezielte Epsilon-Greedy Strategie:**

```python
if random() < epsilon:
    action = sample_frontier(valid_actions, state)  # Bevorzugt Zellen an bekannten Grenzen
else:
    action = argmax(Q(state, valid_only=True))
```

- **Frontier-Sampling**: Zufallsaktionen wählen bevorzugt verdeckte Felder, die an bereits aufgedeckte Zahlen grenzen  
- **Fallback auf Hinweise**: Wenn keine Frontier existiert, wird ein Feld mit dem größten Hinweis-Signal gewählt  
- **Episodenweiser Decay**: `epsilon` wird nach jeder Episode reduziert, nicht nach jedem Schritt

**Training Process (train_step):**

1. **Batch Sampling**: Zufällige Erfahrungen inkl. Aktionsmasken aus Replay Buffer
2. **Current Q-Values**: Q-Network schätzt Q(s, a)
3. **Target Q-Values**: Double DQN – Aktion via Online-Netz wählen, Q-Wert via Target-Netz bewerten
4. **TD Target**: `target = reward + gamma * Q_target(s', argmax_a Q_online(s'))`
5. **Loss**: Smooth L1 (Huber) zwischen Current und Target
6. **Backpropagation**: Gradienten werden berechnet und angewendet
7. **Gradient Clipping**: Verhindert Exploding Gradients (max norm = 1.0)
8. **Epsilon Decay**: Erfolgt episodenweise über `agent.decay_epsilon()`
9. **Target Update**: Alle N Steps wird Target Network aktualisiert

**Warum Target Network?**
- **Stabilität**: Verhindert instabile Q-Value-Updates
- **Convergence**: Hilft beim Konvergieren des Trainings
- **Delayed Updates**: Target Network wird nur periodisch aktualisiert

**Warum Gradient Clipping?**
- **Stabilität**: Verhindert sehr große Gradienten
- **Training**: Stabilisiert den Lernprozess

---

## 4. Training (`src/reinforcement_learning/trainer.py`)

### Training Loop

```python
for episode in range(episodes):
    state = env.reset()
    done = False
    
    while not done:
        # 1. Agent wählt Aktion
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions)
        
        # 2. Environment führt Aktion aus
        next_state, reward, done, info = env.step(action)
        next_valid_actions = env.get_valid_actions()
        
        # 3. Agent speichert Erfahrung
        agent.remember(state, action, reward, next_state, done, next_valid_actions)
        
        # 4. Agent trainiert (wenn genug Erfahrungen)
        agent.train_step()
        
        # 5. Update state
        state = next_state
    
    # 6. Exploration langsam reduzieren
    agent.decay_epsilon()
    
    # Logging alle N Episoden
    if episode % log_interval == 0:
        log_statistics()
```

### Logging

Alle N Episoden werden folgende Statistiken geloggt:
- **Average Reward**: Durchschnittliche Belohnung
- **Average Length**: Durchschnittliche Episodenlänge
- **Win Rate**: Prozentsatz gewonnener Spiele
- **Epsilon**: Aktueller Exploration-Faktor

### Model Saving

- Periodisches Speichern (alle N × log_interval Episoden)
- Finales Speichern nach Training
- Enthält: Q-Network, Target Network, Optimizer State, Epsilon

---

## 5. Design-Entscheidungen und Optimierungen

### Warum DQN?

1. **Discrete Action Space**: Minesweeper hat diskreten Aktionsraum (600 Aktionen)
2. **Value-Based**: Q-Learning passt gut für deterministische Umgebungen
3. **Bewährt**: DQN ist etabliert und gut verstanden

### Warum diese State-Encoding?

1. **Einfachheit**: Ein Kanal, einfache CNN-Architektur
2. **Normalisierung**: Werte zwischen -1 und 1 für besseres Training
3. **Informativ**: Unterscheidet alle wichtigen Zellzustände

### Warum diese Reward-Struktur?

1. **Sparse + Shaped**: Hauptrewards für wichtige Ereignisse, Shaped für Fortschritt
2. **Balance**: Nicht zu viele kleine Rewards (verhindert Overfitting)
3. **Klare Signale**: Gewinn/Verlust sind deutlich signalisiert

### Mögliche Verbesserungen

1. **Dueling DQN**: Trennt State-Value und Advantage
2. **Prioritized Replay**: Wichtige Erfahrungen werden öfter gesampelt
3. **Multi-Step Learning**: N-Step Returns statt 1-Step
4. **State-Encoding**: Mehr Features (z.B. Wahrscheinlichkeiten je Feld)
5. **Curriculum**: Transfer Learning zwischen Schwierigkeitsgraden

---

## 6. Verwendung

### Training starten:

```bash
python -m src.reinforcement_learning.trainer \
    --episodes 1000 \
    --difficulty medium \
    --save-path models/dqn_model.pth
```

### Modell laden:

```python
from src.reinforcement_learning.dqn_agent import DQNAgent
from src.reinforcement_learning.environment import MinesweeperEnvironment

env = MinesweeperEnvironment("medium")
agent = DQNAgent(state_channels=1, action_space_size=600)
agent.load("models/dqn_model.pth")

# Agent spielen lassen
state = env.reset()
done = False
while not done:
    valid_actions = env.get_valid_actions()
    action = agent.select_action(state, valid_actions)
    state, reward, done, info = env.step(action)
```

---

## 7. Zusammenfassung

Das RL-System implementiert:

✅ **Environment Wrapper**: Gymnasium-ähnliches Interface
✅ **State Encoding**: Normalisierte 2D-Repräsentation
✅ **DQN Network**: CNN für räumliche Features
✅ **Experience Replay**: Stabilisiert Training
✅ **Target Network**: Verhindert instabile Updates
✅ **Epsilon-Greedy**: Exploration/Exploitation Balance
✅ **Reward Shaping**: Sparse + Shaped Rewards
✅ **Action Masking**: Verhindert ungültige Aktionen

Das System ist vollständig funktionsfähig und bereit für Training!


```

```markdown
# Reinforcement Learning Implementierungs-Guide

Dieses Dokument erklärt detailliert, wie das Reinforcement Learning System für Minesweeper implementiert ist.

## Übersicht

Das RL-System verwendet **Deep Q-Network (DQN)** mit folgenden Komponenten:
- **Environment**: Wrapper für das Minesweeper-Spiel
- **DQN Network**: Convolutional Neural Network für Q-Value-Schätzung
- **DQN Agent**: Agent mit Experience Replay und Target Network
- **Training**: Episoden-basiertes Training mit Logging

---

## 1. Environment (`src/reinforcement_learning/environment.py`)

### Zweck
Wrappt das Minesweeper-Spiel in ein Gymnasium-ähnliches Interface für RL-Algorithmen.

### Implementierung

#### State Representation (Zustandsrepräsentation)

Der State besitzt **7 Kanäle** und liefert vollständigen Kontext über jede Zelle und ihre Nachbarschaft:

1. **Basis-Encoding** (`state[0]`): alter kombinierter Wert (-0.9 hidden, -0.5 Flag, -1 Mine, 0‑1 Zahl) inkl. Nachbarschaftsbonus/-strafe  
2. **Hidden-Maske** (`state[1]`): 1 für verdeckte Zelle, sonst 0  
3. **Flag-Maske** (`state[2]`): 1 für gesetzte Flagge, sonst 0  
4. **Aufgedeckte Zahl** (`state[3]`): normierte Zahl (0‑1) bzw. -1 bei Mine  
5. **Verdeckte Nachbarn** (`state[4]`): Anteil verdeckter Nachbarn (0‑1)  
6. **Flag-Nachbarn** (`state[5]`): Anteil geflaggter Nachbarn (0‑1)  
7. **Hinweis-Summe** (`state[6]`): normierte Summe aller bekannten Nachbarzahlen (0‑1)

Damit kennt der Agent sowohl den lokalen Status als auch die unmittelbare „Frontier“ und vorhandene Hinweise.

#### Action Space (Reveal vs. Flag)

- **Standard (GUI/Test)**: Reveal + Flag mit Größe `2 × width × height`
- **Training (Standard)**: `use_flag_actions=False` → nur Reveal-Aktionen (`width × height`), analog zum Ansatz aus [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning), damit der Agent sich wie ein menschlicher Spieler auf Progress-Züge konzentriert
- **CLI Flag**: `python -m ...trainer --use-flags` aktiviert wieder das alte Verhalten
- **Indexierung (mit Flags)**:
  - `0 … (N-1)`: Zelle aufdecken
  - `N … (2N-1)`: Flagge toggle
- **Validierung**:
  - Reveal nur für verdeckte, nicht geflaggte Felder
  - Flag nur für verdeckte oder bereits geflaggte Felder

#### Reward System

Skaliert automatisch mit der Brettgröße (`reward_scale = max(1, width*height/100)`):

- **Aufdecken**:
  - `+progress_scale` (mind. 0.5) pro neu aufgedecktem Feld
  - Kettenbonus für Flood-Fill (`+0.4 * progress_scale` pro zusätzlichem Feld)
  - Frontier-Bonus (`+0.6`) sobald angrenzende Hinweise existieren
  - Guess-Penalty (`-0.4 * reward_scale`), falls komplett isoliert geklickt wird
- **Keine Fortschritte**: `-0.15 * progress_scale`
- **Flaggen (optional)**:
  - `+0.2` bei korrekt gesetzter Flagge, `-0.2` bei Fehlflag
  - End-Bonus/-Malus (`±0.5`) für korrekte/inkorrekte Flags bei Spielende
- **Spiel verloren**: `-12 * reward_scale` (starke negative Verstärkung)
- **Spiel gewonnen**: `+18 * reward_scale + 6 * progress_ratio`

#### Valid Actions Masking

Das Environment bietet zwei Methoden für gültige Aktionen:

1. **`get_valid_actions()`**: Boolean-Array (True = gültig)
2. **`get_action_mask()`**: Mask für Q-Values (-inf für ungültig, 0.0 für gültig)

**Warum Masking?**
- Verhindert, dass der Agent bereits aufgedeckte oder flagge Zellen auswählt
- Reduziert den Aktionsraum effektiv
- Verbessert Trainingseffizienz

---

## 2. DQN Network (`src/reinforcement_learning/network.py`)

### Architektur

**Convolutional Neural Network** nach sdlee94-Vorbild:

```
Input: (batch, 7, H, W)

Conv Stack (×4):
- Conv2d(in → 128, kernel=3, padding=1)
- BatchNorm2d(128)
- ReLU

AdaptiveAvgPool2d(8 × 8)  ->  128 × 8 × 8 Features

Fully Connected:
- Linear(8192 → 512) + Dropout(0.25)
- Linear(512 → 512) + Dropout(0.25)
- Linear(512 → num_actions)
```

- **4 × 128er Convs**: stärkere lokale Mustererkennung wie im Referenzprojekt  
- **Adaptive Pooling**: funktioniert auf allen Brettgrößen  
- **Doppelte 512er Dense-Layer**: entspricht `conv128x4_dense512x2` aus [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning)

---

## 3. DQN Agent (`src/reinforcement_learning/dqn_agent.py`)

### Komponenten

#### 3.1 ReplayBuffer

**Zweck**: Speichert Erfahrungen (State, Action, Reward, Next State, Done, Next-Action-Maske) für Experience Replay.

**Implementierung:**
- `deque` mit `maxlen` für automatische Größenbegrenzung
- Speichert zusätzlich die zulässigen Aktionen des Folgezustands zur Maskierung der Ziel-Q-Werte
- Zufälliges Sampling für Batch-Training
- Konvertiert NumPy-Arrays zu PyTorch-Tensoren

**Warum Experience Replay?**
- **Stabilisierung**: Bricht Korrelationen zwischen aufeinanderfolgenden Erfahrungen
- **Efficiency**: Nutzt Erfahrungen mehrfach
- **Diversität**: Batch enthält verschiedene Erfahrungen

#### 3.2 DQN Agent

**Zweck**: Implementiert DQN-Algorithmus mit allen notwendigen Komponenten.

**Hauptkomponenten:**

1. **Q-Network**: Haupt-Netzwerk für Q-Value-Schätzung
2. **Double Target Network**: Double DQN vermeidet Q-Value-Overestimation
3. **Epsilon-Greedy**: Exploration vs. Exploitation
4. **Experience Replay**: Speichert und nutzt vergangene Erfahrungen

**Hyperparameter:**

```python
- Learning Rate: dynamisch (Basis 0.001, skaliert nach Brettgröße)
- Gamma (Discount): 0.95 (≤600 Felder) bzw. 0.98 (größere Boards)
- Epsilon: Linearer Schedule 1.0 → 0.03–0.10 über ~70 % der Episoden
- Loss: Smooth L1 (Huber), Optimizer: Adam
- Buffer Size: 10,000 Erfahrungen
- Batch Size: 32–96 (abhängig von Brettgröße)
- Target Update: Alle 100 Steps (hartes Sync)
- Action Space: Standardmäßig nur Reveal-Aktionen (`use_flag_actions=False`), Flags via CLI aktivierbar
```

**Gezielte Epsilon-Greedy Strategie:**

```python
if random() < epsilon:
    action = sample_frontier(valid_actions, state)  # Bevorzugt Zellen an bekannten Grenzen
else:
    action = argmax(Q(state, valid_only=True))
```

- **Frontier-Sampling**: Zufallsaktionen wählen bevorzugt verdeckte Felder, die an bereits aufgedeckte Zahlen grenzen  
- **Fallback auf Hinweise**: Wenn keine Frontier existiert, wird ein Feld mit dem größten Hinweis-Signal gewählt  
- **Episodenweiser Decay**: `epsilon` wird nach jeder Episode reduziert, nicht nach jedem Schritt

**Training Process (train_step):**

1. **Batch Sampling**: Zufällige Erfahrungen inkl. Aktionsmasken aus Replay Buffer
2. **Current Q-Values**: Q-Network schätzt Q(s, a)
3. **Target Q-Values**: Double DQN – Aktion via Online-Netz wählen, Q-Wert via Target-Netz bewerten
4. **TD Target**: `target = reward + gamma * Q_target(s', argmax_a Q_online(s'))`
5. **Loss**: Smooth L1 (Huber) zwischen Current und Target
6. **Backpropagation**: Gradienten werden berechnet und angewendet
7. **Gradient Clipping**: Verhindert Exploding Gradients (max norm = 1.0)
8. **Epsilon Decay**: Erfolgt episodenweise über `agent.decay_epsilon()`
9. **Target Update**: Alle N Steps wird Target Network aktualisiert

**Warum Target Network?**
- **Stabilität**: Verhindert instabile Q-Value-Updates
- **Convergence**: Hilft beim Konvergieren des Trainings
- **Delayed Updates**: Target Network wird nur periodisch aktualisiert

**Warum Gradient Clipping?**
- **Stabilität**: Verhindert sehr große Gradienten
- **Training**: Stabilisiert den Lernprozess

---

## 4. Training (`src/reinforcement_learning/trainer.py`)

### Training Loop

```python
for episode in range(episodes):
    state = env.reset()
    done = False
    
    while not done:
        # 1. Agent wählt Aktion
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions)
        
        # 2. Environment führt Aktion aus
        next_state, reward, done, info = env.step(action)
        next_valid_actions = env.get_valid_actions()
        
        # 3. Agent speichert Erfahrung
        agent.remember(state, action, reward, next_state, done, next_valid_actions)
        
        # 4. Agent trainiert (wenn genug Erfahrungen)
        agent.train_step()
        
        # 5. Update state
        state = next_state
    
    # 6. Exploration langsam reduzieren
    agent.decay_epsilon()
    
    # Logging alle N Episoden
    if episode % log_interval == 0:
        log_statistics()
```

### Logging

Alle N Episoden werden folgende Statistiken geloggt:
- **Average Reward**: Durchschnittliche Belohnung
- **Average Length**: Durchschnittliche Episodenlänge
- **Win Rate**: Prozentsatz gewonnener Spiele
- **Epsilon**: Aktueller Exploration-Faktor

### Model Saving

- Periodisches Speichern (alle N × log_interval Episoden)
- Finales Speichern nach Training
- Enthält: Q-Network, Target Network, Optimizer State, Epsilon

---

## 5. Design-Entscheidungen und Optimierungen

### Warum DQN?

1. **Discrete Action Space**: Minesweeper hat diskreten Aktionsraum (600 Aktionen)
2. **Value-Based**: Q-Learning passt gut für deterministische Umgebungen
3. **Bewährt**: DQN ist etabliert und gut verstanden

### Warum diese State-Encoding?

1. **Einfachheit**: Ein Kanal, einfache CNN-Architektur
2. **Normalisierung**: Werte zwischen -1 und 1 für besseres Training
3. **Informativ**: Unterscheidet alle wichtigen Zellzustände

### Warum diese Reward-Struktur?

1. **Sparse + Shaped**: Hauptrewards für wichtige Ereignisse, Shaped für Fortschritt
2. **Balance**: Nicht zu viele kleine Rewards (verhindert Overfitting)
3. **Klare Signale**: Gewinn/Verlust sind deutlich signalisiert

### Mögliche Verbesserungen

1. **Dueling DQN**: Trennt State-Value und Advantage
2. **Prioritized Replay**: Wichtige Erfahrungen werden öfter gesampelt
3. **Multi-Step Learning**: N-Step Returns statt 1-Step
4. **State-Encoding**: Mehr Features (z.B. Wahrscheinlichkeiten je Feld)
5. **Curriculum**: Transfer Learning zwischen Schwierigkeitsgraden

---

## 6. Verwendung

### Training starten:

```bash
python -m src.reinforcement_learning.trainer \
    --episodes 1000 \
    --difficulty medium \
    --save-path models/dqn_model.pth
```

### Modell laden:

```python
from src.reinforcement_learning.dqn_agent import DQNAgent
from src.reinforcement_learning.environment import MinesweeperEnvironment

env = MinesweeperEnvironment("medium")
agent = DQNAgent(state_channels=1, action_space_size=600)
agent.load("models/dqn_model.pth")

# Agent spielen lassen
state = env.reset()
done = False
while not done:
    valid_actions = env.get_valid_actions()
    action = agent.select_action(state, valid_actions)
    state, reward, done, info = env.step(action)
```

---

## 7. Zusammenfassung

Das RL-System implementiert:

✅ **Environment Wrapper**: Gymnasium-ähnliches Interface
✅ **State Encoding**: Normalisierte 2D-Repräsentation
✅ **DQN Network**: CNN für räumliche Features
✅ **Experience Replay**: Stabilisiert Training
✅ **Target Network**: Verhindert instabile Updates
✅ **Epsilon-Greedy**: Exploration/Exploitation Balance
✅ **Reward Shaping**: Sparse + Shaped Rewards
✅ **Action Masking**: Verhindert ungültige Aktionen

Das System ist vollständig funktionsfähig und bereit für Training!


```

```markdown
# Reinforcement Learning Implementierungs-Guide

Dieses Dokument erklärt detailliert, wie das Reinforcement Learning System für Minesweeper implementiert ist.

## Übersicht

Das RL-System verwendet **Deep Q-Network (DQN)** mit folgenden Komponenten:
- **Environment**: Wrapper für das Minesweeper-Spiel
- **DQN Network**: Convolutional Neural Network für Q-Value-Schätzung
- **DQN Agent**: Agent mit Experience Replay und Target Network
- **Training**: Episoden-basiertes Training mit Logging

---

## 1. Environment (`src/reinforcement_learning/environment.py`)

### Zweck
Wrappt das Minesweeper-Spiel in ein Gymnasium-ähnliches Interface für RL-Algorithmen.

### Implementierung

#### State Representation (Zustandsrepräsentation)

Der State besitzt **7 Kanäle** und liefert vollständigen Kontext über jede Zelle und ihre Nachbarschaft:

1. **Basis-Encoding** (`state[0]`): alter kombinierter Wert (-0.9 hidden, -0.5 Flag, -1 Mine, 0‑1 Zahl) inkl. Nachbarschaftsbonus/-strafe  
2. **Hidden-Maske** (`state[1]`): 1 für verdeckte Zelle, sonst 0  
3. **Flag-Maske** (`state[2]`): 1 für gesetzte Flagge, sonst 0  
4. **Aufgedeckte Zahl** (`state[3]`): normierte Zahl (0‑1) bzw. -1 bei Mine  
5. **Verdeckte Nachbarn** (`state[4]`): Anteil verdeckter Nachbarn (0‑1)  
6. **Flag-Nachbarn** (`state[5]`): Anteil geflaggter Nachbarn (0‑1)  
7. **Hinweis-Summe** (`state[6]`): normierte Summe aller bekannten Nachbarzahlen (0‑1)

Damit kennt der Agent sowohl den lokalen Status als auch die unmittelbare „Frontier“ und vorhandene Hinweise.

#### Action Space (Reveal vs. Flag)

- **Standard (GUI/Test)**: Reveal + Flag mit Größe `2 × width × height`
- **Training (Standard)**: `use_flag_actions=False` → nur Reveal-Aktionen (`width × height`), analog zum Ansatz aus [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning), damit der Agent sich wie ein menschlicher Spieler auf Progress-Züge konzentriert
- **CLI Flag**: `python -m ...trainer --use-flags` aktiviert wieder das alte Verhalten
- **Indexierung (mit Flags)**:
  - `0 … (N-1)`: Zelle aufdecken
  - `N … (2N-1)`: Flagge toggle
- **Validierung**:
  - Reveal nur für verdeckte, nicht geflaggte Felder
  - Flag nur für verdeckte oder bereits geflaggte Felder

#### Reward System

Skaliert automatisch mit der Brettgröße (`reward_scale = max(1, width*height/100)`):

- **Aufdecken**:
  - `+progress_scale` (mind. 0.5) pro neu aufgedecktem Feld
  - Kettenbonus für Flood-Fill (`+0.4 * progress_scale` pro zusätzlichem Feld)
  - Frontier-Bonus (`+0.6`) sobald angrenzende Hinweise existieren
  - Guess-Penalty (`-0.4 * reward_scale`), falls komplett isoliert geklickt wird
- **Keine Fortschritte**: `-0.15 * progress_scale`
- **Flaggen (optional)**:
  - `+0.2` bei korrekt gesetzter Flagge, `-0.2` bei Fehlflag
  - End-Bonus/-Malus (`±0.5`) für korrekte/inkorrekte Flags bei Spielende
- **Spiel verloren**: `-12 * reward_scale` (starke negative Verstärkung)
- **Spiel gewonnen**: `+18 * reward_scale + 6 * progress_ratio`

#### Valid Actions Masking

Das Environment bietet zwei Methoden für gültige Aktionen:

1. **`get_valid_actions()`**: Boolean-Array (True = gültig)
2. **`get_action_mask()`**: Mask für Q-Values (-inf für ungültig, 0.0 für gültig)

**Warum Masking?**
- Verhindert, dass der Agent bereits aufgedeckte oder flagge Zellen auswählt
- Reduziert den Aktionsraum effektiv
- Verbessert Trainingseffizienz

---

## 2. DQN Network (`src/reinforcement_learning/network.py`)

### Architektur

**Convolutional Neural Network** nach sdlee94-Vorbild:

```
Input: (batch, 7, H, W)

Conv Stack (×4):
- Conv2d(in → 128, kernel=3, padding=1)
- BatchNorm2d(128)
- ReLU

AdaptiveAvgPool2d(8 × 8)  ->  128 × 8 × 8 Features

Fully Connected:
- Linear(8192 → 512) + Dropout(0.25)
- Linear(512 → 512) + Dropout(0.25)
- Linear(512 → num_actions)
```

- **4 × 128er Convs**: stärkere lokale Mustererkennung wie im Referenzprojekt  
- **Adaptive Pooling**: funktioniert auf allen Brettgrößen  
- **Doppelte 512er Dense-Layer**: entspricht `conv128x4_dense512x2` aus [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning)

---

## 3. DQN Agent (`src/reinforcement_learning/dqn_agent.py`)

### Komponenten

#### 3.1 ReplayBuffer

**Zweck**: Speichert Erfahrungen (State, Action, Reward, Next State, Done, Next-Action-Maske) für Experience Replay.

**Implementierung:**
- `deque` mit `maxlen` für automatische Größenbegrenzung
- Speichert zusätzlich die zulässigen Aktionen des Folgezustands zur Maskierung der Ziel-Q-Werte
- Zufälliges Sampling für Batch-Training
- Konvertiert NumPy-Arrays zu PyTorch-Tensoren

**Warum Experience Replay?**
- **Stabilisierung**: Bricht Korrelationen zwischen aufeinanderfolgenden Erfahrungen
- **Efficiency**: Nutzt Erfahrungen mehrfach
- **Diversität**: Batch enthält verschiedene Erfahrungen

#### 3.2 DQN Agent

**Zweck**: Implementiert DQN-Algorithmus mit allen notwendigen Komponenten.

**Hauptkomponenten:**

1. **Q-Network**: Haupt-Netzwerk für Q-Value-Schätzung
2. **Double Target Network**: Double DQN vermeidet Q-Value-Overestimation
3. **Epsilon-Greedy**: Exploration vs. Exploitation
4. **Experience Replay**: Speichert und nutzt vergangene Erfahrungen

**Hyperparameter:**

```python
- Learning Rate: dynamisch (Basis 0.001, skaliert nach Brettgröße)
- Gamma (Discount): 0.95 (≤600 Felder) bzw. 0.98 (größere Boards)
- Epsilon: Linearer Schedule 1.0 → 0.03–0.10 über ~70 % der Episoden
- Loss: Smooth L1 (Huber), Optimizer: Adam
- Buffer Size: 10,000 Erfahrungen
- Batch Size: 32–96 (abhängig von Brettgröße)
- Target Update: Alle 100 Steps (hartes Sync)
- Action Space: Standardmäßig nur Reveal-Aktionen (`use_flag_actions=False`), Flags via CLI aktivierbar
```

**Gezielte Epsilon-Greedy Strategie:**

```python
if random() < epsilon:
    action = sample_frontier(valid_actions, state)  # Bevorzugt Zellen an bekannten Grenzen
else:
    action = argmax(Q(state, valid_only=True))
```

- **Frontier-Sampling**: Zufallsaktionen wählen bevorzugt verdeckte Felder, die an bereits aufgedeckte Zahlen grenzen  
- **Fallback auf Hinweise**: Wenn keine Frontier existiert, wird ein Feld mit dem größten Hinweis-Signal gewählt  
- **Episodenweiser Decay**: `epsilon` wird nach jeder Episode reduziert, nicht nach jedem Schritt

**Training Process (train_step):**

1. **Batch Sampling**: Zufällige Erfahrungen inkl. Aktionsmasken aus Replay Buffer
2. **Current Q-Values**: Q-Network schätzt Q(s, a)
3. **Target Q-Values**: Double DQN – Aktion via Online-Netz wählen, Q-Wert via Target-Netz bewerten
4. **TD Target**: `target = reward + gamma * Q_target(s', argmax_a Q_online(s'))`
5. **Loss**: Smooth L1 (Huber) zwischen Current und Target
6. **Backpropagation**: Gradienten werden berechnet und angewendet
7. **Gradient Clipping**: Verhindert Exploding Gradients (max norm = 1.0)
8. **Epsilon Decay**: Erfolgt episodenweise über `agent.decay_epsilon()`
9. **Target Update**: Alle N Steps wird Target Network aktualisiert

**Warum Target Network?**
- **Stabilität**: Verhindert instabile Q-Value-Updates
- **Convergence**: Hilft beim Konvergieren des Trainings
- **Delayed Updates**: Target Network wird nur periodisch aktualisiert

**Warum Gradient Clipping?**
- **Stabilität**: Verhindert sehr große Gradienten
- **Training**: Stabilisiert den Lernprozess

---

## 4. Training (`src/reinforcement_learning/trainer.py`)

### Training Loop

```python
schedule = LinearSchedule(start=1.0, end=epsilon_floor, duration=int(0.7 * episodes))

for episode in range(episodes):
    state = env.reset()
    done = False
    
    while not done:
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions)
        next_state, reward, done, info = env.step(action)
        next_valid_actions = env.get_valid_actions()
        agent.remember(state, action, reward, next_state, done, next_valid_actions)
        agent.train_step()
        state = next_state
    
    # Linearer Epsilon-Schedule (kein Multiplikationsrauschen mehr)
    agent.epsilon = schedule.value(episode + 1)
    
    if (episode + 1) % log_interval == 0:
        log_statistics()
        greedy_eval(agent, episodes=eval_episodes, epsilon=0.0)
```

### Logging

Alle N Episoden werden folgende Statistiken geloggt:
- **Average Reward**: Durchschnittliche Belohnung
- **Average Length**: Durchschnittliche Episodenlänge
- **Win Rate**: Prozentsatz gewonnener Spiele
- **Epsilon**: Aktueller Exploration-Faktor

### Model Saving

- Periodisches Speichern (alle N × log_interval Episoden)
- Finales Speichern nach Training
- Enthält: Q-Network, Target Network, Optimizer State, Epsilon

---

## 5. Design-Entscheidungen und Optimierungen

### Warum DQN?

1. **Discrete Action Space**: Minesweeper hat diskreten Aktionsraum (600 Aktionen)
2. **Value-Based**: Q-Learning passt gut für deterministische Umgebungen
3. **Bewährt**: DQN ist etabliert und gut verstanden

### Warum diese State-Encoding?

1. **Einfachheit**: Ein Kanal, einfache CNN-Architektur
2. **Normalisierung**: Werte zwischen -1 und 1 für besseres Training
3. **Informativ**: Unterscheidet alle wichtigen Zellzustände

### Warum diese Reward-Struktur?

1. **Sparse + Shaped**: Hauptrewards für wichtige Ereignisse, Shaped für Fortschritt
2. **Balance**: Nicht zu viele kleine Rewards (verhindert Overfitting)
3. **Klare Signale**: Gewinn/Verlust sind deutlich signalisiert

### Mögliche Verbesserungen

1. **Dueling DQN**: Trennt State-Value und Advantage
2. **Prioritized Replay**: Wichtige Erfahrungen werden öfter gesampelt
3. **Multi-Step Learning**: N-Step Returns statt 1-Step
4. **State-Encoding**: Mehr Features (z.B. Wahrscheinlichkeiten je Feld)
5. **Curriculum**: Transfer Learning zwischen Schwierigkeitsgraden

---

## 6. Verwendung

### Training starten:

```bash
python -m src.reinforcement_learning.trainer \
    --episodes 1000 \
    --difficulty medium \
    --save-path models/dqn_model.pth
```

### Modell laden:

```python
from src.reinforcement_learning.dqn_agent import DQNAgent
from src.reinforcement_learning.environment import MinesweeperEnvironment

env = MinesweeperEnvironment("medium")
agent = DQNAgent(state_channels=1, action_space_size=600)
agent.load("models/dqn_model.pth")

# Agent spielen lassen
state = env.reset()
done = False
while not done:
    valid_actions = env.get_valid_actions()
    action = agent.select_action(state, valid_actions)
    state, reward, done, info = env.step(action)
```

---

## 7. Zusammenfassung

Das RL-System implementiert:

✅ **Environment Wrapper**: Gymnasium-ähnliches Interface
✅ **State Encoding**: Normalisierte 2D-Repräsentation
✅ **DQN Network**: CNN für räumliche Features
✅ **Experience Replay**: Stabilisiert Training
✅ **Target Network**: Verhindert instabile Updates
✅ **Epsilon-Greedy**: Exploration/Exploitation Balance
✅ **Reward Shaping**: Sparse + Shaped Rewards
✅ **Action Masking**: Verhindert ungültige Aktionen

Das System ist vollständig funktionsfähig und bereit für Training!


```

```markdown
# Reinforcement Learning Implementierungs-Guide

Dieses Dokument erklärt detailliert, wie das Reinforcement Learning System für Minesweeper implementiert ist.

## Übersicht

Das RL-System verwendet **Deep Q-Network (DQN)** mit folgenden Komponenten:
- **Environment**: Wrapper für das Minesweeper-Spiel
- **DQN Network**: Convolutional Neural Network für Q-Value-Schätzung
- **DQN Agent**: Agent mit Experience Replay und Target Network
- **Training**: Episoden-basiertes Training mit Logging

---

## 1. Environment (`src/reinforcement_learning/environment.py`)

### Zweck
Wrappt das Minesweeper-Spiel in ein Gymnasium-ähnliches Interface für RL-Algorithmen.

### Implementierung

#### State Representation (Zustandsrepräsentation)

Der State besitzt **7 Kanäle** und liefert vollständigen Kontext über jede Zelle und ihre Nachbarschaft:

1. **Basis-Encoding** (`state[0]`): alter kombinierter Wert (-0.9 hidden, -0.5 Flag, -1 Mine, 0‑1 Zahl) inkl. Nachbarschaftsbonus/-strafe  
2. **Hidden-Maske** (`state[1]`): 1 für verdeckte Zelle, sonst 0  
3. **Flag-Maske** (`state[2]`): 1 für gesetzte Flagge, sonst 0  
4. **Aufgedeckte Zahl** (`state[3]`): normierte Zahl (0‑1) bzw. -1 bei Mine  
5. **Verdeckte Nachbarn** (`state[4]`): Anteil verdeckter Nachbarn (0‑1)  
6. **Flag-Nachbarn** (`state[5]`): Anteil geflaggter Nachbarn (0‑1)  
7. **Hinweis-Summe** (`state[6]`): normierte Summe aller bekannten Nachbarzahlen (0‑1)

Damit kennt der Agent sowohl den lokalen Status als auch die unmittelbare „Frontier“ und vorhandene Hinweise.

#### Action Space (Reveal vs. Flag)

- **Standard (GUI/Test)**: Reveal + Flag mit Größe `2 × width × height`
- **Training (Standard)**: `use_flag_actions=False` → nur Reveal-Aktionen (`width × height`), analog zum Ansatz aus [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning), damit der Agent sich wie ein menschlicher Spieler auf Progress-Züge konzentriert
- **CLI Flag**: `python -m ...trainer --use-flags` aktiviert wieder das alte Verhalten
- **Indexierung (mit Flags)**:
  - `0 … (N-1)`: Zelle aufdecken
  - `N … (2N-1)`: Flagge toggle
- **Validierung**:
  - Reveal nur für verdeckte, nicht geflaggte Felder
  - Flag nur für verdeckte oder bereits geflaggte Felder

#### Reward System

Skaliert automatisch mit der Brettgröße (`reward_scale = max(1, width*height/100)`):

- **Aufdecken**:
  - `+progress_scale` (mind. 0.5) pro neu aufgedecktem Feld
  - Kettenbonus für Flood-Fill (`+0.4 * progress_scale` pro zusätzlichem Feld)
  - Frontier-Bonus (`+0.6`) sobald angrenzende Hinweise existieren
  - Guess-Penalty (`-0.4 * reward_scale`), falls komplett isoliert geklickt wird
- **Keine Fortschritte**: `-0.15 * progress_scale`
- **Flaggen (optional)**:
  - `+0.2` bei korrekt gesetzter Flagge, `-0.2` bei Fehlflag
  - End-Bonus/-Malus (`±0.5`) für korrekte/inkorrekte Flags bei Spielende
- **Spiel verloren**: `-12 * reward_scale` (starke negative Verstärkung)
- **Spiel gewonnen**: `+18 * reward_scale + 6 * progress_ratio`

#### Valid Actions Masking

Das Environment bietet zwei Methoden für gültige Aktionen:

1. **`get_valid_actions()`**: Boolean-Array (True = gültig)
2. **`get_action_mask()`**: Mask für Q-Values (-inf für ungültig, 0.0 für gültig)

**Warum Masking?**
- Verhindert, dass der Agent bereits aufgedeckte oder flagge Zellen auswählt
- Reduziert den Aktionsraum effektiv
- Verbessert Trainingseffizienz

---

## 2. DQN Network (`src/reinforcement_learning/network.py`)

### Architektur

**Convolutional Neural Network** nach sdlee94-Vorbild:

```
Input: (batch, 7, H, W)

Conv Stack (×4):
- Conv2d(in → 128, kernel=3, padding=1)
- BatchNorm2d(128)
- ReLU

AdaptiveAvgPool2d(8 × 8)  ->  128 × 8 × 8 Features

Fully Connected:
- Linear(8192 → 512) + Dropout(0.25)
- Linear(512 → 512) + Dropout(0.25)
- Linear(512 → num_actions)
```

- **4 × 128er Convs**: stärkere lokale Mustererkennung wie im Referenzprojekt  
- **Adaptive Pooling**: funktioniert auf allen Brettgrößen  
- **Doppelte 512er Dense-Layer**: entspricht `conv128x4_dense512x2` aus [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning)

---

## 3. DQN Agent (`src/reinforcement_learning/dqn_agent.py`)

### Komponenten

#### 3.1 ReplayBuffer

**Zweck**: Speichert Erfahrungen (State, Action, Reward, Next State, Done, Next-Action-Maske) für Experience Replay.

**Implementierung:**
- `deque` mit `maxlen` für automatische Größenbegrenzung
- Speichert zusätzlich die zulässigen Aktionen des Folgezustands zur Maskierung der Ziel-Q-Werte
- Zufälliges Sampling für Batch-Training
- Konvertiert NumPy-Arrays zu PyTorch-Tensoren

**Warum Experience Replay?**
- **Stabilisierung**: Bricht Korrelationen zwischen aufeinanderfolgenden Erfahrungen
- **Efficiency**: Nutzt Erfahrungen mehrfach
- **Diversität**: Batch enthält verschiedene Erfahrungen

#### 3.2 DQN Agent

**Zweck**: Implementiert DQN-Algorithmus mit allen notwendigen Komponenten.

**Hauptkomponenten:**

1. **Q-Network**: Haupt-Netzwerk für Q-Value-Schätzung
2. **Double Target Network**: Double DQN vermeidet Q-Value-Overestimation
3. **Epsilon-Greedy**: Exploration vs. Exploitation
4. **Experience Replay**: Speichert und nutzt vergangene Erfahrungen

**Hyperparameter:**

```python
- Learning Rate: dynamisch (Basis 0.001, skaliert nach Brettgröße)
- Gamma (Discount): 0.95 (≤600 Felder) bzw. 0.98 (größere Boards)
- Epsilon: Linearer Schedule 1.0 → 0.03–0.10 über ~70 % der Episoden
- Loss: Smooth L1 (Huber), Optimizer: Adam
- Buffer Size: 10,000 Erfahrungen
- Batch Size: 32–96 (abhängig von Brettgröße)
- Target Update: Alle 100 Steps (hartes Sync)
- Action Space: Standardmäßig nur Reveal-Aktionen (`use_flag_actions=False`), Flags via CLI aktivierbar
```

**Gezielte Epsilon-Greedy Strategie:**

```python
if random() < epsilon:
    action = sample_frontier(valid_actions, state)  # Bevorzugt Zellen an bekannten Grenzen
else:
    action = argmax(Q(state, valid_only=True))
```

- **Frontier-Sampling**: Zufallsaktionen wählen bevorzugt verdeckte Felder, die an bereits aufgedeckte Zahlen grenzen  
- **Fallback auf Hinweise**: Wenn keine Frontier existiert, wird ein Feld mit dem größten Hinweis-Signal gewählt  
- **Episodenweiser Decay**: `epsilon` wird nach jeder Episode reduziert, nicht nach jedem Schritt

**Training Process (train_step):**

1. **Batch Sampling**: Zufällige Erfahrungen inkl. Aktionsmasken aus Replay Buffer
2. **Current Q-Values**: Q-Network schätzt Q(s, a)
3. **Target Q-Values**: Double DQN – Aktion via Online-Netz wählen, Q-Wert via Target-Netz bewerten
4. **TD Target**: `target = reward + gamma * Q_target(s', argmax_a Q_online(s'))`
5. **Loss**: Smooth L1 (Huber) zwischen Current und Target
6. **Backpropagation**: Gradienten werden berechnet und angewendet
7. **Gradient Clipping**: Verhindert Exploding Gradients (max norm = 1.0)
8. **Epsilon Decay**: Erfolgt episodenweise über `agent.decay_epsilon()`
9. **Target Update**: Alle N Steps wird Target Network aktualisiert

**Warum Target Network?**
- **Stabilität**: Verhindert instabile Q-Value-Updates
- **Convergence**: Hilft beim Konvergieren des Trainings
- **Delayed Updates**: Target Network wird nur periodisch aktualisiert

**Warum Gradient Clipping?**
- **Stabilität**: Verhindert sehr große Gradienten
- **Training**: Stabilisiert den Lernprozess

---

## 4. Training (`src/reinforcement_learning/trainer.py`)

### Training Loop

```python
schedule = LinearSchedule(start=1.0, end=epsilon_floor, duration=int(0.7 * episodes))

for episode in range(episodes):
    state = env.reset()
    done = False
    
    while not done:
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions)
        next_state, reward, done, info = env.step(action)
        next_valid_actions = env.get_valid_actions()
        agent.remember(state, action, reward, next_state, done, next_valid_actions)
        agent.train_step()
        state = next_state
    
    # Linearer Epsilon-Schedule (kein Multiplikationsrauschen mehr)
    agent.epsilon = schedule.value(episode + 1)
    
    if (episode + 1) % log_interval == 0:
        log_statistics()
        greedy_eval(agent, episodes=eval_episodes, epsilon=0.0)
```

### Logging

Alle `log_interval` Episoden werden sowohl Trainings- als auch Evaluationswerte geloggt:
- **Avg Reward / Length**: Mittelwert der letzten `log_interval` Episoden (mit Exploration)
- **Train Win Rate**: Gewinnrate unter aktuellem ε
- **Eval (ε=0)**: Greedy-Gewinnrate und durchschnittliche Zuganzahl aus `--eval-episodes` Testspielen
- **Epsilon**: Wert des linearen Schedules

### Model Saving

- Periodisches Speichern (alle N × log_interval Episoden)
- Finales Speichern nach Training
- Enthält: Q-Network, Target Network, Optimizer State, Epsilon

---

## 5. Design-Entscheidungen und Optimierungen

### Warum DQN?

1. **Discrete Action Space**: Minesweeper hat diskreten Aktionsraum (600 Aktionen)
2. **Value-Based**: Q-Learning passt gut für deterministische Umgebungen
3. **Bewährt**: DQN ist etabliert und gut verstanden

### Warum diese State-Encoding?

1. **Einfachheit**: Ein Kanal, einfache CNN-Architektur
2. **Normalisierung**: Werte zwischen -1 und 1 für besseres Training
3. **Informativ**: Unterscheidet alle wichtigen Zellzustände

### Warum diese Reward-Struktur?

1. **Sparse + Shaped**: Hauptrewards für wichtige Ereignisse, Shaped für Fortschritt
2. **Balance**: Nicht zu viele kleine Rewards (verhindert Overfitting)
3. **Klare Signale**: Gewinn/Verlust sind deutlich signalisiert

### Mögliche Verbesserungen

1. **Dueling DQN**: Trennt State-Value und Advantage
2. **Prioritized Replay**: Wichtige Erfahrungen werden öfter gesampelt
3. **Multi-Step Learning**: N-Step Returns statt 1-Step
4. **State-Encoding**: Mehr Features (z.B. Wahrscheinlichkeiten je Feld)
5. **Curriculum**: Transfer Learning zwischen Schwierigkeitsgraden

---

## 6. Verwendung

### Training starten:

```bash
python -m src.reinforcement_learning.trainer \
    --episodes 1000 \
    --difficulty medium \
    --save-path models/dqn_model.pth
```

### Modell laden:

```python
from src.reinforcement_learning.dqn_agent import DQNAgent
from src.reinforcement_learning.environment import MinesweeperEnvironment

env = MinesweeperEnvironment("medium")
agent = DQNAgent(state_channels=1, action_space_size=600)
agent.load("models/dqn_model.pth")

# Agent spielen lassen
state = env.reset()
done = False
while not done:
    valid_actions = env.get_valid_actions()
    action = agent.select_action(state, valid_actions)
    state, reward, done, info = env.step(action)
```

---

## 7. Zusammenfassung

Das RL-System implementiert:

✅ **Environment Wrapper**: Gymnasium-ähnliches Interface
✅ **State Encoding**: Normalisierte 2D-Repräsentation
✅ **DQN Network**: CNN für räumliche Features
✅ **Experience Replay**: Stabilisiert Training
✅ **Target Network**: Verhindert instabile Updates
✅ **Epsilon-Greedy**: Exploration/Exploitation Balance
✅ **Reward Shaping**: Sparse + Shaped Rewards
✅ **Action Masking**: Verhindert ungültige Aktionen

Das System ist vollständig funktionsfähig und bereit für Training!


```

```markdown
# Reinforcement Learning Implementierungs-Guide

Dieses Dokument erklärt detailliert, wie das Reinforcement Learning System für Minesweeper implementiert ist.

## Übersicht

Das RL-System verwendet **Deep Q-Network (DQN)** mit folgenden Komponenten:
- **Environment**: Wrapper für das Minesweeper-Spiel
- **DQN Network**: Convolutional Neural Network für Q-Value-Schätzung
- **DQN Agent**: Agent mit Experience Replay und Target Network
- **Training**: Episoden-basiertes Training mit Logging

---

## 1. Environment (`src/reinforcement_learning/environment.py`)

### Zweck
Wrappt das Minesweeper-Spiel in ein Gymnasium-ähnliches Interface für RL-Algorithmen.

### Implementierung

#### State Representation (Zustandsrepräsentation)

Der State besitzt **7 Kanäle** und liefert vollständigen Kontext über jede Zelle und ihre Nachbarschaft:

1. **Basis-Encoding** (`state[0]`): alter kombinierter Wert (-0.9 hidden, -0.5 Flag, -1 Mine, 0‑1 Zahl) inkl. Nachbarschaftsbonus/-strafe  
2. **Hidden-Maske** (`state[1]`): 1 für verdeckte Zelle, sonst 0  
3. **Flag-Maske** (`state[2]`): 1 für gesetzte Flagge, sonst 0  
4. **Aufgedeckte Zahl** (`state[3]`): normierte Zahl (0‑1) bzw. -1 bei Mine  
5. **Verdeckte Nachbarn** (`state[4]`): Anteil verdeckter Nachbarn (0‑1)  
6. **Flag-Nachbarn** (`state[5]`): Anteil geflaggter Nachbarn (0‑1)  
7. **Hinweis-Summe** (`state[6]`): normierte Summe aller bekannten Nachbarzahlen (0‑1)

Damit kennt der Agent sowohl den lokalen Status als auch die unmittelbare „Frontier“ und vorhandene Hinweise.

#### Action Space (Reveal vs. Flag)

- **Standard (GUI/Test)**: Reveal + Flag mit Größe `2 × width × height`
- **Training (Standard)**: `use_flag_actions=False` → nur Reveal-Aktionen (`width × height`), analog zum Ansatz aus [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning), damit der Agent sich wie ein menschlicher Spieler auf Progress-Züge konzentriert
- **CLI Flag**: `python -m ...trainer --use-flags` aktiviert wieder das alte Verhalten
- **Indexierung (mit Flags)**:
  - `0 … (N-1)`: Zelle aufdecken
  - `N … (2N-1)`: Flagge toggle
- **Validierung**:
  - Reveal nur für verdeckte, nicht geflaggte Felder
  - Flag nur für verdeckte oder bereits geflaggte Felder

#### Reward System

Skaliert automatisch mit der Brettgröße (`reward_scale = max(1, width*height/100)`):

- **Aufdecken**:
  - `+progress_scale` (mind. 0.5) pro neu aufgedecktem Feld
  - Kettenbonus für Flood-Fill (`+0.4 * progress_scale` pro zusätzlichem Feld)
  - Frontier-Bonus (`+0.6`) sobald angrenzende Hinweise existieren
  - Guess-Penalty (`-0.4 * reward_scale`), falls komplett isoliert geklickt wird
- **Keine Fortschritte**: `-0.15 * progress_scale`
- **Flaggen (optional)**:
  - `+0.2` bei korrekt gesetzter Flagge, `-0.2` bei Fehlflag
  - End-Bonus/-Malus (`±0.5`) für korrekte/inkorrekte Flags bei Spielende
- **Spiel verloren**: `-12 * reward_scale` (starke negative Verstärkung)
- **Spiel gewonnen**: `+18 * reward_scale + 6 * progress_ratio`

#### Valid Actions Masking

Das Environment bietet zwei Methoden für gültige Aktionen:

1. **`get_valid_actions()`**: Boolean-Array (True = gültig)
2. **`get_action_mask()`**: Mask für Q-Values (-inf für ungültig, 0.0 für gültig)

**Warum Masking?**
- Verhindert, dass der Agent bereits aufgedeckte oder flagge Zellen auswählt
- Reduziert den Aktionsraum effektiv
- Verbessert Trainingseffizienz

---

## 2. DQN Network (`src/reinforcement_learning/network.py`)

### Architektur

**Convolutional Neural Network** nach sdlee94-Vorbild:

```
Input: (batch, 7, H, W)

Conv Stack (×4):
- Conv2d(in → 128, kernel=3, padding=1)
- BatchNorm2d(128)
- ReLU

AdaptiveAvgPool2d(8 × 8)  ->  128 × 8 × 8 Features

Fully Connected:
- Linear(8192 → 512) + Dropout(0.25)
- Linear(512 → 512) + Dropout(0.25)
- Linear(512 → num_actions)
```

- **4 × 128er Convs**: stärkere lokale Mustererkennung wie im Referenzprojekt  
- **Adaptive Pooling**: funktioniert auf allen Brettgrößen  
- **Doppelte 512er Dense-Layer**: entspricht `conv128x4_dense512x2` aus [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning)

---

## 3. DQN Agent (`src/reinforcement_learning/dqn_agent.py`)

### Komponenten

#### 3.1 ReplayBuffer

**Zweck**: Speichert Erfahrungen (State, Action, Reward, Next State, Done, Next-Action-Maske) für Experience Replay.

**Implementierung:**
- `deque` mit `maxlen` für automatische Größenbegrenzung
- Speichert zusätzlich die zulässigen Aktionen des Folgezustands zur Maskierung der Ziel-Q-Werte
- Zufälliges Sampling für Batch-Training
- Konvertiert NumPy-Arrays zu PyTorch-Tensoren

**Warum Experience Replay?**
- **Stabilisierung**: Bricht Korrelationen zwischen aufeinanderfolgenden Erfahrungen
- **Efficiency**: Nutzt Erfahrungen mehrfach
- **Diversität**: Batch enthält verschiedene Erfahrungen

#### 3.2 DQN Agent

**Zweck**: Implementiert DQN-Algorithmus mit allen notwendigen Komponenten.

**Hauptkomponenten:**

1. **Q-Network**: Haupt-Netzwerk für Q-Value-Schätzung
2. **Double Target Network**: Double DQN vermeidet Q-Value-Overestimation
3. **Epsilon-Greedy**: Exploration vs. Exploitation
4. **Experience Replay**: Speichert und nutzt vergangene Erfahrungen

**Hyperparameter:**

```python
- Learning Rate: dynamisch (Basis 0.001, skaliert nach Brettgröße)
- Gamma (Discount): 0.95 (≤600 Felder) bzw. 0.98 (größere Boards)
- Epsilon: Linearer Schedule 1.0 → 0.03–0.10 über ~70 % der Episoden
- Loss: Smooth L1 (Huber), Optimizer: Adam
- Buffer Size: 10,000 Erfahrungen
- Batch Size: 32–96 (abhängig von Brettgröße)
- Target Update: Alle 100 Steps (hartes Sync)
- Action Space: Standardmäßig nur Reveal-Aktionen (`use_flag_actions=False`), Flags via CLI aktivierbar
```

**Gezielte Epsilon-Greedy Strategie:**

```python
if random() < epsilon:
    action = sample_frontier(valid_actions, state)  # Bevorzugt Zellen an bekannten Grenzen
else:
    action = argmax(Q(state, valid_only=True))
```

- **Frontier-Sampling**: Zufallsaktionen wählen bevorzugt verdeckte Felder, die an bereits aufgedeckte Zahlen grenzen  
- **Fallback auf Hinweise**: Wenn keine Frontier existiert, wird ein Feld mit dem größten Hinweis-Signal gewählt  
- **Episodenweiser Decay**: `epsilon` wird nach jeder Episode reduziert, nicht nach jedem Schritt

**Training Process (train_step):**

1. **Batch Sampling**: Zufällige Erfahrungen inkl. Aktionsmasken aus Replay Buffer
2. **Current Q-Values**: Q-Network schätzt Q(s, a)
3. **Target Q-Values**: Double DQN – Aktion via Online-Netz wählen, Q-Wert via Target-Netz bewerten
4. **TD Target**: `target = reward + gamma * Q_target(s', argmax_a Q_online(s'))`
5. **Loss**: Smooth L1 (Huber) zwischen Current und Target
6. **Backpropagation**: Gradienten werden berechnet und angewendet
7. **Gradient Clipping**: Verhindert Exploding Gradients (max norm = 1.0)
8. **Epsilon Decay**: Erfolgt episodenweise über `agent.decay_epsilon()`
9. **Target Update**: Alle N Steps wird Target Network aktualisiert

**Warum Target Network?**
- **Stabilität**: Verhindert instabile Q-Value-Updates
- **Convergence**: Hilft beim Konvergieren des Trainings
- **Delayed Updates**: Target Network wird nur periodisch aktualisiert

**Warum Gradient Clipping?**
- **Stabilität**: Verhindert sehr große Gradienten
- **Training**: Stabilisiert den Lernprozess

---

## 4. Training (`src/reinforcement_learning/trainer.py`)

### Training Loop

```python
schedule = LinearSchedule(start=1.0, end=epsilon_floor, duration=int(0.7 * episodes))

for episode in range(episodes):
    state = env.reset()
    done = False
    
    while not done:
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions)
        next_state, reward, done, info = env.step(action)
        next_valid_actions = env.get_valid_actions()
        agent.remember(state, action, reward, next_state, done, next_valid_actions)
        agent.train_step()
        state = next_state
    
    # Linearer Epsilon-Schedule (kein Multiplikationsrauschen mehr)
    agent.epsilon = schedule.value(episode + 1)
    
    if (episode + 1) % log_interval == 0:
        log_statistics()
        greedy_eval(agent, episodes=eval_episodes, epsilon=0.0)
```

### Logging

Alle `log_interval` Episoden werden sowohl Trainings- als auch Evaluationswerte geloggt:
- **Avg Reward / Length**: Mittelwert der letzten `log_interval` Episoden (mit Exploration)
- **Train Win Rate**: Gewinnrate unter aktuellem ε
- **Eval (ε=0)**: Greedy-Gewinnrate und durchschnittliche Zuganzahl aus `--eval-episodes` Testspielen
- **Epsilon**: Wert des linearen Schedules

### Model Saving

- Periodisches Speichern (alle N × log_interval Episoden)
- Finales Speichern nach Training
- Enthält: Q-Network, Target Network, Optimizer State, Epsilon

---

## 5. Design-Entscheidungen und Optimierungen

### Warum DQN?

1. **Discrete Action Space**: Minesweeper hat diskreten Aktionsraum (600 Aktionen)
2. **Value-Based**: Q-Learning passt gut für deterministische Umgebungen
3. **Bewährt**: DQN ist etabliert und gut verstanden

### Warum diese State-Encoding?

1. **Mehrkanal-Features**: Getrennte Masken für Hidden/Flag/Numbers + Nachbarschaftsdichten → Netz erkennt lokale Muster schneller
2. **Normalisierung**: Alle Werte liegen in `[-1, 1]`, kompatibel mit BatchNorm
3. **Frontier-Hinweise**: Zusätzliche Kanäle (Hinweissumme, Hidden-/Flag-Nachbarn) liefern Vorwissen ohne Regeln zu hardcoden

### Warum diese Reward-Struktur?

1. **Sparse + Shaped**: Hauptrewards für wichtige Ereignisse, Shaped für Fortschritt
2. **Balance**: Nicht zu viele kleine Rewards (verhindert Overfitting)
3. **Klare Signale**: Gewinn/Verlust sind deutlich signalisiert

### Mögliche Verbesserungen

1. **Dueling DQN**: Trennt State-Value und Advantage
2. **Prioritized Replay**: Wichtige Erfahrungen werden öfter gesampelt
3. **Multi-Step Learning**: N-Step Returns statt 1-Step
4. **State-Encoding**: Mehr Features (z.B. Wahrscheinlichkeiten je Feld)
5. **Curriculum**: Transfer Learning zwischen Schwierigkeitsgraden

---

## 6. Verwendung

### Training starten:

```bash
python -m src.reinforcement_learning.trainer \
    --episodes 1000 \
    --difficulty medium \
    --save-path models/dqn_model.pth
```

### Modell laden:

```python
from src.reinforcement_learning.dqn_agent import DQNAgent
from src.reinforcement_learning.environment import MinesweeperEnvironment

env = MinesweeperEnvironment("medium")
agent = DQNAgent(state_channels=1, action_space_size=600)
agent.load("models/dqn_model.pth")

# Agent spielen lassen
state = env.reset()
done = False
while not done:
    valid_actions = env.get_valid_actions()
    action = agent.select_action(state, valid_actions)
    state, reward, done, info = env.step(action)
```

---

## 7. Zusammenfassung

Das RL-System implementiert:

✅ **Environment Wrapper**: Gymnasium-ähnliches Interface
✅ **State Encoding**: Normalisierte 2D-Repräsentation
✅ **DQN Network**: CNN für räumliche Features
✅ **Experience Replay**: Stabilisiert Training
✅ **Target Network**: Verhindert instabile Updates
✅ **Epsilon-Greedy**: Exploration/Exploitation Balance
✅ **Reward Shaping**: Sparse + Shaped Rewards
✅ **Action Masking**: Verhindert ungültige Aktionen

Das System ist vollständig funktionsfähig und bereit für Training!


```

```markdown
# Reinforcement Learning Implementierungs-Guide

Dieses Dokument erklärt detailliert, wie das Reinforcement Learning System für Minesweeper implementiert ist.

## Übersicht

Das RL-System verwendet **Deep Q-Network (DQN)** mit folgenden Komponenten:
- **Environment**: Wrapper für das Minesweeper-Spiel
- **DQN Network**: Convolutional Neural Network für Q-Value-Schätzung
- **DQN Agent**: Agent mit Experience Replay und Target Network
- **Training**: Episoden-basiertes Training mit Logging

---

## 1. Environment (`src/reinforcement_learning/environment.py`)

### Zweck
Wrappt das Minesweeper-Spiel in ein Gymnasium-ähnliches Interface für RL-Algorithmen.

### Implementierung

#### State Representation (Zustandsrepräsentation)

Der State besitzt **7 Kanäle** und liefert vollständigen Kontext über jede Zelle und ihre Nachbarschaft:

1. **Basis-Encoding** (`state[0]`): alter kombinierter Wert (-0.9 hidden, -0.5 Flag, -1 Mine, 0‑1 Zahl) inkl. Nachbarschaftsbonus/-strafe  
2. **Hidden-Maske** (`state[1]`): 1 für verdeckte Zelle, sonst 0  
3. **Flag-Maske** (`state[2]`): 1 für gesetzte Flagge, sonst 0  
4. **Aufgedeckte Zahl** (`state[3]`): normierte Zahl (0‑1) bzw. -1 bei Mine  
5. **Verdeckte Nachbarn** (`state[4]`): Anteil verdeckter Nachbarn (0‑1)  
6. **Flag-Nachbarn** (`state[5]`): Anteil geflaggter Nachbarn (0‑1)  
7. **Hinweis-Summe** (`state[6]`): normierte Summe aller bekannten Nachbarzahlen (0‑1)

Damit kennt der Agent sowohl den lokalen Status als auch die unmittelbare „Frontier“ und vorhandene Hinweise.

#### Action Space (Reveal vs. Flag)

- **Standard (GUI/Test)**: Reveal + Flag mit Größe `2 × width × height`
- **Training (Standard)**: `use_flag_actions=False` → nur Reveal-Aktionen (`width × height`), analog zum Ansatz aus [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning), damit der Agent sich wie ein menschlicher Spieler auf Progress-Züge konzentriert
- **CLI Flag**: `python -m ...trainer --use-flags` aktiviert wieder das alte Verhalten
- **Indexierung (mit Flags)**:
  - `0 … (N-1)`: Zelle aufdecken
  - `N … (2N-1)`: Flagge toggle
- **Validierung**:
  - Reveal nur für verdeckte, nicht geflaggte Felder
  - Flag nur für verdeckte oder bereits geflaggte Felder

#### Reward System

Skaliert automatisch mit der Brettgröße (`reward_scale = max(1, width*height/100)`):

- **Aufdecken**:
  - `+progress_scale` (mind. 0.5) pro neu aufgedecktem Feld
  - Kettenbonus für Flood-Fill (`+0.4 * progress_scale` pro zusätzlichem Feld)
  - Frontier-Bonus (`+0.6`) sobald angrenzende Hinweise existieren
  - Guess-Penalty (`-0.4 * reward_scale`), falls komplett isoliert geklickt wird
- **Keine Fortschritte**: `-0.15 * progress_scale`
- **Flaggen (optional)**:
  - `+0.2` bei korrekt gesetzter Flagge, `-0.2` bei Fehlflag
  - End-Bonus/-Malus (`±0.5`) für korrekte/inkorrekte Flags bei Spielende
- **Spiel verloren**: `-12 * reward_scale` (starke negative Verstärkung)
- **Spiel gewonnen**: `+18 * reward_scale + 6 * progress_ratio`

#### Valid Actions Masking

Das Environment bietet zwei Methoden für gültige Aktionen:

1. **`get_valid_actions()`**: Boolean-Array (True = gültig)
2. **`get_action_mask()`**: Mask für Q-Values (-inf für ungültig, 0.0 für gültig)

**Warum Masking?**
- Verhindert, dass der Agent bereits aufgedeckte oder flagge Zellen auswählt
- Reduziert den Aktionsraum effektiv
- Verbessert Trainingseffizienz

---

## 2. DQN Network (`src/reinforcement_learning/network.py`)

### Architektur

**Convolutional Neural Network** nach sdlee94-Vorbild:

```
Input: (batch, 7, H, W)

Conv Stack (×4):
- Conv2d(in → 128, kernel=3, padding=1)
- BatchNorm2d(128)
- ReLU

AdaptiveAvgPool2d(8 × 8)  ->  128 × 8 × 8 Features

Fully Connected:
- Linear(8192 → 512) + Dropout(0.25)
- Linear(512 → 512) + Dropout(0.25)
- Linear(512 → num_actions)
```

- **4 × 128er Convs**: stärkere lokale Mustererkennung wie im Referenzprojekt  
- **Adaptive Pooling**: funktioniert auf allen Brettgrößen  
- **Doppelte 512er Dense-Layer**: entspricht `conv128x4_dense512x2` aus [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning)

---

## 3. DQN Agent (`src/reinforcement_learning/dqn_agent.py`)

### Komponenten

#### 3.1 ReplayBuffer

**Zweck**: Speichert Erfahrungen (State, Action, Reward, Next State, Done, Next-Action-Maske) für Experience Replay.

**Implementierung:**
- `deque` mit `maxlen` für automatische Größenbegrenzung
- Speichert zusätzlich die zulässigen Aktionen des Folgezustands zur Maskierung der Ziel-Q-Werte
- Zufälliges Sampling für Batch-Training
- Konvertiert NumPy-Arrays zu PyTorch-Tensoren

**Warum Experience Replay?**
- **Stabilisierung**: Bricht Korrelationen zwischen aufeinanderfolgenden Erfahrungen
- **Efficiency**: Nutzt Erfahrungen mehrfach
- **Diversität**: Batch enthält verschiedene Erfahrungen

#### 3.2 DQN Agent

**Zweck**: Implementiert DQN-Algorithmus mit allen notwendigen Komponenten.

**Hauptkomponenten:**

1. **Q-Network**: Haupt-Netzwerk für Q-Value-Schätzung
2. **Double Target Network**: Double DQN vermeidet Q-Value-Overestimation
3. **Epsilon-Greedy**: Exploration vs. Exploitation
4. **Experience Replay**: Speichert und nutzt vergangene Erfahrungen

**Hyperparameter:**

```python
- Learning Rate: dynamisch (Basis 0.001, skaliert nach Brettgröße)
- Gamma (Discount): 0.95 (≤600 Felder) bzw. 0.98 (größere Boards)
- Epsilon: Linearer Schedule 1.0 → 0.03–0.10 über ~70 % der Episoden
- Loss: Smooth L1 (Huber), Optimizer: Adam
- Buffer Size: 10,000 Erfahrungen
- Batch Size: 32–96 (abhängig von Brettgröße)
- Target Update: Alle 100 Steps (hartes Sync)
- Action Space: Standardmäßig nur Reveal-Aktionen (`use_flag_actions=False`), Flags via CLI aktivierbar
```

**Gezielte Epsilon-Greedy Strategie:**

```python
if random() < epsilon:
    action = sample_frontier(valid_actions, state)  # Bevorzugt Zellen an bekannten Grenzen
else:
    action = argmax(Q(state, valid_only=True))
```

- **Frontier-Sampling**: Zufallsaktionen wählen bevorzugt verdeckte Felder, die an bereits aufgedeckte Zahlen grenzen  
- **Fallback auf Hinweise**: Wenn keine Frontier existiert, wird ein Feld mit dem größten Hinweis-Signal gewählt  
- **Episodenweiser Decay**: `epsilon` wird nach jeder Episode reduziert, nicht nach jedem Schritt

**Training Process (train_step):**

1. **Batch Sampling**: Zufällige Erfahrungen inkl. Aktionsmasken aus Replay Buffer
2. **Current Q-Values**: Q-Network schätzt Q(s, a)
3. **Target Q-Values**: Double DQN – Aktion via Online-Netz wählen, Q-Wert via Target-Netz bewerten
4. **TD Target**: `target = reward + gamma * Q_target(s', argmax_a Q_online(s'))`
5. **Loss**: Smooth L1 (Huber) zwischen Current und Target
6. **Backpropagation**: Gradienten werden berechnet und angewendet
7. **Gradient Clipping**: Verhindert Exploding Gradients (max norm = 1.0)
8. **Epsilon Decay**: Erfolgt episodenweise über `agent.decay_epsilon()`
9. **Target Update**: Alle N Steps wird Target Network aktualisiert

**Warum Target Network?**
- **Stabilität**: Verhindert instabile Q-Value-Updates
- **Convergence**: Hilft beim Konvergieren des Trainings
- **Delayed Updates**: Target Network wird nur periodisch aktualisiert

**Warum Gradient Clipping?**
- **Stabilität**: Verhindert sehr große Gradienten
- **Training**: Stabilisiert den Lernprozess

---

## 4. Training (`src/reinforcement_learning/trainer.py`)

### Training Loop

```python
schedule = LinearSchedule(start=1.0, end=epsilon_floor, duration=int(0.7 * episodes))

for episode in range(episodes):
    state = env.reset()
    done = False
    
    while not done:
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions)
        next_state, reward, done, info = env.step(action)
        next_valid_actions = env.get_valid_actions()
        agent.remember(state, action, reward, next_state, done, next_valid_actions)
        agent.train_step()
        state = next_state
    
    # Linearer Epsilon-Schedule (kein Multiplikationsrauschen mehr)
    agent.epsilon = schedule.value(episode + 1)
    
    if (episode + 1) % log_interval == 0:
        log_statistics()
        greedy_eval(agent, episodes=eval_episodes, epsilon=0.0)
```

### Logging

Alle `log_interval` Episoden werden sowohl Trainings- als auch Evaluationswerte geloggt:
- **Avg Reward / Length**: Mittelwert der letzten `log_interval` Episoden (mit Exploration)
- **Train Win Rate**: Gewinnrate unter aktuellem ε
- **Eval (ε=0)**: Greedy-Gewinnrate und durchschnittliche Zuganzahl aus `--eval-episodes` Testspielen
- **Epsilon**: Wert des linearen Schedules

### Model Saving

- Periodisches Speichern (alle N × log_interval Episoden)
- Finales Speichern nach Training
- Enthält: Q-Network, Target Network, Optimizer State, Epsilon

---

## 5. Design-Entscheidungen und Optimierungen

### Warum DQN?

1. **Discrete Action Space**: Minesweeper besitzt einen klar abgegrenzten Aktionsraum (`width × height` Zellen, optional Flags)
2. **Value-Based**: Q-Learning passt gut für deterministische Umgebungen
3. **Bewährt**: DQN ist etabliert und gut verstanden

### Warum diese State-Encoding?

1. **Mehrkanal-Features**: Getrennte Masken für Hidden/Flag/Numbers + Nachbarschaftsdichten → Netz erkennt lokale Muster schneller
2. **Normalisierung**: Alle Werte liegen in `[-1, 1]`, kompatibel mit BatchNorm
3. **Frontier-Hinweise**: Zusätzliche Kanäle (Hinweissumme, Hidden-/Flag-Nachbarn) liefern Vorwissen ohne Regeln zu hardcoden

### Warum diese Reward-Struktur?

1. **Sparse + Shaped**: Hauptrewards für wichtige Ereignisse, Shaped für Fortschritt
2. **Balance**: Nicht zu viele kleine Rewards (verhindert Overfitting)
3. **Klare Signale**: Gewinn/Verlust sind deutlich signalisiert

### Mögliche Verbesserungen

1. **Dueling DQN**: Trennt State-Value und Advantage
2. **Prioritized Replay**: Wichtige Erfahrungen werden öfter gesampelt
3. **Multi-Step Learning**: N-Step Returns statt 1-Step
4. **State-Encoding**: Mehr Features (z.B. Wahrscheinlichkeiten je Feld)
5. **Curriculum**: Transfer Learning zwischen Schwierigkeitsgraden

---

## 6. Verwendung

### Training starten:

```bash
python -m src.reinforcement_learning.trainer \
    --episodes 1000 \
    --difficulty medium \
    --save-path models/dqn_model.pth
```

### Modell laden:

```python
from src.reinforcement_learning.dqn_agent import DQNAgent
from src.reinforcement_learning.environment import MinesweeperEnvironment

env = MinesweeperEnvironment("medium")
agent = DQNAgent(state_channels=1, action_space_size=600)
agent.load("models/dqn_model.pth")

# Agent spielen lassen
state = env.reset()
done = False
while not done:
    valid_actions = env.get_valid_actions()
    action = agent.select_action(state, valid_actions)
    state, reward, done, info = env.step(action)
```

---

## 7. Zusammenfassung

Das RL-System implementiert:

✅ **Environment Wrapper**: Gymnasium-ähnliches Interface
✅ **State Encoding**: Normalisierte 2D-Repräsentation
✅ **DQN Network**: CNN für räumliche Features
✅ **Experience Replay**: Stabilisiert Training
✅ **Target Network**: Verhindert instabile Updates
✅ **Epsilon-Greedy**: Exploration/Exploitation Balance
✅ **Reward Shaping**: Sparse + Shaped Rewards
✅ **Action Masking**: Verhindert ungültige Aktionen

Das System ist vollständig funktionsfähig und bereit für Training!


```

```markdown
# Reinforcement Learning Implementierungs-Guide

Dieses Dokument erklärt detailliert, wie das Reinforcement Learning System für Minesweeper implementiert ist.

## Übersicht

Das RL-System verwendet **Deep Q-Network (DQN)** mit folgenden Komponenten:
- **Environment**: Wrapper für das Minesweeper-Spiel
- **DQN Network**: Convolutional Neural Network für Q-Value-Schätzung
- **DQN Agent**: Agent mit Experience Replay und Target Network
- **Training**: Episoden-basiertes Training mit Logging

---

## 1. Environment (`src/reinforcement_learning/environment.py`)

### Zweck
Wrappt das Minesweeper-Spiel in ein Gymnasium-ähnliches Interface für RL-Algorithmen.

### Implementierung

#### State Representation (Zustandsrepräsentation)

Der State besitzt **7 Kanäle** und liefert vollständigen Kontext über jede Zelle und ihre Nachbarschaft:

1. **Basis-Encoding** (`state[0]`): alter kombinierter Wert (-0.9 hidden, -0.5 Flag, -1 Mine, 0‑1 Zahl) inkl. Nachbarschaftsbonus/-strafe  
2. **Hidden-Maske** (`state[1]`): 1 für verdeckte Zelle, sonst 0  
3. **Flag-Maske** (`state[2]`): 1 für gesetzte Flagge, sonst 0  
4. **Aufgedeckte Zahl** (`state[3]`): normierte Zahl (0‑1) bzw. -1 bei Mine  
5. **Verdeckte Nachbarn** (`state[4]`): Anteil verdeckter Nachbarn (0‑1)  
6. **Flag-Nachbarn** (`state[5]`): Anteil geflaggter Nachbarn (0‑1)  
7. **Hinweis-Summe** (`state[6]`): normierte Summe aller bekannten Nachbarzahlen (0‑1)

Damit kennt der Agent sowohl den lokalen Status als auch die unmittelbare „Frontier“ und vorhandene Hinweise.

#### Action Space (Reveal vs. Flag)

- **Standard (GUI/Test)**: Reveal + Flag mit Größe `2 × width × height`
- **Training (Standard)**: `use_flag_actions=False` → nur Reveal-Aktionen (`width × height`), analog zum Ansatz aus [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning), damit der Agent sich wie ein menschlicher Spieler auf Progress-Züge konzentriert
- **CLI Flag**: `python -m ...trainer --use-flags` aktiviert wieder das alte Verhalten
- **Indexierung (mit Flags)**:
  - `0 … (N-1)`: Zelle aufdecken
  - `N … (2N-1)`: Flagge toggle
- **Validierung**:
  - Reveal nur für verdeckte, nicht geflaggte Felder
  - Flag nur für verdeckte oder bereits geflaggte Felder

#### Reward System

Skaliert automatisch mit der Brettgröße (`reward_scale = max(1, width*height/100)`):

- **Aufdecken**:
  - `+progress_scale` (mind. 0.5) pro neu aufgedecktem Feld
  - Kettenbonus für Flood-Fill (`+0.4 * progress_scale` pro zusätzlichem Feld)
  - Frontier-Bonus (`+0.6`) sobald angrenzende Hinweise existieren
  - Guess-Penalty (`-0.4 * reward_scale`), falls komplett isoliert geklickt wird
- **Keine Fortschritte**: `-0.15 * progress_scale`
- **Flaggen (optional)**:
  - `+0.2` bei korrekt gesetzter Flagge, `-0.2` bei Fehlflag
  - End-Bonus/-Malus (`±0.5`) für korrekte/inkorrekte Flags bei Spielende
- **Spiel verloren**: `-12 * reward_scale` (starke negative Verstärkung)
- **Spiel gewonnen**: `+18 * reward_scale + 6 * progress_ratio`

#### Valid Actions Masking

Das Environment bietet zwei Methoden für gültige Aktionen:

1. **`get_valid_actions()`**: Boolean-Array (True = gültig)
2. **`get_action_mask()`**: Mask für Q-Values (-inf für ungültig, 0.0 für gültig)

**Warum Masking?**
- Verhindert, dass der Agent bereits aufgedeckte oder flagge Zellen auswählt
- Reduziert den Aktionsraum effektiv
- Verbessert Trainingseffizienz

---

## 2. DQN Network (`src/reinforcement_learning/network.py`)

### Architektur

**Convolutional Neural Network** nach sdlee94-Vorbild:

```
Input: (batch, 7, H, W)

Conv Stack (×4):
- Conv2d(in → 128, kernel=3, padding=1)
- BatchNorm2d(128)
- ReLU

AdaptiveAvgPool2d(8 × 8)  ->  128 × 8 × 8 Features

Fully Connected:
- Linear(8192 → 512) + Dropout(0.25)
- Linear(512 → 512) + Dropout(0.25)
- Linear(512 → num_actions)
```

- **4 × 128er Convs**: stärkere lokale Mustererkennung wie im Referenzprojekt  
- **Adaptive Pooling**: funktioniert auf allen Brettgrößen  
- **Doppelte 512er Dense-Layer**: entspricht `conv128x4_dense512x2` aus [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning)

---

## 3. DQN Agent (`src/reinforcement_learning/dqn_agent.py`)

### Komponenten

#### 3.1 ReplayBuffer

**Zweck**: Speichert Erfahrungen (State, Action, Reward, Next State, Done, Next-Action-Maske) für Experience Replay.

**Implementierung:**
- `deque` mit `maxlen` für automatische Größenbegrenzung
- Speichert zusätzlich die zulässigen Aktionen des Folgezustands zur Maskierung der Ziel-Q-Werte
- Zufälliges Sampling für Batch-Training
- Konvertiert NumPy-Arrays zu PyTorch-Tensoren

**Warum Experience Replay?**
- **Stabilisierung**: Bricht Korrelationen zwischen aufeinanderfolgenden Erfahrungen
- **Efficiency**: Nutzt Erfahrungen mehrfach
- **Diversität**: Batch enthält verschiedene Erfahrungen

#### 3.2 DQN Agent

**Zweck**: Implementiert DQN-Algorithmus mit allen notwendigen Komponenten.

**Hauptkomponenten:**

1. **Q-Network**: Haupt-Netzwerk für Q-Value-Schätzung
2. **Double Target Network**: Double DQN vermeidet Q-Value-Overestimation
3. **Epsilon-Greedy**: Exploration vs. Exploitation
4. **Experience Replay**: Speichert und nutzt vergangene Erfahrungen

**Hyperparameter:**

```python
- Learning Rate: dynamisch (Basis 0.001, skaliert nach Brettgröße)
- Gamma (Discount): 0.95 (≤600 Felder) bzw. 0.98 (größere Boards)
- Epsilon: Linearer Schedule 1.0 → 0.03–0.10 über ~70 % der Episoden
- Loss: Smooth L1 (Huber), Optimizer: Adam
- Buffer Size: 10,000 Erfahrungen
- Batch Size: 32–96 (abhängig von Brettgröße)
- Target Update: Alle 100 Steps (hartes Sync)
- Action Space: Standardmäßig nur Reveal-Aktionen (`use_flag_actions=False`), Flags via CLI aktivierbar
```

**Gezielte Epsilon-Greedy Strategie:**

```python
if random() < epsilon:
    action = sample_frontier(valid_actions, state)  # Bevorzugt Zellen an bekannten Grenzen
else:
    action = argmax(Q(state, valid_only=True))
```

- **Frontier-Sampling**: Zufallsaktionen wählen bevorzugt verdeckte Felder, die an bereits aufgedeckte Zahlen grenzen  
- **Fallback auf Hinweise**: Wenn keine Frontier existiert, wird ein Feld mit dem größten Hinweis-Signal gewählt  
- **Episodenweiser Decay**: `epsilon` wird nach jeder Episode reduziert, nicht nach jedem Schritt

**Training Process (train_step):**

1. **Batch Sampling**: Zufällige Erfahrungen inkl. Aktionsmasken aus Replay Buffer
2. **Current Q-Values**: Q-Network schätzt Q(s, a)
3. **Target Q-Values**: Double DQN – Aktion via Online-Netz wählen, Q-Wert via Target-Netz bewerten
4. **TD Target**: `target = reward + gamma * Q_target(s', argmax_a Q_online(s'))`
5. **Loss**: Smooth L1 (Huber) zwischen Current und Target
6. **Backpropagation**: Gradienten werden berechnet und angewendet
7. **Gradient Clipping**: Verhindert Exploding Gradients (max norm = 1.0)
8. **Epsilon Decay**: Erfolgt episodenweise über `agent.decay_epsilon()`
9. **Target Update**: Alle N Steps wird Target Network aktualisiert

**Warum Target Network?**
- **Stabilität**: Verhindert instabile Q-Value-Updates
- **Convergence**: Hilft beim Konvergieren des Trainings
- **Delayed Updates**: Target Network wird nur periodisch aktualisiert

**Warum Gradient Clipping?**
- **Stabilität**: Verhindert sehr große Gradienten
- **Training**: Stabilisiert den Lernprozess

---

## 4. Training (`src/reinforcement_learning/trainer.py`)

### Training Loop

```python
schedule = LinearSchedule(start=1.0, end=epsilon_floor, duration=int(0.7 * episodes))

for episode in range(episodes):
    state = env.reset()
    done = False
    
    while not done:
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions)
        next_state, reward, done, info = env.step(action)
        next_valid_actions = env.get_valid_actions()
        agent.remember(state, action, reward, next_state, done, next_valid_actions)
        agent.train_step()
        state = next_state
    
    # Linearer Epsilon-Schedule (kein Multiplikationsrauschen mehr)
    agent.epsilon = schedule.value(episode + 1)
    
    if (episode + 1) % log_interval == 0:
        log_statistics()
        greedy_eval(agent, episodes=eval_episodes, epsilon=0.0)
```

### Logging

Alle `log_interval` Episoden werden sowohl Trainings- als auch Evaluationswerte geloggt:
- **Avg Reward / Length**: Mittelwert der letzten `log_interval` Episoden (mit Exploration)
- **Train Win Rate**: Gewinnrate unter aktuellem ε
- **Eval (ε=0)**: Greedy-Gewinnrate und durchschnittliche Zuganzahl aus `--eval-episodes` Testspielen
- **Epsilon**: Wert des linearen Schedules

### Model Saving

- Periodisches Speichern (alle N × log_interval Episoden)
- Finales Speichern nach Training
- Enthält: Q-Network, Target Network, Optimizer State, Epsilon

---

## 5. Design-Entscheidungen und Optimierungen

### Warum DQN?

1. **Discrete Action Space**: Minesweeper besitzt einen klar abgegrenzten Aktionsraum (`width × height` Zellen, optional Flags)
2. **Value-Based**: Q-Learning passt gut für deterministische Umgebungen
3. **Bewährt**: DQN ist etabliert und gut verstanden

### Warum diese State-Encoding?

1. **Mehrkanal-Features**: Getrennte Masken für Hidden/Flag/Numbers + Nachbarschaftsdichten → Netz erkennt lokale Muster schneller
2. **Normalisierung**: Alle Werte liegen in `[-1, 1]`, kompatibel mit BatchNorm
3. **Frontier-Hinweise**: Zusätzliche Kanäle (Hinweissumme, Hidden-/Flag-Nachbarn) liefern Vorwissen ohne Regeln zu hardcoden

### Warum diese Reward-Struktur?

1. **Sparse + Shaped**: Hauptrewards für wichtige Ereignisse, Shaped für Fortschritt
2. **Balance**: Nicht zu viele kleine Rewards (verhindert Overfitting)
3. **Klare Signale**: Gewinn/Verlust sind deutlich signalisiert

### Mögliche Verbesserungen

1. **Dueling DQN**: Trennt State-Value und Advantage
2. **Prioritized Replay**: Wichtige Erfahrungen werden öfter gesampelt
3. **Multi-Step Learning**: N-Step Returns statt 1-Step
4. **State-Encoding**: Mehr Features (z.B. Wahrscheinlichkeiten je Feld)
5. **Curriculum**: Transfer Learning zwischen Schwierigkeitsgraden

---

## 6. Verwendung

### Training starten:

```bash
python -m src.reinforcement_learning.trainer \
    --episodes 1500 \
    --difficulty easy \
    --width 9 --height 9 \
    --eval-episodes 25 \
    --save-path models/dqn_model_9x9.pth
```

### Modell laden:

```python
from src.reinforcement_learning.dqn_agent import DQNAgent
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS

env = MinesweeperEnvironment("medium", width=20, height=15, use_flag_actions=False)
agent = DQNAgent(
    state_channels=STATE_CHANNELS,
    action_space_size=env.action_space_size,
    board_height=env.height,
    board_width=env.width
)
agent.load("models/dqn_model.pth")

state = env.reset()
done = False
while not done:
    valid_actions = env.get_valid_actions()
    action = agent.select_action(state, valid_actions)
    state, reward, done, info = env.step(action)
```

---

## 7. Zusammenfassung

Das RL-System implementiert:

✅ **Environment Wrapper**: Gymnasium-ähnliches Interface
✅ **State Encoding**: Normalisierte 2D-Repräsentation
✅ **DQN Network**: CNN für räumliche Features
✅ **Experience Replay**: Stabilisiert Training
✅ **Target Network**: Verhindert instabile Updates
✅ **Epsilon-Greedy**: Exploration/Exploitation Balance
✅ **Reward Shaping**: Sparse + Shaped Rewards
✅ **Action Masking**: Verhindert ungültige Aktionen

Das System ist vollständig funktionsfähig und bereit für Training!


```

```markdown
# Implementierungs-Guide: Minesweeper mit RL

Dieses Dokument erklärt die Implementierung jedes Teils des Projekts und die Design-Entscheidungen.

## Phase 1: Projekt-Setup und Grundstruktur

### Was wurde gemacht:

1. **Projektstruktur erstellt** - Modulare Struktur:
   - `src/minesweeper/` - Spiellogik (Kern)
   - `src/gui/` - GUI-Komponenten
   - `src/reinforcement_learning/` - RL-Implementierung
   - `src/utils/` - Hilfsfunktionen

2. **Konstanten definiert** (`src/utils/constants.py`):
   - Spielfeldgröße: 20x30 (600 Zellen)
   - 3 Schwierigkeitsgrade:
     - Leicht: ~10% Minen (~60 Minen)
     - Mittel: ~15% Minen (~90 Minen) - Standard
     - Schwer: ~20% Minen (~120 Minen)
   - Zellzustände: HIDDEN, REVEALED, FLAGGED, MINE

3. **Dependencies** (`requirements.txt`):
   - PySide6: GUI Framework
   - PyTorch: Deep Learning für RL
   - NumPy: Numerische Operationen

---

## Phase 2: Minesweeper Kern-Logik

### 2.1 Cell-Klasse (`src/minesweeper/cell.py`)

**Zweck:** Repräsentiert eine einzelne Zelle im Spielfeld.

**Implementierung:**
- **Zustandsverwaltung:** Jede Zelle hat einen Zustand (verdeckt, aufgedeckt, markiert)
- **Minen-Markierung:** `set_mine()` markiert die Zelle als Mine
- **Nachbarzählung:** `adjacent_mines` speichert die Anzahl benachbarter Minen
- **Reveal-Logik:** `reveal()` deckt die Zelle auf (nur wenn verdeckt)
- **Flag-Logik:** `flag()` togglet die Flagge (nur wenn verdeckt)

**Design-Entscheidungen:**
- Verwendet Konstanten statt Magic Numbers für bessere Lesbarkeit
- Rückgabewerte (`True`/`False`) zeigen an, ob Operation erfolgreich war
- Getter-Methoden (`is_revealed()`, `is_flagged()`, `is_hidden()`) für klare API

**Mögliche Optimierungen:**
- ✅ Aktuell: Gut strukturiert, keine Optimierungen nötig

---

### 2.2 Board-Klasse (`src/minesweeper/board.py`)

**Zweck:** Verwaltet das gesamte Spielfeld und die Minen-Platzierung.

**Implementierung:**
- **Spielfeld-Generation:** Erstellt 2D-Array von Cell-Objekten
- **Minen-Platzierung:** `place_mines()` platziert Minen zufällig (ausschließlich erster Klick)
- **Nachbarzählung:** `_calculate_adjacent_mines()` berechnet für jede Zelle die Anzahl benachbarter Minen
- **Nachbar-Abfrage:** `get_neighbors()` gibt alle 8 Nachbarzellen zurück

**Design-Entscheidungen:**
- **Lazy Mine Placement:** Minen werden erst beim ersten Klick platziert (verhindert sofortigen Verlust)
- **Mine-Positions-Tracking:** `mine_positions` Set speichert alle Minen-Positionen für schnellen Zugriff
- **Grenzenprüfung:** `get_cell()` prüft Array-Grenzen und gibt `None` für ungültige Positionen zurück

**Mögliche Optimierungen:**
- ✅ Aktuell: Effizient implementiert
- 💡 Potenzial: Caching von Nachbarzellen für sehr große Spielfelder (aktuell nicht nötig bei 20x30)

---

### 2.3 Game-Klasse (`src/minesweeper/game.py`)

**Zweck:** Verwaltet die gesamte Spiellogik und Spielzustand.

**Implementierung:**
- **Spielzustand:** PLAYING, WON, LOST
- **Erster Klick:** Triggert Minen-Platzierung (ausschließlich geklickter Zelle)
- **Aufdecken:** `reveal_cell()` prüft auf Mine, deckt auf, prüft Gewinn
- **Auto-Aufdecken:** `_auto_reveal_safe_neighbors()` deckt automatisch sichere Nachbarn auf (BFS-Algorithmus)
- **Flaggen:** `toggle_flag()` setzt/entfernt Flaggen
- **Schwierigkeitsgrade:** Dynamische Minen-Anzahl basierend auf Schwierigkeit

**Design-Entscheidungen:**
- **Auto-Aufdecken:** BFS (Breadth-First Search) für effizientes Aufdecken von Bereichen mit 0 Minen
- **Erster Klick:** Garantiert, dass erste Zelle sicher ist (keine Mine)
- **State Management:** Klare Zustandsverwaltung mit `GameState` Enumeration

**Mögliche Optimierungen:**
- ✅ Aktuell: Gut implementiert
- 💡 Potenzial: 
  - Timer-Integration (bereits in GUI vorhanden)
  - Highscore-System
  - Hint-System für schwierige Situationen

---

## Phase 3: GUI Implementation

### 3.1 GameBoard Widget (`src/gui/game_board.py`)

**Zweck:** Zeigt das Spielfeld als interaktive Buttons an.

**Implementierung:**
- **Custom Button:** `CellButton` erbt von `QPushButton` mit Signal-System
- **Grid-Layout:** 20x30 Grid von Buttons
- **Interaktion:** Linksklick = Aufdecken, Rechtsklick = Flagge
- **Visualisierung:** Farbcodierung für Zahlen, Icons für Minen/Flaggen

**Design-Entscheidungen:**
- **Signals:** Verwendet PySide6 Signals für lose Kopplung
- **Update-Mechanismus:** `_update_display()` aktualisiert alle Buttons basierend auf Spielzustand

---

### 3.2 MainWindow (`src/gui/main_window.py`)

**Zweck:** Hauptfenster der Anwendung.

**Implementierung:**
- **Menu-Bar:** Schwierigkeitsgrade, Neues Spiel
- **Status-Bar:** Minen-Zähler, Timer
- **Game-Board:** Integriertes Spielfeld
- **Event-Handling:** Gewinn/Verlust-Meldungen

---

## Phase 4: Reinforcement Learning

### 4.1 Environment (`src/reinforcement_learning/environment.py`)

**Zweck:** Gym-ähnlicher Wrapper rund um die Minesweeper-Logik.

**Implementierung (aktuelle Version):**
- **State Representation:** 7 Kanäle (Hidden-, Flag-, Zahlenmasken, Nachbarschaftsdichten und Hinweissumme). Alle Werte liegen in `[-1, 1]`.
- **Action Space:** Standardmäßig `width × height` (Reveal-only, inspiriert durch [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning)). Flags können per `--use-flags` wieder zugeschaltet werden.
- **Reward System:** Fortschrittsbasierte Skalierung (`reward_scale = max(1, width*height/100)`), starke Verluststrafe (`-12 * scale`), hoher Gewinnbonus (`+18 * scale`). Guess-Klicks erhalten einen Malus, Frontier-Züge Bonuspunkte.
- **Action Masking:** `get_valid_actions()` liefert boolsche Maske; `get_action_mask()` erzeugt -inf für ungültige Aktionen (wird direkt in den Q-Werten verwendet).

**Tests:** 13 Testfälle (Initialisierung, Reset, Rewards, Masken, Flag-Rewards, usw.)

---

### 4.2 DQN Network (`src/reinforcement_learning/network.py`)

**Zweck:** Convolutional Neural Network für Q-Value-Schätzung.

**Architektur:**
```
Input: (batch, 1, 20, 30)
├── Conv2d(1 → 32, kernel=3, padding=1) + ReLU
├── Conv2d(32 → 64, kernel=3, padding=1) + ReLU
├── Conv2d(64 → 64, kernel=3, padding=1) + ReLU
├── Flatten: 64 × 20 × 30 = 38,400
├── Linear(38,400 → 512) + ReLU
├── Linear(512 → 256) + ReLU
└── Linear(256 → 600)  # Q-values für alle Aktionen
Output: (batch, 600)
```

**Design-Entscheidungen:**
- **CNN für räumliche Features:** Lernt Nachbarschaftsbeziehungen automatisch
- **Padding=1:** Behält Spielfeldgröße bei (20×30 bleibt konstant)
- **Progressive Channels:** 32 → 64 → 64 für mehr Features pro Layer
- **FC Layers:** 512 → 256 → 600 für Entscheidungsfindung
- **Output:** Q-Werte für alle 600 möglichen Aktionen

**Tests:** 7 Testfälle (Initialization, Forward Pass, Gradients, Parameter Count, etc.)

---

### 4.3 DQN Agent (`src/reinforcement_learning/dqn_agent.py`)

**Zweck:** DQN-Agent mit Experience Replay und Target Network.

**Komponenten:**
1. **ReplayBuffer:** Speichert Erfahrungen (State, Action, Reward, Next State, Done)
2. **Q-Network:** Haupt-Netzwerk für Q-Value-Schätzung
3. **Target Network:** Stabilisiert Training durch verzögerte Updates (alle 100 Steps)
4. **Epsilon-Greedy:** Exploration (ε=1.0) → Exploitation (ε=0.01, decay=0.995)
5. **Training:** MSE Loss, Adam Optimizer (lr=0.001), Gradient Clipping (max_norm=1.0)

**Hyperparameter:**
- Learning Rate: 0.001
- Gamma (Discount Factor): 0.99
- Epsilon: 1.0 → 0.01 (exponentielles Decay: 0.995)
- Buffer Size: 10,000 Erfahrungen
- Batch Size: 32 Erfahrungen pro Training
- Target Update: Alle 100 Training Steps

**Training Process:**
1. Batch Sampling aus Replay Buffer
2. Current Q-Values: Q(s, a) vom Q-Network
3. Target Q-Values: Q(s', a') vom Target Network
4. TD Target: `target = reward + gamma * max(Q(s', a'))`
5. Loss: MSE zwischen Current und Target
6. Backpropagation mit Gradient Clipping
7. Epsilon Decay
8. Periodisches Target Network Update

**Design-Entscheidungen:**
- **Experience Replay:** Stabilisiert Training, bricht Korrelationen
- **Target Network:** Verhindert instabile Q-Value-Updates
- **Epsilon-Greedy:** Balance zwischen Exploration und Exploitation
- **Action Masking:** Berücksichtigt nur gültige Aktionen (-1e9 für ungültig)
- **Gradient Clipping:** Verhindert Exploding Gradients

**Tests:** 13 Testfälle (Initialization, Action Selection, Training, Save/Load, Integration, etc.)

---

## Tests

### Test-Struktur:

```
tests/
├── minesweeper/
│   ├── test_cell.py      # Cell-Klasse Tests (9 Tests)
│   ├── test_board.py     # Board-Klasse Tests (7 Tests)
│   └── test_game.py      # Game-Klasse Tests (10 Tests)
├── reinforcement_learning/
│   ├── test_environment.py    # Environment Tests (13 Tests)
│   ├── test_network.py        # DQN Network Tests (7 Tests)
│   └── test_dqn_agent.py      # DQN Agent Tests (13 Tests)
└── run_tests.py          # Test-Runner
```

### Test-Statistik:

- **Gesamt:** 57 Tests
- **Minesweeper:** 24 Tests
- **Reinforcement Learning:** 33 Tests
- **Alle Tests:** ✅ Bestanden

### Tests ausführen:

```bash
python tests/run_tests.py
# oder
python -m pytest tests/
```

---

## Zusammenfassung der Design-Entscheidungen

1. **Modulare Struktur:** Klare Trennung von Spiellogik, GUI und RL
2. **Lazy Mine Placement:** Minen werden erst beim ersten Klick platziert
3. **Auto-Aufdecken:** BFS-Algorithmus für benutzerfreundliches Spiel
4. **State Management:** Klare Zustandsverwaltung mit Enumerationen
5. **Signal-basierte GUI:** Lose Kopplung zwischen GUI und Spiellogik
6. **RL Environment:** Gymnasium-ähnliches Interface für Wiederverwendbarkeit


```

```markdown
# Implementierungs-Guide: Minesweeper mit RL

Dieses Dokument erklärt die Implementierung jedes Teils des Projekts und die Design-Entscheidungen.

## Phase 1: Projekt-Setup und Grundstruktur

### Was wurde gemacht:

1. **Projektstruktur erstellt** - Modulare Struktur:
   - `src/minesweeper/` - Spiellogik (Kern)
   - `src/gui/` - GUI-Komponenten
   - `src/reinforcement_learning/` - RL-Implementierung
   - `src/utils/` - Hilfsfunktionen

2. **Konstanten definiert** (`src/utils/constants.py`):
   - Spielfeldgröße: 20x30 (600 Zellen)
   - 3 Schwierigkeitsgrade:
     - Leicht: ~10% Minen (~60 Minen)
     - Mittel: ~15% Minen (~90 Minen) - Standard
     - Schwer: ~20% Minen (~120 Minen)
   - Zellzustände: HIDDEN, REVEALED, FLAGGED, MINE

3. **Dependencies** (`requirements.txt`):
   - PySide6: GUI Framework
   - PyTorch: Deep Learning für RL
   - NumPy: Numerische Operationen

---

## Phase 2: Minesweeper Kern-Logik

### 2.1 Cell-Klasse (`src/minesweeper/cell.py`)

**Zweck:** Repräsentiert eine einzelne Zelle im Spielfeld.

**Implementierung:**
- **Zustandsverwaltung:** Jede Zelle hat einen Zustand (verdeckt, aufgedeckt, markiert)
- **Minen-Markierung:** `set_mine()` markiert die Zelle als Mine
- **Nachbarzählung:** `adjacent_mines` speichert die Anzahl benachbarter Minen
- **Reveal-Logik:** `reveal()` deckt die Zelle auf (nur wenn verdeckt)
- **Flag-Logik:** `flag()` togglet die Flagge (nur wenn verdeckt)

**Design-Entscheidungen:**
- Verwendet Konstanten statt Magic Numbers für bessere Lesbarkeit
- Rückgabewerte (`True`/`False`) zeigen an, ob Operation erfolgreich war
- Getter-Methoden (`is_revealed()`, `is_flagged()`, `is_hidden()`) für klare API

**Mögliche Optimierungen:**
- ✅ Aktuell: Gut strukturiert, keine Optimierungen nötig

---

### 2.2 Board-Klasse (`src/minesweeper/board.py`)

**Zweck:** Verwaltet das gesamte Spielfeld und die Minen-Platzierung.

**Implementierung:**
- **Spielfeld-Generation:** Erstellt 2D-Array von Cell-Objekten
- **Minen-Platzierung:** `place_mines()` platziert Minen zufällig (ausschließlich erster Klick)
- **Nachbarzählung:** `_calculate_adjacent_mines()` berechnet für jede Zelle die Anzahl benachbarter Minen
- **Nachbar-Abfrage:** `get_neighbors()` gibt alle 8 Nachbarzellen zurück

**Design-Entscheidungen:**
- **Lazy Mine Placement:** Minen werden erst beim ersten Klick platziert (verhindert sofortigen Verlust)
- **Mine-Positions-Tracking:** `mine_positions` Set speichert alle Minen-Positionen für schnellen Zugriff
- **Grenzenprüfung:** `get_cell()` prüft Array-Grenzen und gibt `None` für ungültige Positionen zurück

**Mögliche Optimierungen:**
- ✅ Aktuell: Effizient implementiert
- 💡 Potenzial: Caching von Nachbarzellen für sehr große Spielfelder (aktuell nicht nötig bei 20x30)

---

### 2.3 Game-Klasse (`src/minesweeper/game.py`)

**Zweck:** Verwaltet die gesamte Spiellogik und Spielzustand.

**Implementierung:**
- **Spielzustand:** PLAYING, WON, LOST
- **Erster Klick:** Triggert Minen-Platzierung (ausschließlich geklickter Zelle)
- **Aufdecken:** `reveal_cell()` prüft auf Mine, deckt auf, prüft Gewinn
- **Auto-Aufdecken:** `_auto_reveal_safe_neighbors()` deckt automatisch sichere Nachbarn auf (BFS-Algorithmus)
- **Flaggen:** `toggle_flag()` setzt/entfernt Flaggen
- **Schwierigkeitsgrade:** Dynamische Minen-Anzahl basierend auf Schwierigkeit

**Design-Entscheidungen:**
- **Auto-Aufdecken:** BFS (Breadth-First Search) für effizientes Aufdecken von Bereichen mit 0 Minen
- **Erster Klick:** Garantiert, dass erste Zelle sicher ist (keine Mine)
- **State Management:** Klare Zustandsverwaltung mit `GameState` Enumeration

**Mögliche Optimierungen:**
- ✅ Aktuell: Gut implementiert
- 💡 Potenzial: 
  - Timer-Integration (bereits in GUI vorhanden)
  - Highscore-System
  - Hint-System für schwierige Situationen

---

## Phase 3: GUI Implementation

### 3.1 GameBoard Widget (`src/gui/game_board.py`)

**Zweck:** Zeigt das Spielfeld als interaktive Buttons an.

**Implementierung:**
- **Custom Button:** `CellButton` erbt von `QPushButton` mit Signal-System
- **Grid-Layout:** 20x30 Grid von Buttons
- **Interaktion:** Linksklick = Aufdecken, Rechtsklick = Flagge
- **Visualisierung:** Farbcodierung für Zahlen, Icons für Minen/Flaggen

**Design-Entscheidungen:**
- **Signals:** Verwendet PySide6 Signals für lose Kopplung
- **Update-Mechanismus:** `_update_display()` aktualisiert alle Buttons basierend auf Spielzustand

---

### 3.2 MainWindow (`src/gui/main_window.py`)

**Zweck:** Hauptfenster der Anwendung.

**Implementierung:**
- **Menu-Bar:** Schwierigkeitsgrade, Neues Spiel
- **Status-Bar:** Minen-Zähler, Timer
- **Game-Board:** Integriertes Spielfeld
- **Event-Handling:** Gewinn/Verlust-Meldungen

---

## Phase 4: Reinforcement Learning

### 4.1 Environment (`src/reinforcement_learning/environment.py`)

**Zweck:** Gym-ähnlicher Wrapper rund um die Minesweeper-Logik.

**Implementierung (aktuelle Version):**
- **State Representation:** 7 Kanäle (Hidden-, Flag-, Zahlenmasken, Nachbarschaftsdichten und Hinweissumme). Alle Werte liegen in `[-1, 1]`.
- **Action Space:** Standardmäßig `width × height` (Reveal-only, inspiriert durch [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning)). Flags können per `--use-flags` wieder zugeschaltet werden.
- **Reward System:** Fortschrittsbasierte Skalierung (`reward_scale = max(1, width*height/100)`), starke Verluststrafe (`-12 * scale`), hoher Gewinnbonus (`+18 * scale`). Guess-Klicks erhalten einen Malus, Frontier-Züge Bonuspunkte.
- **Action Masking:** `get_valid_actions()` liefert boolsche Maske; `get_action_mask()` erzeugt -inf für ungültige Aktionen (wird direkt in den Q-Werten verwendet).

**Tests:** 13 Testfälle (Initialisierung, Reset, Rewards, Masken, Flag-Rewards, usw.)

---

### 4.2 DQN Network (`src/reinforcement_learning/network.py`)

**Zweck:** CNN extrahiert räumliche Muster und gibt Q-Werte für jede erlaubte Aktion zurück.

**Architektur (conv128x4_dense512x2):**
```
Input: (batch, 7, H, W)
├── [Conv2d + BatchNorm + ReLU] × 4   (je 128 Filter, kernel=3, padding=1)
├── AdaptiveAvgPool2d(8 × 8)          (grenzenlos für verschiedene Brettgrößen)
├── Flatten → 128 × 8 × 8 = 8192 Features
├── Linear(8192 → 512) + ReLU + Dropout(0.25)
├── Linear(512 → 512) + ReLU + Dropout(0.25)
└── Linear(512 → num_actions)
```

**Reasoning:**
- 4 tiefe Conv-Blöcke entsprechen dem in [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning) erprobten Setup und verbessern die Frontier-Erkennung.
- Adaptive Pooling sorgt dafür, dass auch 5×5- oder 40×25-Bretter ohne Architekturänderung funktionieren.
- Dropout reduziert Overfitting auf kleinen Boards.

**Tests:** 7 Testfälle (Initialisierung, Vorwärtspass, Gradienten, Parameteranzahl etc.)

---

### 4.3 DQN Agent (`src/reinforcement_learning/dqn_agent.py`)

**Zweck:** DQN-Agent mit Experience Replay und Target Network.

**Komponenten:**
1. **ReplayBuffer:** Speichert Erfahrungen (State, Action, Reward, Next State, Done)
2. **Q-Network:** Haupt-Netzwerk für Q-Value-Schätzung
3. **Target Network:** Stabilisiert Training durch verzögerte Updates (alle 100 Steps)
4. **Epsilon-Greedy:** Exploration (ε=1.0) → Exploitation (ε=0.01, decay=0.995)
5. **Training:** MSE Loss, Adam Optimizer (lr=0.001), Gradient Clipping (max_norm=1.0)

**Hyperparameter:**
- Learning Rate: 0.001
- Gamma (Discount Factor): 0.99
- Epsilon: 1.0 → 0.01 (exponentielles Decay: 0.995)
- Buffer Size: 10,000 Erfahrungen
- Batch Size: 32 Erfahrungen pro Training
- Target Update: Alle 100 Training Steps

**Training Process:**
1. Batch Sampling aus Replay Buffer
2. Current Q-Values: Q(s, a) vom Q-Network
3. Target Q-Values: Q(s', a') vom Target Network
4. TD Target: `target = reward + gamma * max(Q(s', a'))`
5. Loss: MSE zwischen Current und Target
6. Backpropagation mit Gradient Clipping
7. Epsilon Decay
8. Periodisches Target Network Update

**Design-Entscheidungen:**
- **Experience Replay:** Stabilisiert Training, bricht Korrelationen
- **Target Network:** Verhindert instabile Q-Value-Updates
- **Epsilon-Greedy:** Balance zwischen Exploration und Exploitation
- **Action Masking:** Berücksichtigt nur gültige Aktionen (-1e9 für ungültig)
- **Gradient Clipping:** Verhindert Exploding Gradients

**Tests:** 13 Testfälle (Initialization, Action Selection, Training, Save/Load, Integration, etc.)

---

## Tests

### Test-Struktur:

```
tests/
├── minesweeper/
│   ├── test_cell.py      # Cell-Klasse Tests (9 Tests)
│   ├── test_board.py     # Board-Klasse Tests (7 Tests)
│   └── test_game.py      # Game-Klasse Tests (10 Tests)
├── reinforcement_learning/
│   ├── test_environment.py    # Environment Tests (13 Tests)
│   ├── test_network.py        # DQN Network Tests (7 Tests)
│   └── test_dqn_agent.py      # DQN Agent Tests (13 Tests)
└── run_tests.py          # Test-Runner
```

### Test-Statistik:

- **Gesamt:** 57 Tests
- **Minesweeper:** 24 Tests
- **Reinforcement Learning:** 33 Tests
- **Alle Tests:** ✅ Bestanden

### Tests ausführen:

```bash
python tests/run_tests.py
# oder
python -m pytest tests/
```

---

## Zusammenfassung der Design-Entscheidungen

1. **Modulare Struktur:** Klare Trennung von Spiellogik, GUI und RL
2. **Lazy Mine Placement:** Minen werden erst beim ersten Klick platziert
3. **Auto-Aufdecken:** BFS-Algorithmus für benutzerfreundliches Spiel
4. **State Management:** Klare Zustandsverwaltung mit Enumerationen
5. **Signal-basierte GUI:** Lose Kopplung zwischen GUI und Spiellogik
6. **RL Environment:** Gymnasium-ähnliches Interface für Wiederverwendbarkeit


```

```markdown
*** Begin Patch
*** Update File: docs/IMPLEMENTATION_GUIDE.md
@@
-### 4.3 DQN Agent (`src/reinforcement_learning/dqn_agent.py`)
-
-**Zweck:** DQN-Agent mit Experience Replay und Target Network.
-
-**Komponenten:**
-1. **ReplayBuffer:** Speichert Erfahrungen (State, Action, Reward, Next State, Done)
-2. **Q-Network:** Haupt-Netzwerk für Q-Value-Schätzung
-3. **Target Network:** Stabilisiert Training durch verzögerte Updates (alle 100 Steps)
-4. **Epsilon-Greedy:** Exploration (ε=1.0) → Exploitation (ε=0.01, decay=0.995)
-
-**Hyperparameter:**
-- Learning Rate: 0.001
-- Gamma (Discount Factor): 0.99
-- Epsilon: 1.0 → 0.01 (exponentielles Decay: 0.995)
-- Buffer Size: 10,000 Erfahrungen
-- Batch Size: 32 Erfahrungen pro Training
-- Target Update: Alle 100 Training Steps
-
-**Training Process:**
-1. Batch Sampling aus Replay Buffer
-2. Current Q-Values: Q(s, a) vom Q-Network
-3. Target Q-Values: Q(s', a') vom Target Network
-4. TD Target: `target = reward + gamma * max(Q(s', a'))`
-5. Loss: MSE zwischen Current und Target
-6. Backpropagation mit Gradient Clipping
-7. Epsilon Decay
-8. Periodisches Target Network Update
-
-**Design-Entscheidungen:**
-- **Experience Replay:** Stabilisiert Training, bricht Korrelationen
-- **Target Network:** Verhindert instabile Q-Value-Updates
-- **Epsilon-Greedy:** Balance zwischen Exploration und Exploitation
-- **Action Masking:** Berücksichtigt nur gültige Aktionen (-1e9 für ungültig)
-- **Gradient Clipping:** Verhindert Exploding Gradients
-
-**Tests:** 13 Testfälle (Initialization, Action Selection, Training, Save/Load, Integration, etc.)
+### 4.3 DQN Agent (`src/reinforcement_learning/dqn_agent.py`)
+
+**Zweck:** Kern des RL-Systems: Experience Replay + Double DQN + Masking.
+
+**Komponenten:**
+1. **ReplayBuffer:** speichert `(state, action, reward, next_state, done, next_valid_actions)`.
+2. **Q-Network / Target-Network:** identische Architektur, Hard-Update alle 100 Train-Steps.
+3. **Epsilon-Greedy:** Exploration kontrolliert über einen linearen Scheduler (aus dem Trainer gesteuert).
+
+**Hyperparameter / Training:**
+- `lr`: adaptiv (Basis 0.001), Optimizer: Adam
+- `gamma`: 0.95 bzw. 0.98 (große Boards)
+- `loss`: SmoothL1Loss (Huber)
+- `batch_size`: 32–96 (abhängig vom Brett)
+- `target_update`: 100 Schritte
+- `epsilon`: `schedule.value(episode)` (linear 1.0 → 0.03/0.05/0.10)
+- Masking: ungültige Aktionen werden mit `-1e9` in den Q-Werten belegt
+- Gradient Clipping (`max_norm=1.0`)
+
+**Trainingsschritte:**
+1. Sample Batch aus Replay Buffer
+2. Online-Netz berechnet `argmax_a Q(s', a)` über gültige Aktionen
+3. Target-Netz liefert `Q_target(s', argmax_a ...)`
+4. TD-Target mit `gamma` + Reward
+5. SmoothL1-Loss → Backprop → Gradient Clipping
+6. Alle 100 Schritte Target-Netz synchronisieren
+
+**Tests:** 13 Testfälle (Action Selection, Replay Buffer, Save/Load, Integration mit Environment)
*** End Patch
```

```markdown
# Implementierungs-Guide: Minesweeper mit RL

Dieses Dokument erklärt die Implementierung jedes Teils des Projekts und die Design-Entscheidungen.

## Phase 1: Projekt-Setup und Grundstruktur

### Was wurde gemacht:

1. **Projektstruktur erstellt** - Modulare Struktur:
   - `src/minesweeper/` - Spiellogik (Kern)
   - `src/gui/` - GUI-Komponenten
   - `src/reinforcement_learning/` - RL-Implementierung
   - `src/utils/` - Hilfsfunktionen

2. **Konstanten definiert** (`src/utils/constants.py`):
   - Spielfeldgröße: 20x30 (600 Zellen)
   - 3 Schwierigkeitsgrade:
     - Leicht: ~10% Minen (~60 Minen)
     - Mittel: ~15% Minen (~90 Minen) - Standard
     - Schwer: ~20% Minen (~120 Minen)
   - Zellzustände: HIDDEN, REVEALED, FLAGGED, MINE

3. **Dependencies** (`requirements.txt`):
   - PySide6: GUI Framework
   - PyTorch: Deep Learning für RL
   - NumPy: Numerische Operationen

---

## Phase 2: Minesweeper Kern-Logik

### 2.1 Cell-Klasse (`src/minesweeper/cell.py`)

**Zweck:** Repräsentiert eine einzelne Zelle im Spielfeld.

**Implementierung:**
- **Zustandsverwaltung:** Jede Zelle hat einen Zustand (verdeckt, aufgedeckt, markiert)
- **Minen-Markierung:** `set_mine()` markiert die Zelle als Mine
- **Nachbarzählung:** `adjacent_mines` speichert die Anzahl benachbarter Minen
- **Reveal-Logik:** `reveal()` deckt die Zelle auf (nur wenn verdeckt)
- **Flag-Logik:** `flag()` togglet die Flagge (nur wenn verdeckt)

**Design-Entscheidungen:**
- Verwendet Konstanten statt Magic Numbers für bessere Lesbarkeit
- Rückgabewerte (`True`/`False`) zeigen an, ob Operation erfolgreich war
- Getter-Methoden (`is_revealed()`, `is_flagged()`, `is_hidden()`) für klare API

**Mögliche Optimierungen:**
- ✅ Aktuell: Gut strukturiert, keine Optimierungen nötig

---

### 2.2 Board-Klasse (`src/minesweeper/board.py`)

**Zweck:** Verwaltet das gesamte Spielfeld und die Minen-Platzierung.

**Implementierung:**
- **Spielfeld-Generation:** Erstellt 2D-Array von Cell-Objekten
- **Minen-Platzierung:** `place_mines()` platziert Minen zufällig (ausschließlich erster Klick)
- **Nachbarzählung:** `_calculate_adjacent_mines()` berechnet für jede Zelle die Anzahl benachbarter Minen
- **Nachbar-Abfrage:** `get_neighbors()` gibt alle 8 Nachbarzellen zurück

**Design-Entscheidungen:**
- **Lazy Mine Placement:** Minen werden erst beim ersten Klick platziert (verhindert sofortigen Verlust)
- **Mine-Positions-Tracking:** `mine_positions` Set speichert alle Minen-Positionen für schnellen Zugriff
- **Grenzenprüfung:** `get_cell()` prüft Array-Grenzen und gibt `None` für ungültige Positionen zurück

**Mögliche Optimierungen:**
- ✅ Aktuell: Effizient implementiert
- 💡 Potenzial: Caching von Nachbarzellen für sehr große Spielfelder (aktuell nicht nötig bei 20x30)

---

### 2.3 Game-Klasse (`src/minesweeper/game.py`)

**Zweck:** Verwaltet die gesamte Spiellogik und Spielzustand.

**Implementierung:**
- **Spielzustand:** PLAYING, WON, LOST
- **Erster Klick:** Triggert Minen-Platzierung (ausschließlich geklickter Zelle)
- **Aufdecken:** `reveal_cell()` prüft auf Mine, deckt auf, prüft Gewinn
- **Auto-Aufdecken:** `_auto_reveal_safe_neighbors()` deckt automatisch sichere Nachbarn auf (BFS-Algorithmus)
- **Flaggen:** `toggle_flag()` setzt/entfernt Flaggen
- **Schwierigkeitsgrade:** Dynamische Minen-Anzahl basierend auf Schwierigkeit

**Design-Entscheidungen:**
- **Auto-Aufdecken:** BFS (Breadth-First Search) für effizientes Aufdecken von Bereichen mit 0 Minen
- **Erster Klick:** Garantiert, dass erste Zelle sicher ist (keine Mine)
- **State Management:** Klare Zustandsverwaltung mit `GameState` Enumeration

**Mögliche Optimierungen:**
- ✅ Aktuell: Gut implementiert
- 💡 Potenzial: 
  - Timer-Integration (bereits in GUI vorhanden)
  - Highscore-System
  - Hint-System für schwierige Situationen

---

## Phase 3: GUI Implementation

### 3.1 GameBoard Widget (`src/gui/game_board.py`)

**Zweck:** Zeigt das Spielfeld als interaktive Buttons an.

**Implementierung:**
- **Custom Button:** `CellButton` erbt von `QPushButton` mit Signal-System
- **Grid-Layout:** 20x30 Grid von Buttons
- **Interaktion:** Linksklick = Aufdecken, Rechtsklick = Flagge
- **Visualisierung:** Farbcodierung für Zahlen, Icons für Minen/Flaggen

**Design-Entscheidungen:**
- **Signals:** Verwendet PySide6 Signals für lose Kopplung
- **Update-Mechanismus:** `_update_display()` aktualisiert alle Buttons basierend auf Spielzustand

---

### 3.2 MainWindow (`src/gui/main_window.py`)

**Zweck:** Hauptfenster der Anwendung.

**Implementierung:**
- **Menu-Bar:** Schwierigkeitsgrade, Neues Spiel
- **Status-Bar:** Minen-Zähler, Timer
- **Game-Board:** Integriertes Spielfeld
- **Event-Handling:** Gewinn/Verlust-Meldungen

---

## Phase 4: Reinforcement Learning

### 4.1 Environment (`src/reinforcement_learning/environment.py`)

**Zweck:** Gym-ähnlicher Wrapper rund um die Minesweeper-Logik.

**Implementierung (aktuelle Version):**
- **State Representation:** 7 Kanäle (Hidden-, Flag-, Zahlenmasken, Nachbarschaftsdichten und Hinweissumme). Alle Werte liegen in `[-1, 1]`.
- **Action Space:** Standardmäßig `width × height` (Reveal-only, inspiriert durch [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning)). Flags können per `--use-flags` wieder zugeschaltet werden.
- **Reward System:** Fortschrittsbasierte Skalierung (`reward_scale = max(1, width*height/100)`), starke Verluststrafe (`-12 * scale`), hoher Gewinnbonus (`+18 * scale`). Guess-Klicks erhalten einen Malus, Frontier-Züge Bonuspunkte.
- **Action Masking:** `get_valid_actions()` liefert boolsche Maske; `get_action_mask()` erzeugt -inf für ungültige Aktionen (wird direkt in den Q-Werten verwendet).

**Tests:** 13 Testfälle (Initialisierung, Reset, Rewards, Masken, Flag-Rewards, usw.)

---

### 4.2 DQN Network (`src/reinforcement_learning/network.py`)

**Zweck:** CNN extrahiert räumliche Muster und gibt Q-Werte für jede erlaubte Aktion zurück.

**Architektur (conv128x4_dense512x2):**
```
Input: (batch, 7, H, W)
├── [Conv2d + BatchNorm + ReLU] × 4   (je 128 Filter, kernel=3, padding=1)
├── AdaptiveAvgPool2d(8 × 8)          (grenzenlos für verschiedene Brettgrößen)
├── Flatten → 128 × 8 × 8 = 8192 Features
├── Linear(8192 → 512) + ReLU + Dropout(0.25)
├── Linear(512 → 512) + ReLU + Dropout(0.25)
└── Linear(512 → num_actions)
```

**Reasoning:**
- 4 tiefe Conv-Blöcke entsprechen dem in [sdlee94](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning) erprobten Setup und verbessern die Frontier-Erkennung.
- Adaptive Pooling sorgt dafür, dass auch 5×5- oder 40×25-Bretter ohne Architekturänderung funktionieren.
- Dropout reduziert Overfitting auf kleinen Boards.

**Tests:** 7 Testfälle (Initialisierung, Vorwärtspass, Gradienten, Parameteranzahl etc.)

---

### 4.3 DQN Agent (`src/reinforcement_learning/dqn_agent.py`)

**Zweck:** Double-DQN-Agent mit Experience Replay, Masking und linearem Explorations-Schedule.

**Komponenten:**
1. **ReplayBuffer:** `deque` mit max. 10k Einträgen, speichert zusätzlich die zulässigen Aktionen des Folgezustands.
2. **Q-/Target-Network:** identische Netze; Target wird alle 100 Trainingsschritte synchronisiert.
3. **Epsilon-Greedy:** Training verwendet einen linearen Scheduler (1.0 → 0.03/0.05/0.10), gesteuert im Trainer.

**Hyperparameter (abhängig vom Brett):**
- `lr`: Basis 0.001, skaliert für kleinere Bretter leicht nach oben
- `gamma`: 0.95 (≤600 Felder) / 0.98 (größer)
- `batch_size`: 32–96
- `loss`: SmoothL1Loss (Huber)
- `optimizer`: Adam
- `target_update`: alle 100 Steps

**Training Process:**
1. Replay-Sampling + Maskierung ungültiger Aktionen
2. Online-Netz liefert `argmax_a Q(s', a)` nur über gültige Aktionen
3. Target-Netz bewertet diese Aktion (Double DQN)
4. TD-Target = Reward + `gamma * Q_target`
5. Backpropagation + Gradient Clipping (`max_norm=1.0`)
6. Zielnetz-Sync alle 100 Steps
7. Epsilon wird nach jeder Episode via `LinearSchedule` gesetzt (kein Multiplikationsrauschen mehr)

**Design-Entscheidungen:**
- **Frontier-Sampling:** Auch bei Exploration werden Züge nahe bekannter Zahlen bevorzugt.
- **Action Masking:** `-1e9` auf ungültigen Aktionen sorgt dafür, dass `argmax` nie auf bereits aufgedeckte Zellen fällt.
- **Greedy Evaluation:** Während des Trainings werden regelmäßig episodenweise Testläufe mit `ε=0` durchgeführt, um echte Leistung zu messen.

**Tests:** 13 Testfälle (ReplayBuffer, Action Selection, Training Step, Save/Load, Environment-Integration)

---

## Tests

### Test-Struktur:

```
tests/
├── minesweeper/
│   ├── test_cell.py      # Cell-Klasse Tests (9 Tests)
│   ├── test_board.py     # Board-Klasse Tests (7 Tests)
│   └── test_game.py      # Game-Klasse Tests (10 Tests)
├── reinforcement_learning/
│   ├── test_environment.py    # Environment Tests (13 Tests)
│   ├── test_network.py        # DQN Network Tests (7 Tests)
│   └── test_dqn_agent.py      # DQN Agent Tests (13 Tests)
└── run_tests.py          # Test-Runner
```

### Test-Statistik:

- **Gesamt:** 57 Tests
- **Minesweeper:** 24 Tests
- **Reinforcement Learning:** 33 Tests
- **Alle Tests:** ✅ Bestanden

### Tests ausführen:

```bash
python tests/run_tests.py
# oder
python -m pytest tests/
```

---

## Zusammenfassung der Design-Entscheidungen

1. **Modulare Struktur:** Klare Trennung von Spiellogik, GUI und RL
2. **Lazy Mine Placement:** Minen werden erst beim ersten Klick platziert
3. **Auto-Aufdecken:** BFS-Algorithmus für benutzerfreundliches Spiel
4. **State Management:** Klare Zustandsverwaltung mit Enumerationen
5. **Signal-basierte GUI:** Lose Kopplung zwischen GUI und Spiellogik
6. **RL Environment:** Gymnasium-ähnliches Interface für Wiederverwendbarkeit


```

## Überblick
- Rewards, Action-Space und Trainingsablauf wurden komplett überarbeitet, um stärkere Lernsignale und bessere Messbarkeit zu liefern – inspiriert von [sdlee94s Minesweeper-DQN](https://github.com/sdlee94/Minesweeper-AI-Reinforcement-Learning).
- Das Trainingsskript unterstützt nun lineare Epsilon-Schedules, frei wählbare Brettgrößen, optionale Flag-Aktionen sowie automatische greedy Evaluationsläufe (ε = 0) zur echten Leistungsbewertung.
- Alle RL-bezogenen Dokumente (README, Implementation Guide, RL Implementation Guide, RL Training Guide) wurden auf den aktuellen Stand gebracht.

## Details
- **Environment & Rewards:** Reveal-only ist jetzt Standard (`use_flag_actions=False`), Rewards skalieren mit der Brettgröße; Guess-Klicks werden bestraft, Frontier-Züge belohnt, Verlust/Gewinn liefern klare Signale.

```160:210:src/reinforcement_learning/environment.py
        if self.game.is_won():
            return 18.0 * board_scale + 6.0 * progress_ratio

        if cells_revealed > 0:
            base_reward = self.progress_scale * cells_revealed
            chain_bonus = 0.4 * self.progress_scale * max(0, cells_revealed - 1)
            stability_bonus = 0.3 * (0.0 if is_guess else 1.0)
            shaping = 0.4 * frontier_factor
            return base_reward + chain_bonus + stability_bonus + shaping + 0.25 * frontier_bonus
```

- **Trainer:** Neues `LinearSchedule` ersetzt das alte multiplikative Decay, CLI versteht `--width/--height`, `--eval-episodes` und `--use-flags`. Nach jedem Log-Block (und am Ende) wird eine greedy Evaluation gedruckt, sodass auch bei kleinen Brettern eine klare Winrate ohne Exploration sichtbar ist.

```12:76:src/reinforcement_learning/trainer.py
class LinearSchedule:
    def __init__(self, start: float, end: float, duration: int):
        self.start = start
        self.end = end
        self.duration = max(1, duration)

def run_greedy_evaluation(...):
    # setzt agent.epsilon = 0.0, misst Winrate/Steps/R
```

- **Docs & README:** Alle RL-Beschreibungen nennen jetzt 7-Kanal-State, optionalen Action-Space, lineare Exploration, neue Rewards sowie die neuen CLI-Switches. Das RL Training Guide erklärt die zusätzlichen Parameter und die neuen Eval-Ausgaben.
- **GUI / Tests:** RL-Visualizer erkennt automatisch, ob ein geladenes Modell Flags nutzt; Environment-Tests unterscheiden klar zwischen Reveal-only und Flag-Modus.

## Tests
- `python -m pytest tests/reinforcement_learning`

## Nächste Schritte
- Längere Trainingsläufe mit den neuen Einstellungen (inkl. `--eval-episodes`) durchführen, um die greedy Winrate auf gewünschten Brettern zu beobachten.
- Optional `--use-flags` aktivieren, wenn Flag-Strategien ebenfalls gelernt werden sollen.

```python
        if self.game.is_won():
            return 18.0 * board_scale + 6.0 * progress_ratio

        if cells_revealed > 0:
            base_reward = self.progress_scale * cells_revealed
            chain_bonus = 0.4 * self.progress_scale * max(0, cells_revealed - 1)
            stability_bonus = 0.3 * (0.0 if is_guess else 1.0)
            shaping = 0.4 * frontier_factor
            return base_reward + chain_bonus + stability_bonus + shaping + 0.25 * frontier_bonus
```

```python
class LinearSchedule:
    def __init__(self, start: float, end: float, duration: int):
        self.start = start
        self.end = end
        self.duration = max(1, duration)

def run_greedy_evaluation(...):
    # setzt agent.epsilon = 0.0, misst Winrate/Steps/R
```

---

**User**

@powershell (776-790) 

---

**Cursor**

```python
"""Training script for DQN agent."""

import os
import argparse
import numpy as np
from src.reinforcement_learning.dqn_agent import DQNAgent
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
    agent: DQNAgent,
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
            action = agent.select_action(state, valid_actions)
            state, reward, done, info = eval_env.step(action)
            reward_sum += reward
            steps += 1
        
        if info["won"]:
            wins += 1
        total_rewards.append(reward_sum)
        total_lengths.append(steps)
    
    agent.epsilon = original_epsilon
    
    return {
        "episodes": episodes,
        "win_rate": (wins / episodes) * 100.0,
        "avg_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "avg_length": float(np.mean(total_lengths)) if total_lengths else 0.0
    }


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
    eval_episodes: int = 20
):
    """
    Train DQN agent on Minesweeper with visualization support.
    
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
    
    # Adjust hyperparameters based on board size
    # Smaller boards need faster learning, larger boards need more exploration
    board_size_factor = (width * height) / 600.0  # Normalize to default 30x20
    
    # Learning rate scaled by board size (smaller boards learn faster)
    base_lr = 0.001
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
    
    # Batch size tuned per board size
    batch_size = min(96, max(32, int(48 * (1.0 / max(board_size_factor, 0.6)))))
    
    agent = DQNAgent(
        state_channels=env.state_channels,
        action_space_size=env.action_space_size,
        board_height=height,
        board_width=width,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_floor,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size
    )
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    episode_wins = []  # Track wins for each episode
    
    print(f"Starting training for {episodes} episodes...")
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
            
            # Select action
            action = agent.select_action(state, valid_actions)
            
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
    eval_episodes: int = 20
):
    """
    Train DQN agent on Minesweeper (without visualization).
    
    Args:
        episodes: Number of training episodes
        difficulty: Game difficulty
        save_path: Path to save model
        log_interval: Episodes between logging
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
        eval_episodes=eval_episodes
    )


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train DQN agent on Minesweeper")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--difficulty", type=str, default="medium", 
                       choices=["easy", "medium", "hard"], help="Game difficulty")
    parser.add_argument("--save-path", type=str, default="models/dqn_model.pth",
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
    
    args = parser.parse_args()
    
    train(
        episodes=args.episodes,
        difficulty=args.difficulty,
        save_path=args.save_path,
        log_interval=args.log_interval,
        use_flag_actions=args.use_flags,
        width=args.width,
        height=args.height,
        eval_episodes=args.eval_episodes
    )


if __name__ == "__main__":
    main()

```

Fehlerursache war, dass im Trainer nach der Umstellung auf den linearen Epsilon-Schedule keine Variable `epsilon_decay` mehr gesetzt wurde, der `DQNAgent` diese aber weiterhin beim Initialisieren erwartet. Ich habe die (zuvor verwendete) Formel für `epsilon_decay` wieder eingefügt (`(epsilon_floor / epsilon_start) ** (1.0 / decay_span)`), sodass der Agent wie zuvor einen Wert erhält, auch wenn der Schedule inzwischen extern gesteuert wird. Damit läuft `train_with_visualization` wieder ohne `NameError`. 

Bitte `python main.py` erneut ausführen – der Thread-Fehler sollte verschwinden.

