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
    
    def select_action(self, state: np.ndarray, valid_actions: Optional[np.ndarray] = None, game=None) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            valid_actions: Boolean array of valid actions
            game: Game instance (not used in DQNAgent, for compatibility with HybridAgent)
            
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


