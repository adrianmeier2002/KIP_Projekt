# Try to improve RL
_Exported on 11/24/2025 at 17:38:26 GMT+1 from Cursor (2.0.38)_

---

**User**

Aktueller Stand: Ich habe ein Minesweeper gebaut, welches funktioniert. Dazu wollte ich ein Reeinforcement learning Programm machen, welches minesweeper dann lernen und spielen kann. Wie gesagt funktioniert das Minesweeper selbst aber das Reeinforcementlearning noch nicht. Kannst du mir über den kompletten code schauen und versuchen Fehler zu entdecken, warum das RL nicht lernt?

---

**Cursor**

## Analyse Reinforcement Learning

- **Ungültige Aktionen verzerren das Bellman-Ziel**  
  Beim Training wird der Maximalwert über *alle* 600 Aktionen berechnet, ohne die bereits aufgedeckten Felder auszublenden. Dadurch können unmögliche Züge mit künstlich hohen Q-Werten das Ziel dominieren, obwohl die Policy sie später maskiert. Das verhindert Konvergenz, weil die Targets nie realisierbar sind.
  
```170:179:src/reinforcement_learning/dqn_agent.py
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q = next_q_values.max(1)[0]
            target_q = rewards + (1 - dones.float()) * self.gamma * next_q
```

  *Fix*: Speichere bei jeder Transition auch die gültigen Aktionen (oder rekonstruiere sie aus dem Zustand) und maskiere sowohl `q_values` als auch `next_q_values`, z.B. durch Minus-Unendlich wie bereits in `select_action` verwendet. Die vorhandene Environment-Hilfe `get_action_mask()` bleibt sonst ungenutzt.

- **Exploration fällt nach wenigen Episoden praktisch auf 0.2**  
  Die Trainer-Kommentare sprechen von „sehr langsamer“ Epsilon-Abnahme (`0.9995/0.9998`), tatsächlich wird `epsilon` aber nach *jedem* Optimierungsschritt multipliziert und erreicht den Bodenwert schon nach ca. 3 200 Updates (≈5 Episoden auf einem 30×20-Brett). Danach erkundet der Agent nur noch 20 % der Zeit und sammelt kaum neue Erfahrungen.
  
```57:75:src/reinforcement_learning/trainer.py
    # Epsilon decay: MUCH slower decay to allow extensive exploration
    ...
    epsilon_decay = 0.9998 if board_size_factor < 0.5 else 0.9995  # Very slow decay
    ...
        agent = DQNAgent(
            ...
            epsilon_end=0.2,  # Keep high exploration (was 0.1)
            epsilon_decay=epsilon_decay,
            ...
        )
```

```190:197:src/reinforcement_learning/dqn_agent.py
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update target network
        self.update_counter += 1
```

  *Fix*: Entweder decayepsilon nur einmal pro Episode oder verwende einen wesentlich größeren Decay-Faktor (z. B. `exp(log(target/start)/total_updates)`), wenn du pro Schritt updatest. Ansonsten bleibt der Agent in einem frühen, zufälligen Verhaltensmuster stecken.

- **Minenstrafe skaliert genau entgegengesetzt zur Dokumentation**  
  Der Kommentar verspricht geringere Strafen auf kleinen Brettern, tatsächlich wird durch die Division über `max(0.5, size_factor)` das Gegenteil erreicht: 15×10 → ‑200 Punkte, 40×25 → ca. ‑60 Punkte. Damit werden gerade die Trainings-Setups mit kleinerem Suchraum übermäßig bestraft.

```143:158:src/reinforcement_learning/environment.py
        if self.game.is_lost():
            # ... smaller boards = less penalty (fewer cells to learn from)
            base_penalty = -100.0
            size_factor = (self.width * self.height) / 600.0  # Normalize to 30x20
            return base_penalty / max(0.5, size_factor)
```

  *Fix*: Multipliziere statt zu dividieren (`base_penalty * size_factor`) oder dividiere durch `min(1.0, size_factor)`, damit kleinere Boards tatsächlich weniger Strafe auslösen und positive Rewards nicht völlig überlagert werden.

- **State-Encoding bewertet hohe Minenhinweise fälschlich positiv**  
  Für verdeckte Felder erhöht `hint_bonus` den Wert proportional zur Summe der Nachbarzahlen. In Minesweeper bedeutet eine große Zahl jedoch ein hohes Minenrisiko; dein Encoding macht genau diese Felder attraktiver und lenkt den Agenten gezielt in gefährliche Regionen.

```101:126:src/reinforcement_learning/environment.py
                else:
                    ...
                    for dr, dc in [(-1, -1), ...]:
                        ...
                                if not neighbor.is_mine:
                                    adjacent_mines_hint += neighbor.adjacent_mines
                    ...
                    hint_bonus = min(0.3, adjacent_mines_hint / 30.0)  # Normalize hint
                    state[row, col] = -0.9 + neighbor_bonus + hint_bonus
```

  *Fix*: Drehe das Vorzeichen (z. B. `-hint_penalty`) oder normalisiere auf einen Bereich, der kleinere Werte für hohe Hinweise vergibt. Alternativ: nutze zwei Kanäle (Nachbaranzahl separat), damit das Netz selbst die Beziehung lernt.

## Empfehlungen & Nächste Schritte
- Passe die Reward-Skalierung und das Feature-Encoding an, damit kleine Boards als Übungsfeld sinnvoll belohnt werden.
- Erweitere Replay-Einträge um Aktionsmasken oder rekonstruiere sie aus dem Zustand, damit das TD-Ziel nur gültige Folgeaktionen berücksichtigt.
- Überarbeite die Epsilon-Strategie (z. B. episodische Decays + höheres `epsilon_end`) und prüfe nach der Änderung mit kurzen Trainingsläufen, ob die durchschnittliche Belohnung nicht mehr sofort kollabiert.
- Wenn obige Punkte implementiert sind, führe erneut `python -m src.reinforcement_learning.trainer --episodes 200 --difficulty easy` aus, um die Wirkung schnell zu verifizieren, bevor du längere Trainingsläufe startest.

Sobald du Anpassungen vorgenommen hast, helfe ich gerne beim erneuten Review oder bei der Auswertung der neuen Trainingslogs.

```python
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q = next_q_values.max(1)[0]
            target_q = rewards + (1 - dones.float()) * self.gamma * next_q
```

```python
    # Epsilon decay: MUCH slower decay to allow extensive exploration
    ...
    epsilon_decay = 0.9998 if board_size_factor < 0.5 else 0.9995  # Very slow decay
    ...
        agent = DQNAgent(
            ...
            epsilon_end=0.2,  # Keep high exploration (was 0.1)
            epsilon_decay=epsilon_decay,
            ...
        )
```

```python
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Update target network
        self.update_counter += 1
```

```python
        if self.game.is_lost():
            # ... smaller boards = less penalty (fewer cells to learn from)
            base_penalty = -100.0
            size_factor = (self.width * self.height) / 600.0  # Normalize to 30x20
            return base_penalty / max(0.5, size_factor)
```

```python
                else:
                    ...
                    for dr, dc in [(-1, -1), ...]:
                        ...
                                if not neighbor.is_mine:
                                    adjacent_mines_hint += neighbor.adjacent_mines
                    ...
                    hint_bonus = min(0.3, adjacent_mines_hint / 30.0)  # Normalize hint
                    state[row, col] = -0.9 + neighbor_bonus + hint_bonus
```

---

**User**

Passe mir alle empfehlungen, die du gemacht hast an

---

**Cursor**

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
from src.reinforcement_learning.environment import MinesweeperEnvironment


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
        state_channels: int = 1,
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
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
    
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
            # Random action
            if valid_actions is not None:
                valid_indices = np.where(valid_actions)[0]
                return int(random.choice(valid_indices))
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
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.masked_fill(~next_valid_actions, -1e9)
            next_q = next_q_values.max(1)[0]
            target_q = rewards + (1 - dones.float()) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
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
    
    # Learning rate: higher for better learning, scaled by board size
    base_lr = 0.001  # Increased base learning rate
    lr = base_lr * (1.0 + 0.5 * (1.0 / max(board_size_factor, 0.5)))
    
    # Epsilon decay: MUCH slower decay to allow extensive exploration
    # Minesweeper requires heavy exploration - agent needs to try many cells
    # For 1000 episodes: 0.999 means epsilon ~0.37 at end, 0.9995 means ~0.61
    epsilon_decay = 0.9998 if board_size_factor < 0.5 else 0.9995  # Very slow decay
    
    # Batch size: larger for smaller boards (more samples per update)
    batch_size = min(64, max(32, int(32 * (1.0 / board_size_factor))))
    
    agent = DQNAgent(
        state_channels=1,
        action_space_size=env.action_space_size,
        board_height=height,
        board_width=width,
        lr=lr,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.2,  # Keep high exploration (was 0.1)
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

```python
"""RL Environment wrapper for Minesweeper."""

import numpy as np
from typing import Tuple, Optional
from src.minesweeper.game import Game
from src.utils.constants import (
    BOARD_WIDTH, BOARD_HEIGHT,
    CELL_HIDDEN, CELL_REVEALED, CELL_FLAGGED, CELL_MINE
)


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
        self.action_space_size = width * height
    
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
        row = action // self.width
        col = action % self.width
        
        # Get previous state
        prev_revealed = self.game.revealed_count
        
        # Execute action
        valid = self.game.reveal_cell(row, col)
        
        # Calculate reward
        reward = self._calculate_reward(prev_revealed, valid)
        
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
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation with better features.
        
        Returns:
            State array of shape (1, height, width)
        """
        state = np.zeros((self.height, self.width), dtype=np.float32)
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                
                if cell.is_revealed():
                    if cell.is_mine:
                        # Mine revealed (shouldn't happen in normal play)
                        state[row, col] = -1.0
                    else:
                        # Number of adjacent mines (normalized to 0-1)
                        state[row, col] = cell.adjacent_mines / 8.0
                elif cell.is_flagged():
                    # Flagged cell
                    state[row, col] = -0.5
                else:
                    # Hidden cell - but add information about adjacent revealed cells
                    # This helps the agent learn patterns
                    adjacent_revealed = 0
                    adjacent_mines_hint = 0
                    
                    for dr, dc in [(-1, -1), (-1, 0), (-1, 1),
                                   (0, -1),           (0, 1),
                                   (1, -1),  (1, 0),  (1, 1)]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < self.height and 0 <= nc < self.width:
                            neighbor = self.game.board.get_cell(nr, nc)
                            if neighbor.is_revealed():
                                adjacent_revealed += 1
                                if not neighbor.is_mine:
                                    adjacent_mines_hint += neighbor.adjacent_mines
                    
                    # Encode hidden cell with neighborhood information
                    # Base value for hidden, plus bonus for having revealed neighbors
                    # This creates a gradient that helps learning
                    # More revealed neighbors = safer cell (usually)
                    neighbor_bonus = min(0.4, adjacent_revealed * 0.08)  # Max 0.4 bonus
                    # Sum of adjacent mine hints: higher = more likely to be safe
                    hint_penalty = min(0.3, adjacent_mines_hint / 30.0)  # Higher hints = riskier
                    cell_value = -0.9 + neighbor_bonus - hint_penalty
                    state[row, col] = max(-1.0, cell_value)  # Keep encoded range consistent
        
        # Add channel dimension
        state = state[np.newaxis, :, :]
        
        return state
    
    def _calculate_reward(self, prev_revealed: int, valid_action: bool) -> float:
        """
        Calculate reward for the action.
        
        Args:
            prev_revealed: Number of revealed cells before action
            valid_action: Whether the action was valid
            
        Returns:
            Reward value
        """
        if not valid_action:
            return -0.5  # Penalty for invalid action
        
        if self.game.is_lost():
            # Penalty for hitting a mine - balanced to allow learning
            # Too large penalty prevents learning from mistakes
            # Scale with board size: smaller boards = less penalty (fewer cells to learn from)
            base_penalty = -100.0
            size_factor = (self.width * self.height) / 600.0  # Normalize to 30x20
            penalty_scale = max(0.5, size_factor)  # Minimum 50% penalty on tiny boards
            return base_penalty * penalty_scale
        
        if self.game.is_won():
            # Reward based on board size - larger boards = more reward
            base_reward = 1000.0
            size_bonus = (self.width * self.height) / 10.0
            return base_reward + size_bonus
        
        # Reward for revealing cells - LINEAR scaling, not quadratic
        cells_revealed = self.game.revealed_count - prev_revealed
        if cells_revealed > 0:
            # Linear reward: small base reward per cell, capped bonus for multiple cells
            # This prevents quadratic scaling
            base_reward_per_cell = 0.1
            # Small bonus for revealing multiple cells at once (capped)
            multi_cell_bonus = min(0.5, 0.02 * cells_revealed)  # Cap at 0.5
            
            # Linear reward: just sum of cells (no multiplication!)
            reward = base_reward_per_cell * cells_revealed + multi_cell_bonus
            
            # Small progress bonus (linear, not quadratic)
            progress = self.game.revealed_count / max(1, self.game.total_safe_cells)
            progress_bonus = 0.1 * progress  # Much smaller progress bonus
            
            return reward + progress_bonus
        
        # Small negative reward for no progress (to encourage exploration)
        return -0.01
    
    def get_valid_actions(self) -> np.ndarray:
        """
        Get mask of valid actions (cells that can be clicked).
        
        Returns:
            Boolean array of valid actions
        """
        valid = np.zeros(self.action_space_size, dtype=bool)
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                action_idx = row * self.width + col
                
                # Valid if cell is hidden and not flagged
                if cell.is_hidden() and not cell.is_flagged():
                    valid[action_idx] = True
        
        return valid
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get action mask for masking invalid actions in Q-values.
        
        Returns:
            Mask array (1.0 for valid, -inf for invalid)
        """
        valid_actions = self.get_valid_actions()
        mask = np.where(valid_actions, 0.0, -np.inf)
        return mask



```

```python
"""Tests for DQNAgent and ReplayBuffer."""

import unittest
import numpy as np
import torch
import os
import tempfile
from src.reinforcement_learning.dqn_agent import DQNAgent, ReplayBuffer
from src.reinforcement_learning.environment import MinesweeperEnvironment
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class TestReplayBuffer(unittest.TestCase):
    """Test cases for ReplayBuffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ReplayBuffer(capacity=100)
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
    
    def test_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(len(self.buffer), 0)
    
    def test_push(self):
        """Test pushing experiences."""
        state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        action = 0
        reward = 1.0
        next_state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        done = False
        next_valid = np.ones(self.action_space_size, dtype=bool)
        
        self.buffer.push(state, action, reward, next_state, done, next_valid)
        self.assertEqual(len(self.buffer), 1)
    
    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)
        
        for i in range(15):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            next_valid = np.ones(self.action_space_size, dtype=bool)
            buffer.push(state, 0, 1.0, state, False, next_valid)
        
        # Should not exceed capacity
        self.assertEqual(len(buffer), 10)
    
    def test_sample(self):
        """Test sampling from buffer."""
        # Add some experiences
        for i in range(32):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            next_valid = np.ones(self.action_space_size, dtype=bool)
            self.buffer.push(state, i % 10, float(i), state, i % 2 == 0, next_valid)
        
        # Sample batch
        batch_size = 16
        states, actions, rewards, next_states, dones, next_valid_actions = self.buffer.sample(batch_size)
        
        # Check types
        self.assertIsInstance(states, torch.Tensor)
        self.assertIsInstance(actions, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)
        self.assertIsInstance(next_valid_actions, torch.Tensor)
        
        # Check shapes
        self.assertEqual(states.shape[0], batch_size)
        self.assertEqual(actions.shape[0], batch_size)
        self.assertEqual(rewards.shape[0], batch_size)
        self.assertEqual(next_states.shape[0], batch_size)
        self.assertEqual(dones.shape[0], batch_size)
        self.assertEqual(next_valid_actions.shape[0], batch_size)


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQNAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
        self.agent = DQNAgent(
            state_channels=1,
            action_space_size=self.action_space_size,
            lr=0.001,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=1000,
            batch_size=32,
            target_update=100
        )
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.action_space_size, self.action_space_size)
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertIsNotNone(self.agent.q_network)
        self.assertIsNotNone(self.agent.target_network)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.memory)
    
    def test_select_action_random(self):
        """Test action selection in random mode (epsilon=1)."""
        state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        
        # With epsilon=1, should always be random
        actions = [self.agent.select_action(state) for _ in range(10)]
        
        # Should be valid actions
        for action in actions:
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.action_space_size)
    
    def test_select_action_greedy(self):
        """Test action selection in greedy mode (epsilon=0)."""
        self.agent.epsilon = 0.0
        state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        
        action = self.agent.select_action(state)
        
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_space_size)
    
    def test_select_action_with_valid_mask(self):
        """Test action selection with valid actions mask."""
        state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        valid_actions = np.zeros(self.action_space_size, dtype=bool)
        valid_actions[0:10] = True  # Only first 10 actions are valid
        
        self.agent.epsilon = 0.0  # Greedy mode
        action = self.agent.select_action(state, valid_actions)
        
        # Should be one of the valid actions
        self.assertIn(action, range(10))
    
    def test_remember(self):
        """Test storing experiences."""
        state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        action = 5
        reward = 1.0
        next_state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        done = False
        
        initial_len = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done)
        self.assertEqual(len(self.agent.memory), initial_len + 1)
    
    def test_train_step_insufficient_samples(self):
        """Test training with insufficient samples."""
        # Should return None if not enough samples
        loss = self.agent.train_step()
        self.assertIsNone(loss)
    
    def test_train_step_sufficient_samples(self):
        """Test training with sufficient samples."""
        # Add enough experiences
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            action = i % self.action_space_size
            reward = np.random.randn()
            next_state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            done = i % 10 == 0
            
            self.agent.remember(state, action, reward, next_state, done)
        
        # Train step
        loss = self.agent.train_step()
        
        # Should return a loss value
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)  # Loss should be non-negative
    
    def test_epsilon_decay(self):
        """Test epsilon decay."""
        initial_epsilon = self.agent.epsilon
        
        # Train multiple times to trigger epsilon decay
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False)
        
        # Train to trigger epsilon decay
        self.agent.train_step()
        
        # Epsilon should have decayed
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_end)
    
    def test_save_and_load(self):
        """Test saving and loading agent."""
        # Add some experiences and train
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False)
        
        self.agent.train_step()
        initial_epsilon = self.agent.epsilon
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            tmp_path = tmp.name
        
        try:
            self.agent.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Create new agent and load
            new_agent = DQNAgent(
                state_channels=1,
                action_space_size=self.action_space_size
            )
            new_agent.load(tmp_path)
            
            # Check that epsilon was loaded
            self.assertEqual(new_agent.epsilon, initial_epsilon)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_target_network_update(self):
        """Test target network update."""
        initial_update_counter = self.agent.update_counter
        
        # Train multiple times to trigger target update
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False)
        
        # Train enough times to trigger target update
        for _ in range(self.agent.target_update + 1):
            if len(self.agent.memory) >= self.agent.batch_size:
                self.agent.train_step()
        
        # Update counter should have increased
        self.assertGreater(self.agent.update_counter, initial_update_counter)


class TestDQNAgentIntegration(unittest.TestCase):
    """Integration tests for DQNAgent with Environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MinesweeperEnvironment("medium")
        self.agent = DQNAgent(
            state_channels=1,
            action_space_size=self.env.action_space_size,
            buffer_size=100,
            batch_size=16
        )
    
    def test_agent_environment_interaction(self):
        """Test interaction between agent and environment."""
        state = self.env.reset()
        
        # Agent selects action
        valid_actions = self.env.get_valid_actions()
        action = self.agent.select_action(state, valid_actions)
        
        # Environment executes action
        next_state, reward, done, info = self.env.step(action)
        
        # Agent remembers experience
        self.agent.remember(state, action, reward, next_state, done)
        
        # Check that experience was stored
        self.assertGreater(len(self.agent.memory), 0)


if __name__ == "__main__":
    unittest.main()


```

```python
"""Tests for DQNAgent and ReplayBuffer."""

import unittest
import numpy as np
import torch
import os
import tempfile
from src.reinforcement_learning.dqn_agent import DQNAgent, ReplayBuffer
from src.reinforcement_learning.environment import MinesweeperEnvironment
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class TestReplayBuffer(unittest.TestCase):
    """Test cases for ReplayBuffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ReplayBuffer(capacity=100)
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
    
    def test_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(len(self.buffer), 0)
    
    def test_push(self):
        """Test pushing experiences."""
        state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        action = 0
        reward = 1.0
        next_state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        done = False
        next_valid = np.ones(self.action_space_size, dtype=bool)
        
        self.buffer.push(state, action, reward, next_state, done, next_valid)
        self.assertEqual(len(self.buffer), 1)
    
    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)
        
        for i in range(15):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            next_valid = np.ones(self.action_space_size, dtype=bool)
            buffer.push(state, 0, 1.0, state, False, next_valid)
        
        # Should not exceed capacity
        self.assertEqual(len(buffer), 10)
    
    def test_sample(self):
        """Test sampling from buffer."""
        # Add some experiences
        for i in range(32):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            next_valid = np.ones(self.action_space_size, dtype=bool)
            self.buffer.push(state, i % 10, float(i), state, i % 2 == 0, next_valid)
        
        # Sample batch
        batch_size = 16
        states, actions, rewards, next_states, dones, next_valid_actions = self.buffer.sample(batch_size)
        
        # Check types
        self.assertIsInstance(states, torch.Tensor)
        self.assertIsInstance(actions, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)
        self.assertIsInstance(next_valid_actions, torch.Tensor)
        
        # Check shapes
        self.assertEqual(states.shape[0], batch_size)
        self.assertEqual(actions.shape[0], batch_size)
        self.assertEqual(rewards.shape[0], batch_size)
        self.assertEqual(next_states.shape[0], batch_size)
        self.assertEqual(dones.shape[0], batch_size)
        self.assertEqual(next_valid_actions.shape[0], batch_size)


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQNAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
        self.agent = DQNAgent(
            state_channels=1,
            action_space_size=self.action_space_size,
            lr=0.001,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=1000,
            batch_size=32,
            target_update=100
        )
    
    def _all_valid_actions(self):
        """Helper to generate a fully valid action mask."""
        return np.ones(self.action_space_size, dtype=bool)
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.action_space_size, self.action_space_size)
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertIsNotNone(self.agent.q_network)
        self.assertIsNotNone(self.agent.target_network)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.memory)
    
    def test_select_action_random(self):
        """Test action selection in random mode (epsilon=1)."""
        state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        
        # With epsilon=1, should always be random
        actions = [self.agent.select_action(state) for _ in range(10)]
        
        # Should be valid actions
        for action in actions:
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.action_space_size)
    
    def test_select_action_greedy(self):
        """Test action selection in greedy mode (epsilon=0)."""
        self.agent.epsilon = 0.0
        state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        
        action = self.agent.select_action(state)
        
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_space_size)
    
    def test_select_action_with_valid_mask(self):
        """Test action selection with valid actions mask."""
        state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        valid_actions = np.zeros(self.action_space_size, dtype=bool)
        valid_actions[0:10] = True  # Only first 10 actions are valid
        
        self.agent.epsilon = 0.0  # Greedy mode
        action = self.agent.select_action(state, valid_actions)
        
        # Should be one of the valid actions
        self.assertIn(action, range(10))
    
    def test_remember(self):
        """Test storing experiences."""
        state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        action = 5
        reward = 1.0
        next_state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        done = False
        
        initial_len = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        self.assertEqual(len(self.agent.memory), initial_len + 1)
    
    def test_train_step_insufficient_samples(self):
        """Test training with insufficient samples."""
        # Should return None if not enough samples
        loss = self.agent.train_step()
        self.assertIsNone(loss)
    
    def test_train_step_sufficient_samples(self):
        """Test training with sufficient samples."""
        # Add enough experiences
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            action = i % self.action_space_size
            reward = np.random.randn()
            next_state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            done = i % 10 == 0
            
            self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        
        # Train step
        loss = self.agent.train_step()
        
        # Should return a loss value
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)  # Loss should be non-negative
    
    def test_epsilon_decay(self):
        """Test epsilon decay."""
        initial_epsilon = self.agent.epsilon
        
        # Train multiple times to trigger epsilon decay
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train to trigger epsilon decay
        self.agent.train_step()
        
        # Epsilon should stay the same until decay is triggered explicitly
        self.assertEqual(self.agent.epsilon, initial_epsilon)
        
        # After calling decay, epsilon should decrease
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_end)
    
    def test_save_and_load(self):
        """Test saving and loading agent."""
        # Add some experiences and train
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        self.agent.train_step()
        initial_epsilon = self.agent.epsilon
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            tmp_path = tmp.name
        
        try:
            self.agent.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Create new agent and load
            new_agent = DQNAgent(
                state_channels=1,
                action_space_size=self.action_space_size
            )
            new_agent.load(tmp_path)
            
            # Check that epsilon was loaded
            self.assertEqual(new_agent.epsilon, initial_epsilon)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_target_network_update(self):
        """Test target network update."""
        initial_update_counter = self.agent.update_counter
        
        # Train multiple times to trigger target update
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train enough times to trigger target update
        for _ in range(self.agent.target_update + 1):
            if len(self.agent.memory) >= self.agent.batch_size:
                self.agent.train_step()
        
        # Update counter should have increased
        self.assertGreater(self.agent.update_counter, initial_update_counter)


class TestDQNAgentIntegration(unittest.TestCase):
    """Integration tests for DQNAgent with Environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MinesweeperEnvironment("medium")
        self.agent = DQNAgent(
            state_channels=1,
            action_space_size=self.env.action_space_size,
            buffer_size=100,
            batch_size=16
        )
    
    def test_agent_environment_interaction(self):
        """Test interaction between agent and environment."""
        state = self.env.reset()
        
        # Agent selects action
        valid_actions = self.env.get_valid_actions()
        action = self.agent.select_action(state, valid_actions)
        
        # Environment executes action
        next_state, reward, done, info = self.env.step(action)
        
        # Agent remembers experience
        self.agent.remember(state, action, reward, next_state, done)
        
        # Check that experience was stored
        self.assertGreater(len(self.agent.memory), 0)


if __name__ == "__main__":
    unittest.main()


```

```python
"""Tests for DQNAgent and ReplayBuffer."""

import unittest
import numpy as np
import torch
import os
import tempfile
from src.reinforcement_learning.dqn_agent import DQNAgent, ReplayBuffer
from src.reinforcement_learning.environment import MinesweeperEnvironment
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class TestReplayBuffer(unittest.TestCase):
    """Test cases for ReplayBuffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ReplayBuffer(capacity=100)
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
    
    def test_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(len(self.buffer), 0)
    
    def test_push(self):
        """Test pushing experiences."""
        state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        action = 0
        reward = 1.0
        next_state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        done = False
        next_valid = np.ones(self.action_space_size, dtype=bool)
        
        self.buffer.push(state, action, reward, next_state, done, next_valid)
        self.assertEqual(len(self.buffer), 1)
    
    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)
        
        for i in range(15):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            next_valid = np.ones(self.action_space_size, dtype=bool)
            buffer.push(state, 0, 1.0, state, False, next_valid)
        
        # Should not exceed capacity
        self.assertEqual(len(buffer), 10)
    
    def test_sample(self):
        """Test sampling from buffer."""
        # Add some experiences
        for i in range(32):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            next_valid = np.ones(self.action_space_size, dtype=bool)
            self.buffer.push(state, i % 10, float(i), state, i % 2 == 0, next_valid)
        
        # Sample batch
        batch_size = 16
        states, actions, rewards, next_states, dones, next_valid_actions = self.buffer.sample(batch_size)
        
        # Check types
        self.assertIsInstance(states, torch.Tensor)
        self.assertIsInstance(actions, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)
        self.assertIsInstance(next_valid_actions, torch.Tensor)
        
        # Check shapes
        self.assertEqual(states.shape[0], batch_size)
        self.assertEqual(actions.shape[0], batch_size)
        self.assertEqual(rewards.shape[0], batch_size)
        self.assertEqual(next_states.shape[0], batch_size)
        self.assertEqual(dones.shape[0], batch_size)
        self.assertEqual(next_valid_actions.shape[0], batch_size)


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQNAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
        self.agent = DQNAgent(
            state_channels=1,
            action_space_size=self.action_space_size,
            lr=0.001,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=1000,
            batch_size=32,
            target_update=100
        )
    
    def _all_valid_actions(self):
        """Helper to generate a fully valid action mask."""
        return np.ones(self.action_space_size, dtype=bool)
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.action_space_size, self.action_space_size)
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertIsNotNone(self.agent.q_network)
        self.assertIsNotNone(self.agent.target_network)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.memory)
    
    def test_select_action_random(self):
        """Test action selection in random mode (epsilon=1)."""
        state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        
        # With epsilon=1, should always be random
        actions = [self.agent.select_action(state) for _ in range(10)]
        
        # Should be valid actions
        for action in actions:
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.action_space_size)
    
    def test_select_action_greedy(self):
        """Test action selection in greedy mode (epsilon=0)."""
        self.agent.epsilon = 0.0
        state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        
        action = self.agent.select_action(state)
        
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_space_size)
    
    def test_select_action_with_valid_mask(self):
        """Test action selection with valid actions mask."""
        state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        valid_actions = np.zeros(self.action_space_size, dtype=bool)
        valid_actions[0:10] = True  # Only first 10 actions are valid
        
        self.agent.epsilon = 0.0  # Greedy mode
        action = self.agent.select_action(state, valid_actions)
        
        # Should be one of the valid actions
        self.assertIn(action, range(10))
    
    def test_remember(self):
        """Test storing experiences."""
        state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        action = 5
        reward = 1.0
        next_state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        done = False
        
        initial_len = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        self.assertEqual(len(self.agent.memory), initial_len + 1)
    
    def test_train_step_insufficient_samples(self):
        """Test training with insufficient samples."""
        # Should return None if not enough samples
        loss = self.agent.train_step()
        self.assertIsNone(loss)
    
    def test_train_step_sufficient_samples(self):
        """Test training with sufficient samples."""
        # Add enough experiences
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            action = i % self.action_space_size
            reward = np.random.randn()
            next_state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            done = i % 10 == 0
            
            self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        
        # Train step
        loss = self.agent.train_step()
        
        # Should return a loss value
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)  # Loss should be non-negative
    
    def test_epsilon_decay(self):
        """Test epsilon decay."""
        initial_epsilon = self.agent.epsilon
        
        # Train multiple times to trigger epsilon decay
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train to trigger epsilon decay
        self.agent.train_step()
        
        # Epsilon should stay the same until decay is triggered explicitly
        self.assertEqual(self.agent.epsilon, initial_epsilon)
        
        # After calling decay, epsilon should decrease
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_end)
    
    def test_save_and_load(self):
        """Test saving and loading agent."""
        # Add some experiences and train
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        self.agent.train_step()
        initial_epsilon = self.agent.epsilon
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            tmp_path = tmp.name
        
        try:
            self.agent.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Create new agent and load
            new_agent = DQNAgent(
                state_channels=1,
                action_space_size=self.action_space_size
            )
            new_agent.load(tmp_path)
            
            # Check that epsilon was loaded
            self.assertEqual(new_agent.epsilon, initial_epsilon)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_target_network_update(self):
        """Test target network update."""
        initial_update_counter = self.agent.update_counter
        
        # Train multiple times to trigger target update
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train enough times to trigger target update
        for _ in range(self.agent.target_update + 1):
            if len(self.agent.memory) >= self.agent.batch_size:
                self.agent.train_step()
        
        # Update counter should have increased
        self.assertGreater(self.agent.update_counter, initial_update_counter)


class TestDQNAgentIntegration(unittest.TestCase):
    """Integration tests for DQNAgent with Environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MinesweeperEnvironment("medium")
        self.agent = DQNAgent(
            state_channels=1,
            action_space_size=self.env.action_space_size,
            buffer_size=100,
            batch_size=16
        )
    
    def test_agent_environment_interaction(self):
        """Test interaction between agent and environment."""
        state = self.env.reset()
        
        # Agent selects action
        valid_actions = self.env.get_valid_actions()
        action = self.agent.select_action(state, valid_actions)
        
        # Environment executes action
        next_state, reward, done, info = self.env.step(action)
        next_valid_actions = self.env.get_valid_actions()
        
        # Agent remembers experience
        self.agent.remember(state, action, reward, next_state, done, next_valid_actions)
        
        # Check that experience was stored
        self.assertGreater(len(self.agent.memory), 0)


if __name__ == "__main__":
    unittest.main()


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

Der State ist ein 2D-Array mit normalisierten Werten:

```python
State Encoding:
- -0.9: Verdeckte Zelle (hidden)
- -0.5: Flagge gesetzt (flagged)
- -1.0: Mine aufgedeckt (mine revealed)
- 0.0 - 1.0: Anzahl benachbarter Minen (normalisiert auf 0-1)
  - 0/8 = 0.0 (keine Minen)
  - 1/8 = 0.125
  - ...
  - 8/8 = 1.0 (alle Nachbarn sind Minen)
```

**Warum diese Encoding?**
- Negative Werte für verdeckte/flagge Zellen unterscheiden sie von aufgedeckten
- Normalisierte Werte (0-1) für Zahlen verbessern das Training
- Ein Kanal (1, height, width) für einfache CNN-Architektur

#### Action Space

- **Größe**: 600 Aktionen (20 × 30 = alle Zellen)
- **Action Index**: `action = row * BOARD_WIDTH + col`
- **Umkehrung**: `row = action // BOARD_WIDTH`, `col = action % BOARD_WIDTH`

#### Reward System

Das Reward-System ist **sparse** mit **shaped rewards**:

```python
Rewards:
- +100.0: Spiel gewonnen (große Belohnung)
- -10.0:  Mine getroffen (große Strafe)
- +0.1 × N: N Zellen aufgedeckt (kleine Belohnung pro Zelle)
- -0.1: Ungültige Aktion (kleine Strafe)
```

**Design-Entscheidungen:**
- **Sparse Hauptrewards**: Gewinn/Verlust sind die wichtigsten Signale
- **Shaped Rewards**: Fortschritt (aufgedeckte Zellen) wird belohnt
- **Penalties**: Verhindern ungültige Aktionen

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
Input: (batch, 1, 20, 30)  # State representation

Conv Layers:
- Conv2d(1 → 32, kernel=3, padding=1)  # Erste Feature-Ebene
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
2. **Target Network**: Stabilisiert Training durch verzögerte Updates
3. **Epsilon-Greedy**: Exploration vs. Exploitation
4. **Experience Replay**: Speichert und nutzt vergangene Erfahrungen

**Hyperparameter:**

```python
- Learning Rate: 0.001 (Adam Optimizer)
- Gamma (Discount): 0.99 (langfristige Belohnung)
- Epsilon Start: 1.0 (100% Exploration)
- Epsilon End: 0.01 (1% Exploration)
- Epsilon Decay: 0.995 (exponentielles Decay)
- Buffer Size: 10,000 Erfahrungen
- Batch Size: 32 Erfahrungen pro Training
- Target Update: Alle 100 Steps
```

**Epsilon-Greedy Strategie:**

```python
if random() < epsilon:
    # Exploration: Zufällige Aktion
    action = random(valid_actions)
else:
    # Exploitation: Beste Aktion nach Q-Network
    action = argmax(Q(state))
```

**Warum Epsilon-Greedy?**
- **Exploration**: Agent lernt neue Strategien
- **Exploitation**: Agent nutzt gelerntes Wissen
- **Balance**: Epsilon decay von 1.0 → 0.01 über Training

**Training Process (train_step):**

1. **Batch Sampling**: Zufällige Erfahrungen inkl. Aktionsmasken aus Replay Buffer
2. **Current Q-Values**: Q-Network schätzt Q(s, a)
3. **Target Q-Values**: Target Network schätzt Q(s', a') und maskiert ungültige Aktionen
4. **TD Target**: `target = reward + gamma * max(Q(s', a') | a' gültig)`
5. **Loss**: Mean Squared Error zwischen Current und Target
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

## Änderungen
- DQN-Speicher und -Lernschritt speichern jetzt die Action-Masken des Folgezustands, maskieren damit die Target-Q-Werte und entkoppeln die Epsilon-Reduktion vom Schritt- hin zum Episoden-Takt (`decay_epsilon()`), sodass nur noch tatsächlich ausführbare Züge das Bellman-Ziel bestimmen und die Exploration wirklich langsam ausläuft.  
```26:216:src/reinforcement_learning/dqn_agent.py
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
...
        states, actions, rewards, next_states, dones, next_valid_actions = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        next_valid_actions = next_valid_actions.to(self.device)
...
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.masked_fill(~next_valid_actions, -1e9)
            next_q = next_q_values.max(1)[0]
            target_q = rewards + (1 - dones.float()) * self.gamma * next_q
...
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
```
- Trainer und GUI-Workflow sammeln nun nach jedem Schritt sofort die neuen Valid-Action-Masken und rufen erst nach vollständiger Episode `agent.decay_epsilon()`, womit Logging/Visualisierung den richtigen Epsilon-Wert anzeigen.  
```100:125:src/reinforcement_learning/trainer.py
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
...
        # Episode finished: decay epsilon once
        agent.decay_epsilon()
```
- State-Encoding und Rewards wurden korrigiert: hohe Nachbarhinweise wirken jetzt als Strafterm (geclamped auf ≥ -1.0) und die Minenstrafe wächst mit dem Brett statt zu schrumpfen, wodurch kleine Boards wirklich sanfter bestraft werden.  
```122:155:src/reinforcement_learning/environment.py
                    neighbor_bonus = min(0.4, adjacent_revealed * 0.08)  # Max 0.4 bonus
                    # Sum of adjacent mine hints: higher = more likely to be safe
                    hint_penalty = min(0.3, adjacent_mines_hint / 30.0)  # Higher hints = riskier
                    cell_value = -0.9 + neighbor_bonus - hint_penalty
                    state[row, col] = max(-1.0, cell_value)  # Keep encoded range consistent
...
            base_penalty = -100.0
            size_factor = (self.width * self.height) / 600.0  # Normalize to 30x20
            penalty_scale = max(0.5, size_factor)  # Minimum 50% penalty on tiny boards
            return base_penalty * penalty_scale
```
- Alle betroffenen Tests und die Dokumentation wurden auf das neue Replay-Format sowie den episodischen Epsilon-Decay angepasst, damit die Guides und Unittests dieselbe API verwenden.

## Tests
- `python -m pytest tests\reinforcement_learning` *(fehlgeschlagen: Modul `pytest` ist in der Umgebung nicht installiert)*

Bitte `pytest` installieren oder in einer Umgebung mit PyTest erneut ausführen, falls du die Tests lokal verifizieren möchtest.

```python
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
...
        states, actions, rewards, next_states, dones, next_valid_actions = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        next_valid_actions = next_valid_actions.to(self.device)
...
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.masked_fill(~next_valid_actions, -1e9)
            next_q = next_q_values.max(1)[0]
            target_q = rewards + (1 - dones.float()) * self.gamma * next_q
...
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
```

```python
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
...
        # Episode finished: decay epsilon once
        agent.decay_epsilon()
```

```python
                    neighbor_bonus = min(0.4, adjacent_revealed * 0.08)  # Max 0.4 bonus
                    # Sum of adjacent mine hints: higher = more likely to be safe
                    hint_penalty = min(0.3, adjacent_mines_hint / 30.0)  # Higher hints = riskier
                    cell_value = -0.9 + neighbor_bonus - hint_penalty
                    state[row, col] = max(-1.0, cell_value)  # Keep encoded range consistent
...
            base_penalty = -100.0
            size_factor = (self.width * self.height) / 600.0  # Normalize to 30x20
            penalty_scale = max(0.5, size_factor)  # Minimum 50% penalty on tiny boards
            return base_penalty * penalty_scale
```

---

**User**

@powershell (137-210) Das ist mein aktuelles RL. Es performt sehr schlecht und lernt die regeln noch nicht wirklich. In den Visualisierungen habe ich das gefühl, das Programm versucht aktuell immer per Zufall ein Feld auszuwählen. Gerade wenn es ganze flächen freischaltet, fährt es nicht dort fort und versucht Bomben zu erkennen und angernzende Felder ohne Bomben aufzudecken sondern lässt zufällig wieder irgendwelche felder aufdecken. Wie könnte ich das verbessern, dass er auch dies lernt? Wäre es eine Möglichkeit ih für richtig gesetzte flaggen zu belohnen und für falsch gesetzte zu bestrafen? Oder ist dies nicht zielführend? Mache mir zuerst nur vorschläge ohne den code zu ändern, ich entscheide danach, was geändert werden soll.

---

**Cursor**

- **Reward-Shaping gezielt erweitern**  
  *Flaggen belohnen/bestrafen*: Ja, kann helfen – aber nur, wenn Flaggen wirklich Teil des Aktionsraums werden (derzeit klickt der Agent nur). Ohne separate „Flag“-Aktion erzeugst du sonst keine Datenpunkte. Falls du das Aktionsset erweiterst, könntest du z. B. +0.5 für korrekt gesetzte Flaggen, ‑0.5 für falsche Flaggen vergeben und zusätzlich eine kleine Belohnung fürs anschließende sichere Aufdecken, damit das Flaggen mit tatsächlichem Fortschritt verknüpft ist.  
  *Teilbelohnungen für logische Züge*: Gib kleinen Reward, wenn der Agent nach einem großen Flood-Fill bewusst angrenzende sichere Felder wählt (z. B. +0.2, wenn eine „1“ vollständig umschlossen wird). Das lenkt die Policy Richtung deterministische Muster, statt wieder komplett zufällig zu klicken.

- **Curriculum & Startzustände**  
  Beginne mit sehr kleinen Brettern (5×5) und wenigen Minen oder sogar „handgebauten“ Situationen (nur zwei Zahlen mit klarer Lösung). Nach einigen tausend Episoden erhöhst du die Größe bzw. Minendichte. So sammelt der Replay-Buffer überhaupt erfolgreiche Trajektorien, aus denen das Netz logische Folgerungen lernen kann.

- **Exploration gezielt steuern**  
  Obwohl Epsilon jetzt nur episodisch sinkt, bleibt der Effekt gleich: Nach 200 Episoden liegt es bei 0.2 und der Agent klickt dauerhaft zu 20 % zufällig. Du kannst das Decay noch langsamer machen (z. B. 0.9999 pro Episode) oder adaptive Exploration nutzen (höheres Epsilon bei langen Verlustserien, niedriger bei besseren Rewards). Optional: füge „Noisy Layers“ oder Parameter Noise hinzu, damit auch bei kleinem Epsilon neue Strategien entstehen.

- **State-Encoding weiter anreichern**  
  Aktuell sieht das Netz nur einen Kanal. Ergänze zusätzliche Kanäle, z. B. „Anzahl markierter Minen pro Zahl“, „Anzahl verdeckter Nachbarn“, oder eine Binärkodierung (flag, revealed, unsolved). So kann das CNN leichter lokale Muster erkennen (etwa ein „1“ mit einem einzigen verdeckten Nachbarn).

- **Gedächtnis und N-Step-Returns**  
  Minesweeper hat lange Folgen ohne unmittelbare Belohnung. Nutze N-step TD (z. B. n=5) oder sogar Monte-Carlo-Rollouts, damit Fortschritt über mehrere sichere Klicks hinweg bewertet wird. Gleichzeitig lohnt sich „Prioritized Replay“, damit seltene, informative Situationen häufiger trainiert werden.

- **Policy-Bias nach Flood-Fill**  
  Wenn der Agent eine große Fläche aufgedeckt hat, fülle den Aktionsraum nicht sofort wieder komplett mit zufälligen Klicks, sondern erzeuge eine Heuristik-Maske: Z. B. markiere alle Zellen, die direkt an die neuen Zahlen grenzen, als bevorzugte Aktionen (Softmax-Temperature senken). So wird das Netz gezwungen, dort weiterzumachen, wo deterministische Informationen vorhanden sind.

- **Zusätzliche Loss-Kontrollen**  
  Überwache neben dem MSE auch, wie oft der Agent in validierten Situationen deterministisch richtige Aktionen wählt (z. B. bei einer künstlichen Batch von „lehrbuchhaften“ Mustern). Das zeigt, ob das Netz überhaupt Logik versteht oder nur Belohnungen aus Zufall sammelt.

Fazit: Flaggen-Belohnungen sind sinnvoll, wenn du das Aktionsset entsprechend erweiterst. Noch wichtiger ist jedoch, dem Agenten strukturierte Lerngelegenheiten zu geben (Curriculum, zusätzliche Features, deterministische Masken) und die Reward-Signale stärker an logische Fortschritte zu koppeln.

---

**User**

Okey Korrigiere mir also die gezieltere Exploration, und reichere mir vor allem das State encoding weiter an. der Agent soll wissen wie sein umfeld aussieht. 

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
        self.action_space_size = width * height
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
        row = action // self.width
        col = action % self.width
        
        # Get previous state
        prev_revealed = self.game.revealed_count
        
        # Execute action
        valid = self.game.reveal_cell(row, col)
        
        # Calculate reward
        reward = self._calculate_reward(prev_revealed, valid)
        
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
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation with better features.
        
        Returns:
            State array of shape (1, height, width)
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
    
    def _calculate_reward(self, prev_revealed: int, valid_action: bool) -> float:
        """
        Calculate reward for the action.
        
        Args:
            prev_revealed: Number of revealed cells before action
            valid_action: Whether the action was valid
            
        Returns:
            Reward value
        """
        if not valid_action:
            return -0.5  # Penalty for invalid action
        
        if self.game.is_lost():
            # Penalty for hitting a mine - balanced to allow learning
            # Too large penalty prevents learning from mistakes
            # Scale with board size: smaller boards = less penalty (fewer cells to learn from)
            base_penalty = -100.0
            size_factor = (self.width * self.height) / 600.0  # Normalize to 30x20
            penalty_scale = max(0.5, size_factor)  # Minimum 50% penalty on tiny boards
            return base_penalty * penalty_scale
        
        if self.game.is_won():
            # Reward based on board size - larger boards = more reward
            base_reward = 1000.0
            size_bonus = (self.width * self.height) / 10.0
            return base_reward + size_bonus
        
        # Reward for revealing cells - LINEAR scaling, not quadratic
        cells_revealed = self.game.revealed_count - prev_revealed
        if cells_revealed > 0:
            # Linear reward: small base reward per cell, capped bonus for multiple cells
            # This prevents quadratic scaling
            base_reward_per_cell = 0.1
            # Small bonus for revealing multiple cells at once (capped)
            multi_cell_bonus = min(0.5, 0.02 * cells_revealed)  # Cap at 0.5
            
            # Linear reward: just sum of cells (no multiplication!)
            reward = base_reward_per_cell * cells_revealed + multi_cell_bonus
            
            # Small progress bonus (linear, not quadratic)
            progress = self.game.revealed_count / max(1, self.game.total_safe_cells)
            progress_bonus = 0.1 * progress  # Much smaller progress bonus
            
            return reward + progress_bonus
        
        # Small negative reward for no progress (to encourage exploration)
        return -0.01
    
    def get_valid_actions(self) -> np.ndarray:
        """
        Get mask of valid actions (cells that can be clicked).
        
        Returns:
            Boolean array of valid actions
        """
        valid = np.zeros(self.action_space_size, dtype=bool)
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                action_idx = row * self.width + col
                
                # Valid if cell is hidden and not flagged
                if cell.is_hidden() and not cell.is_flagged():
                    valid[action_idx] = True
        
        return valid
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get action mask for masking invalid actions in Q-values.
        
        Returns:
            Mask array (1.0 for valid, -inf for invalid)
        """
        valid_actions = self.get_valid_actions()
        mask = np.where(valid_actions, 0.0, -np.inf)
        return mask



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
        state_channels: int = 1,
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
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
    
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
            # Random action
            if valid_actions is not None:
                valid_indices = np.where(valid_actions)[0]
                return int(random.choice(valid_indices))
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
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.masked_fill(~next_valid_actions, -1e9)
            next_q = next_q_values.max(1)[0]
            target_q = rewards + (1 - dones.float()) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
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
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
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
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.masked_fill(~next_valid_actions, -1e9)
            next_q = next_q_values.max(1)[0]
            target_q = rewards + (1 - dones.float()) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
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
        hidden_mask = state[1] > 0.5
        flagged_mask = state[2] > 0.5 if state.shape[0] > 2 else np.zeros_like(hidden_mask, dtype=bool)
        revealed_mask = ~(hidden_mask | flagged_mask)
        
        valid_matrix = valid_actions.reshape(height, width)
        frontier_mask = np.zeros((height, width), dtype=bool)
        
        for row in range(height):
            for col in range(width):
                if not hidden_mask[row, col] or not valid_matrix[row, col]:
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
            valid_indices = np.where(valid_actions)[0]
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
    
    # Learning rate: higher for better learning, scaled by board size
    base_lr = 0.001  # Increased base learning rate
    lr = base_lr * (1.0 + 0.5 * (1.0 / max(board_size_factor, 0.5)))
    
    # Epsilon decay: MUCH slower decay to allow extensive exploration
    # Minesweeper requires heavy exploration - agent needs to try many cells
    # For 1000 episodes: 0.999 means epsilon ~0.37 at end, 0.9995 means ~0.61
    epsilon_decay = 0.9998 if board_size_factor < 0.5 else 0.9995  # Very slow decay
    
    # Batch size: larger for smaller boards (more samples per update)
    batch_size = min(64, max(32, int(32 * (1.0 / board_size_factor))))
    
    agent = DQNAgent(
        state_channels=env.state_channels,
        action_space_size=env.action_space_size,
        board_height=height,
        board_width=width,
        lr=lr,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.2,  # Keep high exploration (was 0.1)
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
    
    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.difficulty, "medium")
        self.assertEqual(self.env.action_space_size, BOARD_WIDTH * BOARD_HEIGHT)
        self.assertIsNotNone(self.env.game)
    
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
        self.assertEqual(valid_actions.shape, (BOARD_WIDTH * BOARD_HEIGHT,))
        self.assertEqual(valid_actions.dtype, bool)
        
        # Initially all actions should be valid (all cells hidden)
        self.assertTrue(np.all(valid_actions))
        
        # After first action, some should become invalid
        self.env.step(0)
        valid_actions = self.env.get_valid_actions()
        self.assertFalse(np.all(valid_actions))  # At least some should be invalid
    
    def test_get_action_mask(self):
        """Test action masking."""
        state = self.env.reset()
        
        mask = self.env.get_action_mask()
        
        # Check shape
        self.assertEqual(mask.shape, (BOARD_WIDTH * BOARD_HEIGHT,))
        
        # Valid actions should be 0.0, invalid should be -inf
        valid_actions = self.env.get_valid_actions()
        self.assertTrue(np.all(mask[valid_actions] == 0.0))
        self.assertTrue(np.all(np.isinf(mask[~valid_actions])))
    
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
        self.assertIsNotNone(self.network.conv1)
        self.assertIsNotNone(self.network.conv2)
        self.assertIsNotNone(self.network.conv3)
        self.assertIsNotNone(self.network.fc1)
        self.assertIsNotNone(self.network.fc2)
        self.assertIsNotNone(self.network.fc3)
    
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

```python
"""Tests for DQNAgent and ReplayBuffer."""

import unittest
import numpy as np
import torch
import os
import tempfile
from src.reinforcement_learning.dqn_agent import DQNAgent, ReplayBuffer
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class TestReplayBuffer(unittest.TestCase):
    """Test cases for ReplayBuffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ReplayBuffer(capacity=100)
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
    
    def _random_state(self):
        return np.random.randn(STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
    
    def test_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(len(self.buffer), 0)
    
    def test_push(self):
        """Test pushing experiences."""
        state = self._random_state()
        action = 0
        reward = 1.0
        next_state = self._random_state()
        done = False
        next_valid = np.ones(self.action_space_size, dtype=bool)
        
        self.buffer.push(state, action, reward, next_state, done, next_valid)
        self.assertEqual(len(self.buffer), 1)
    
    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)
        
        for i in range(15):
            state = self._random_state()
            next_valid = np.ones(self.action_space_size, dtype=bool)
            buffer.push(state, 0, 1.0, state, False, next_valid)
        
        # Should not exceed capacity
        self.assertEqual(len(buffer), 10)
    
    def test_sample(self):
        """Test sampling from buffer."""
        # Add some experiences
        for i in range(32):
            state = self._random_state()
            next_valid = np.ones(self.action_space_size, dtype=bool)
            self.buffer.push(state, i % 10, float(i), state, i % 2 == 0, next_valid)
        
        # Sample batch
        batch_size = 16
        states, actions, rewards, next_states, dones, next_valid_actions = self.buffer.sample(batch_size)
        
        # Check types
        self.assertIsInstance(states, torch.Tensor)
        self.assertIsInstance(actions, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)
        self.assertIsInstance(next_valid_actions, torch.Tensor)
        
        # Check shapes
        self.assertEqual(states.shape[0], batch_size)
        self.assertEqual(actions.shape[0], batch_size)
        self.assertEqual(rewards.shape[0], batch_size)
        self.assertEqual(next_states.shape[0], batch_size)
        self.assertEqual(dones.shape[0], batch_size)
        self.assertEqual(next_valid_actions.shape[0], batch_size)


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQNAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
        self.agent = DQNAgent(
            state_channels=1,
            action_space_size=self.action_space_size,
            lr=0.001,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=1000,
            batch_size=32,
            target_update=100
        )
    
    def _all_valid_actions(self):
        """Helper to generate a fully valid action mask."""
        return np.ones(self.action_space_size, dtype=bool)
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.action_space_size, self.action_space_size)
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertIsNotNone(self.agent.q_network)
        self.assertIsNotNone(self.agent.target_network)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.memory)
    
    def test_select_action_random(self):
        """Test action selection in random mode (epsilon=1)."""
        state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        
        # With epsilon=1, should always be random
        actions = [self.agent.select_action(state) for _ in range(10)]
        
        # Should be valid actions
        for action in actions:
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.action_space_size)
    
    def test_select_action_greedy(self):
        """Test action selection in greedy mode (epsilon=0)."""
        self.agent.epsilon = 0.0
        state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        
        action = self.agent.select_action(state)
        
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_space_size)
    
    def test_select_action_with_valid_mask(self):
        """Test action selection with valid actions mask."""
        state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        valid_actions = np.zeros(self.action_space_size, dtype=bool)
        valid_actions[0:10] = True  # Only first 10 actions are valid
        
        self.agent.epsilon = 0.0  # Greedy mode
        action = self.agent.select_action(state, valid_actions)
        
        # Should be one of the valid actions
        self.assertIn(action, range(10))
    
    def test_remember(self):
        """Test storing experiences."""
        state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        action = 5
        reward = 1.0
        next_state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        done = False
        
        initial_len = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        self.assertEqual(len(self.agent.memory), initial_len + 1)
    
    def test_train_step_insufficient_samples(self):
        """Test training with insufficient samples."""
        # Should return None if not enough samples
        loss = self.agent.train_step()
        self.assertIsNone(loss)
    
    def test_train_step_sufficient_samples(self):
        """Test training with sufficient samples."""
        # Add enough experiences
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            action = i % self.action_space_size
            reward = np.random.randn()
            next_state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            done = i % 10 == 0
            
            self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        
        # Train step
        loss = self.agent.train_step()
        
        # Should return a loss value
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)  # Loss should be non-negative
    
    def test_epsilon_decay(self):
        """Test epsilon decay."""
        initial_epsilon = self.agent.epsilon
        
        # Train multiple times to trigger epsilon decay
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train to trigger epsilon decay
        self.agent.train_step()
        
        # Epsilon should stay the same until decay is triggered explicitly
        self.assertEqual(self.agent.epsilon, initial_epsilon)
        
        # After calling decay, epsilon should decrease
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_end)
    
    def test_save_and_load(self):
        """Test saving and loading agent."""
        # Add some experiences and train
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        self.agent.train_step()
        initial_epsilon = self.agent.epsilon
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            tmp_path = tmp.name
        
        try:
            self.agent.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Create new agent and load
            new_agent = DQNAgent(
                state_channels=1,
                action_space_size=self.action_space_size
            )
            new_agent.load(tmp_path)
            
            # Check that epsilon was loaded
            self.assertEqual(new_agent.epsilon, initial_epsilon)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_target_network_update(self):
        """Test target network update."""
        initial_update_counter = self.agent.update_counter
        
        # Train multiple times to trigger target update
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train enough times to trigger target update
        for _ in range(self.agent.target_update + 1):
            if len(self.agent.memory) >= self.agent.batch_size:
                self.agent.train_step()
        
        # Update counter should have increased
        self.assertGreater(self.agent.update_counter, initial_update_counter)


class TestDQNAgentIntegration(unittest.TestCase):
    """Integration tests for DQNAgent with Environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MinesweeperEnvironment("medium")
        self.agent = DQNAgent(
            state_channels=1,
            action_space_size=self.env.action_space_size,
            buffer_size=100,
            batch_size=16
        )
    
    def test_agent_environment_interaction(self):
        """Test interaction between agent and environment."""
        state = self.env.reset()
        
        # Agent selects action
        valid_actions = self.env.get_valid_actions()
        action = self.agent.select_action(state, valid_actions)
        
        # Environment executes action
        next_state, reward, done, info = self.env.step(action)
        next_valid_actions = self.env.get_valid_actions()
        
        # Agent remembers experience
        self.agent.remember(state, action, reward, next_state, done, next_valid_actions)
        
        # Check that experience was stored
        self.assertGreater(len(self.agent.memory), 0)


if __name__ == "__main__":
    unittest.main()


```

```python
"""Tests for DQNAgent and ReplayBuffer."""

import unittest
import numpy as np
import torch
import os
import tempfile
from src.reinforcement_learning.dqn_agent import DQNAgent, ReplayBuffer
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class TestReplayBuffer(unittest.TestCase):
    """Test cases for ReplayBuffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ReplayBuffer(capacity=100)
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
    
    def _random_state(self):
        return np.random.randn(STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
    
    def test_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(len(self.buffer), 0)
    
    def test_push(self):
        """Test pushing experiences."""
        state = self._random_state()
        action = 0
        reward = 1.0
        next_state = self._random_state()
        done = False
        next_valid = np.ones(self.action_space_size, dtype=bool)
        
        self.buffer.push(state, action, reward, next_state, done, next_valid)
        self.assertEqual(len(self.buffer), 1)
    
    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)
        
        for i in range(15):
            state = self._random_state()
            next_valid = np.ones(self.action_space_size, dtype=bool)
            buffer.push(state, 0, 1.0, state, False, next_valid)
        
        # Should not exceed capacity
        self.assertEqual(len(buffer), 10)
    
    def test_sample(self):
        """Test sampling from buffer."""
        # Add some experiences
        for i in range(32):
            state = self._random_state()
            next_valid = np.ones(self.action_space_size, dtype=bool)
            self.buffer.push(state, i % 10, float(i), state, i % 2 == 0, next_valid)
        
        # Sample batch
        batch_size = 16
        states, actions, rewards, next_states, dones, next_valid_actions = self.buffer.sample(batch_size)
        
        # Check types
        self.assertIsInstance(states, torch.Tensor)
        self.assertIsInstance(actions, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)
        self.assertIsInstance(next_valid_actions, torch.Tensor)
        
        # Check shapes
        self.assertEqual(states.shape[0], batch_size)
        self.assertEqual(actions.shape[0], batch_size)
        self.assertEqual(rewards.shape[0], batch_size)
        self.assertEqual(next_states.shape[0], batch_size)
        self.assertEqual(dones.shape[0], batch_size)
        self.assertEqual(next_valid_actions.shape[0], batch_size)


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQNAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
        self.agent = DQNAgent(
            state_channels=STATE_CHANNELS,
            action_space_size=self.action_space_size,
            lr=0.001,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=1000,
            batch_size=32,
            target_update=100
        )
        self.state_shape = (STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH)
    
    def _all_valid_actions(self):
        """Helper to generate a fully valid action mask."""
        return np.ones(self.action_space_size, dtype=bool)
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.action_space_size, self.action_space_size)
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertIsNotNone(self.agent.q_network)
        self.assertIsNotNone(self.agent.target_network)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.memory)
    
    def test_select_action_random(self):
        """Test action selection in random mode (epsilon=1)."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        
        # With epsilon=1, should always be random
        actions = [self.agent.select_action(state) for _ in range(10)]
        
        # Should be valid actions
        for action in actions:
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.action_space_size)
    
    def test_select_action_greedy(self):
        """Test action selection in greedy mode (epsilon=0)."""
        self.agent.epsilon = 0.0
        state = np.random.randn(*self.state_shape).astype(np.float32)
        
        action = self.agent.select_action(state)
        
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_space_size)
    
    def test_select_action_with_valid_mask(self):
        """Test action selection with valid actions mask."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        valid_actions = np.zeros(self.action_space_size, dtype=bool)
        valid_actions[0:10] = True  # Only first 10 actions are valid
        
        self.agent.epsilon = 0.0  # Greedy mode
        action = self.agent.select_action(state, valid_actions)
        
        # Should be one of the valid actions
        self.assertIn(action, range(10))
    
    def test_remember(self):
        """Test storing experiences."""
        state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        action = 5
        reward = 1.0
        next_state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
        done = False
        
        initial_len = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        self.assertEqual(len(self.agent.memory), initial_len + 1)
    
    def test_train_step_insufficient_samples(self):
        """Test training with insufficient samples."""
        # Should return None if not enough samples
        loss = self.agent.train_step()
        self.assertIsNone(loss)
    
    def test_train_step_sufficient_samples(self):
        """Test training with sufficient samples."""
        # Add enough experiences
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            action = i % self.action_space_size
            reward = np.random.randn()
            next_state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            done = i % 10 == 0
            
            self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        
        # Train step
        loss = self.agent.train_step()
        
        # Should return a loss value
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)  # Loss should be non-negative
    
    def test_epsilon_decay(self):
        """Test epsilon decay."""
        initial_epsilon = self.agent.epsilon
        
        # Train multiple times to trigger epsilon decay
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train to trigger epsilon decay
        self.agent.train_step()
        
        # Epsilon should stay the same until decay is triggered explicitly
        self.assertEqual(self.agent.epsilon, initial_epsilon)
        
        # After calling decay, epsilon should decrease
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_end)
    
    def test_save_and_load(self):
        """Test saving and loading agent."""
        # Add some experiences and train
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        self.agent.train_step()
        initial_epsilon = self.agent.epsilon
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            tmp_path = tmp.name
        
        try:
            self.agent.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Create new agent and load
            new_agent = DQNAgent(
                state_channels=1,
                action_space_size=self.action_space_size
            )
            new_agent.load(tmp_path)
            
            # Check that epsilon was loaded
            self.assertEqual(new_agent.epsilon, initial_epsilon)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_target_network_update(self):
        """Test target network update."""
        initial_update_counter = self.agent.update_counter
        
        # Train multiple times to trigger target update
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train enough times to trigger target update
        for _ in range(self.agent.target_update + 1):
            if len(self.agent.memory) >= self.agent.batch_size:
                self.agent.train_step()
        
        # Update counter should have increased
        self.assertGreater(self.agent.update_counter, initial_update_counter)


class TestDQNAgentIntegration(unittest.TestCase):
    """Integration tests for DQNAgent with Environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MinesweeperEnvironment("medium")
        self.agent = DQNAgent(
            state_channels=1,
            action_space_size=self.env.action_space_size,
            buffer_size=100,
            batch_size=16
        )
    
    def test_agent_environment_interaction(self):
        """Test interaction between agent and environment."""
        state = self.env.reset()
        
        # Agent selects action
        valid_actions = self.env.get_valid_actions()
        action = self.agent.select_action(state, valid_actions)
        
        # Environment executes action
        next_state, reward, done, info = self.env.step(action)
        next_valid_actions = self.env.get_valid_actions()
        
        # Agent remembers experience
        self.agent.remember(state, action, reward, next_state, done, next_valid_actions)
        
        # Check that experience was stored
        self.assertGreater(len(self.agent.memory), 0)


if __name__ == "__main__":
    unittest.main()


```

```python
"""Tests for DQNAgent and ReplayBuffer."""

import unittest
import numpy as np
import torch
import os
import tempfile
from src.reinforcement_learning.dqn_agent import DQNAgent, ReplayBuffer
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class TestReplayBuffer(unittest.TestCase):
    """Test cases for ReplayBuffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ReplayBuffer(capacity=100)
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
    
    def _random_state(self):
        return np.random.randn(STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
    
    def test_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(len(self.buffer), 0)
    
    def test_push(self):
        """Test pushing experiences."""
        state = self._random_state()
        action = 0
        reward = 1.0
        next_state = self._random_state()
        done = False
        next_valid = np.ones(self.action_space_size, dtype=bool)
        
        self.buffer.push(state, action, reward, next_state, done, next_valid)
        self.assertEqual(len(self.buffer), 1)
    
    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)
        
        for i in range(15):
            state = self._random_state()
            next_valid = np.ones(self.action_space_size, dtype=bool)
            buffer.push(state, 0, 1.0, state, False, next_valid)
        
        # Should not exceed capacity
        self.assertEqual(len(buffer), 10)
    
    def test_sample(self):
        """Test sampling from buffer."""
        # Add some experiences
        for i in range(32):
            state = self._random_state()
            next_valid = np.ones(self.action_space_size, dtype=bool)
            self.buffer.push(state, i % 10, float(i), state, i % 2 == 0, next_valid)
        
        # Sample batch
        batch_size = 16
        states, actions, rewards, next_states, dones, next_valid_actions = self.buffer.sample(batch_size)
        
        # Check types
        self.assertIsInstance(states, torch.Tensor)
        self.assertIsInstance(actions, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)
        self.assertIsInstance(next_valid_actions, torch.Tensor)
        
        # Check shapes
        self.assertEqual(states.shape[0], batch_size)
        self.assertEqual(actions.shape[0], batch_size)
        self.assertEqual(rewards.shape[0], batch_size)
        self.assertEqual(next_states.shape[0], batch_size)
        self.assertEqual(dones.shape[0], batch_size)
        self.assertEqual(next_valid_actions.shape[0], batch_size)


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQNAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
        self.agent = DQNAgent(
            state_channels=STATE_CHANNELS,
            action_space_size=self.action_space_size,
            lr=0.001,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=1000,
            batch_size=32,
            target_update=100
        )
        self.state_shape = (STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH)
    
    def _all_valid_actions(self):
        """Helper to generate a fully valid action mask."""
        return np.ones(self.action_space_size, dtype=bool)
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.action_space_size, self.action_space_size)
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertIsNotNone(self.agent.q_network)
        self.assertIsNotNone(self.agent.target_network)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.memory)
    
    def test_select_action_random(self):
        """Test action selection in random mode (epsilon=1)."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        
        # With epsilon=1, should always be random
        actions = [self.agent.select_action(state) for _ in range(10)]
        
        # Should be valid actions
        for action in actions:
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.action_space_size)
    
    def test_select_action_greedy(self):
        """Test action selection in greedy mode (epsilon=0)."""
        self.agent.epsilon = 0.0
        state = np.random.randn(*self.state_shape).astype(np.float32)
        
        action = self.agent.select_action(state)
        
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_space_size)
    
    def test_select_action_with_valid_mask(self):
        """Test action selection with valid actions mask."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        valid_actions = np.zeros(self.action_space_size, dtype=bool)
        valid_actions[0:10] = True  # Only first 10 actions are valid
        
        self.agent.epsilon = 0.0  # Greedy mode
        action = self.agent.select_action(state, valid_actions)
        
        # Should be one of the valid actions
        self.assertIn(action, range(10))
    
    def test_remember(self):
        """Test storing experiences."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        action = 5
        reward = 1.0
        next_state = np.random.randn(*self.state_shape).astype(np.float32)
        done = False
        
        initial_len = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        self.assertEqual(len(self.agent.memory), initial_len + 1)
    
    def test_train_step_insufficient_samples(self):
        """Test training with insufficient samples."""
        # Should return None if not enough samples
        loss = self.agent.train_step()
        self.assertIsNone(loss)
    
    def test_train_step_sufficient_samples(self):
        """Test training with sufficient samples."""
        # Add enough experiences
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            action = i % self.action_space_size
            reward = np.random.randn()
            next_state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            next_state = np.random.randn(*self.state_shape).astype(np.float32)
            done = i % 10 == 0
            
            self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        
        # Train step
        loss = self.agent.train_step()
        
        # Should return a loss value
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)  # Loss should be non-negative
    
    def test_epsilon_decay(self):
        """Test epsilon decay."""
        initial_epsilon = self.agent.epsilon
        
        # Train multiple times to trigger epsilon decay
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train to trigger epsilon decay
        self.agent.train_step()
        
        # Epsilon should stay the same until decay is triggered explicitly
        self.assertEqual(self.agent.epsilon, initial_epsilon)
        
        # After calling decay, epsilon should decrease
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_end)
    
    def test_save_and_load(self):
        """Test saving and loading agent."""
        # Add some experiences and train
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        self.agent.train_step()
        initial_epsilon = self.agent.epsilon
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            tmp_path = tmp.name
        
        try:
            self.agent.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Create new agent and load
            new_agent = DQNAgent(
                state_channels=1,
                action_space_size=self.action_space_size
            )
            new_agent.load(tmp_path)
            
            # Check that epsilon was loaded
            self.assertEqual(new_agent.epsilon, initial_epsilon)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_target_network_update(self):
        """Test target network update."""
        initial_update_counter = self.agent.update_counter
        
        # Train multiple times to trigger target update
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train enough times to trigger target update
        for _ in range(self.agent.target_update + 1):
            if len(self.agent.memory) >= self.agent.batch_size:
                self.agent.train_step()
        
        # Update counter should have increased
        self.assertGreater(self.agent.update_counter, initial_update_counter)


class TestDQNAgentIntegration(unittest.TestCase):
    """Integration tests for DQNAgent with Environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MinesweeperEnvironment("medium")
        self.agent = DQNAgent(
            state_channels=1,
            action_space_size=self.env.action_space_size,
            buffer_size=100,
            batch_size=16
        )
    
    def test_agent_environment_interaction(self):
        """Test interaction between agent and environment."""
        state = self.env.reset()
        
        # Agent selects action
        valid_actions = self.env.get_valid_actions()
        action = self.agent.select_action(state, valid_actions)
        
        # Environment executes action
        next_state, reward, done, info = self.env.step(action)
        next_valid_actions = self.env.get_valid_actions()
        
        # Agent remembers experience
        self.agent.remember(state, action, reward, next_state, done, next_valid_actions)
        
        # Check that experience was stored
        self.assertGreater(len(self.agent.memory), 0)


if __name__ == "__main__":
    unittest.main()


```

```python
*** Begin Patch
*** Update File: tests/reinforcement_learning/test_dqn_agent.py
@@
         for i in range(self.agent.batch_size + 10):
-            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
-            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
+            state = np.random.randn(*self.state_shape).astype(np.float32)
+            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
@@
         for i in range(self.agent.batch_size + 10):
-            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
-            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
+            state = np.random.randn(*self.state_shape).astype(np.float32)
+            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
@@
             new_agent = DQNAgent(
-                state_channels=1,
+                state_channels=STATE_CHANNELS,
                 action_space_size=self.action_space_size
             )
*** End Patch
```

```python
"""Tests for DQNAgent and ReplayBuffer."""

import unittest
import numpy as np
import torch
import os
import tempfile
from src.reinforcement_learning.dqn_agent import DQNAgent, ReplayBuffer
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class TestReplayBuffer(unittest.TestCase):
    """Test cases for ReplayBuffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ReplayBuffer(capacity=100)
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
    
    def _random_state(self):
        return np.random.randn(STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
    
    def test_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(len(self.buffer), 0)
    
    def test_push(self):
        """Test pushing experiences."""
        state = self._random_state()
        action = 0
        reward = 1.0
        next_state = self._random_state()
        done = False
        next_valid = np.ones(self.action_space_size, dtype=bool)
        
        self.buffer.push(state, action, reward, next_state, done, next_valid)
        self.assertEqual(len(self.buffer), 1)
    
    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)
        
        for i in range(15):
            state = self._random_state()
            next_valid = np.ones(self.action_space_size, dtype=bool)
            buffer.push(state, 0, 1.0, state, False, next_valid)
        
        # Should not exceed capacity
        self.assertEqual(len(buffer), 10)
    
    def test_sample(self):
        """Test sampling from buffer."""
        # Add some experiences
        for i in range(32):
            state = self._random_state()
            next_valid = np.ones(self.action_space_size, dtype=bool)
            self.buffer.push(state, i % 10, float(i), state, i % 2 == 0, next_valid)
        
        # Sample batch
        batch_size = 16
        states, actions, rewards, next_states, dones, next_valid_actions = self.buffer.sample(batch_size)
        
        # Check types
        self.assertIsInstance(states, torch.Tensor)
        self.assertIsInstance(actions, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)
        self.assertIsInstance(next_valid_actions, torch.Tensor)
        
        # Check shapes
        self.assertEqual(states.shape[0], batch_size)
        self.assertEqual(actions.shape[0], batch_size)
        self.assertEqual(rewards.shape[0], batch_size)
        self.assertEqual(next_states.shape[0], batch_size)
        self.assertEqual(dones.shape[0], batch_size)
        self.assertEqual(next_valid_actions.shape[0], batch_size)


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQNAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
        self.agent = DQNAgent(
            state_channels=STATE_CHANNELS,
            action_space_size=self.action_space_size,
            lr=0.001,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=1000,
            batch_size=32,
            target_update=100
        )
        self.state_shape = (STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH)
    
    def _all_valid_actions(self):
        """Helper to generate a fully valid action mask."""
        return np.ones(self.action_space_size, dtype=bool)
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.action_space_size, self.action_space_size)
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertIsNotNone(self.agent.q_network)
        self.assertIsNotNone(self.agent.target_network)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.memory)
    
    def test_select_action_random(self):
        """Test action selection in random mode (epsilon=1)."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        
        # With epsilon=1, should always be random
        actions = [self.agent.select_action(state) for _ in range(10)]
        
        # Should be valid actions
        for action in actions:
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.action_space_size)
    
    def test_select_action_greedy(self):
        """Test action selection in greedy mode (epsilon=0)."""
        self.agent.epsilon = 0.0
        state = np.random.randn(*self.state_shape).astype(np.float32)
        
        action = self.agent.select_action(state)
        
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_space_size)
    
    def test_select_action_with_valid_mask(self):
        """Test action selection with valid actions mask."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        valid_actions = np.zeros(self.action_space_size, dtype=bool)
        valid_actions[0:10] = True  # Only first 10 actions are valid
        
        self.agent.epsilon = 0.0  # Greedy mode
        action = self.agent.select_action(state, valid_actions)
        
        # Should be one of the valid actions
        self.assertIn(action, range(10))
    
    def test_remember(self):
        """Test storing experiences."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        action = 5
        reward = 1.0
        next_state = np.random.randn(*self.state_shape).astype(np.float32)
        done = False
        
        initial_len = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        self.assertEqual(len(self.agent.memory), initial_len + 1)
    
    def test_train_step_insufficient_samples(self):
        """Test training with insufficient samples."""
        # Should return None if not enough samples
        loss = self.agent.train_step()
        self.assertIsNone(loss)
    
    def test_train_step_sufficient_samples(self):
        """Test training with sufficient samples."""
        # Add enough experiences
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            action = i % self.action_space_size
            reward = np.random.randn()
            next_state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            next_state = np.random.randn(*self.state_shape).astype(np.float32)
            done = i % 10 == 0
            
            self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        
        # Train step
        loss = self.agent.train_step()
        
        # Should return a loss value
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)  # Loss should be non-negative
    
    def test_epsilon_decay(self):
        """Test epsilon decay."""
        initial_epsilon = self.agent.epsilon
        
        # Train multiple times to trigger epsilon decay
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train to trigger epsilon decay
        self.agent.train_step()
        
        # Epsilon should stay the same until decay is triggered explicitly
        self.assertEqual(self.agent.epsilon, initial_epsilon)
        
        # After calling decay, epsilon should decrease
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_end)
    
    def test_save_and_load(self):
        """Test saving and loading agent."""
        # Add some experiences and train
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        self.agent.train_step()
        initial_epsilon = self.agent.epsilon
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            tmp_path = tmp.name
        
        try:
            self.agent.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Create new agent and load
            new_agent = DQNAgent(
                state_channels=1,
                action_space_size=self.action_space_size
            )
            new_agent.load(tmp_path)
            
            # Check that epsilon was loaded
            self.assertEqual(new_agent.epsilon, initial_epsilon)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_target_network_update(self):
        """Test target network update."""
        initial_update_counter = self.agent.update_counter
        
        # Train multiple times to trigger target update
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train enough times to trigger target update
        for _ in range(self.agent.target_update + 1):
            if len(self.agent.memory) >= self.agent.batch_size:
                self.agent.train_step()
        
        # Update counter should have increased
        self.assertGreater(self.agent.update_counter, initial_update_counter)


class TestDQNAgentIntegration(unittest.TestCase):
    """Integration tests for DQNAgent with Environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MinesweeperEnvironment("medium")
        self.agent = DQNAgent(
            state_channels=1,
            action_space_size=self.env.action_space_size,
            buffer_size=100,
            batch_size=16
        )
    
    def test_agent_environment_interaction(self):
        """Test interaction between agent and environment."""
        state = self.env.reset()
        
        # Agent selects action
        valid_actions = self.env.get_valid_actions()
        action = self.agent.select_action(state, valid_actions)
        
        # Environment executes action
        next_state, reward, done, info = self.env.step(action)
        next_valid_actions = self.env.get_valid_actions()
        
        # Agent remembers experience
        self.agent.remember(state, action, reward, next_state, done, next_valid_actions)
        
        # Check that experience was stored
        self.assertGreater(len(self.agent.memory), 0)


if __name__ == "__main__":
    unittest.main()


```

```python
"""Tests for DQNAgent and ReplayBuffer."""

import unittest
import numpy as np
import torch
import os
import tempfile
from src.reinforcement_learning.dqn_agent import DQNAgent, ReplayBuffer
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class TestReplayBuffer(unittest.TestCase):
    """Test cases for ReplayBuffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ReplayBuffer(capacity=100)
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
    
    def _random_state(self):
        return np.random.randn(STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
    
    def test_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(len(self.buffer), 0)
    
    def test_push(self):
        """Test pushing experiences."""
        state = self._random_state()
        action = 0
        reward = 1.0
        next_state = self._random_state()
        done = False
        next_valid = np.ones(self.action_space_size, dtype=bool)
        
        self.buffer.push(state, action, reward, next_state, done, next_valid)
        self.assertEqual(len(self.buffer), 1)
    
    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)
        
        for i in range(15):
            state = self._random_state()
            next_valid = np.ones(self.action_space_size, dtype=bool)
            buffer.push(state, 0, 1.0, state, False, next_valid)
        
        # Should not exceed capacity
        self.assertEqual(len(buffer), 10)
    
    def test_sample(self):
        """Test sampling from buffer."""
        # Add some experiences
        for i in range(32):
            state = self._random_state()
            next_valid = np.ones(self.action_space_size, dtype=bool)
            self.buffer.push(state, i % 10, float(i), state, i % 2 == 0, next_valid)
        
        # Sample batch
        batch_size = 16
        states, actions, rewards, next_states, dones, next_valid_actions = self.buffer.sample(batch_size)
        
        # Check types
        self.assertIsInstance(states, torch.Tensor)
        self.assertIsInstance(actions, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)
        self.assertIsInstance(next_valid_actions, torch.Tensor)
        
        # Check shapes
        self.assertEqual(states.shape[0], batch_size)
        self.assertEqual(actions.shape[0], batch_size)
        self.assertEqual(rewards.shape[0], batch_size)
        self.assertEqual(next_states.shape[0], batch_size)
        self.assertEqual(dones.shape[0], batch_size)
        self.assertEqual(next_valid_actions.shape[0], batch_size)


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQNAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
        self.agent = DQNAgent(
            state_channels=STATE_CHANNELS,
            action_space_size=self.action_space_size,
            lr=0.001,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=1000,
            batch_size=32,
            target_update=100
        )
        self.state_shape = (STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH)
    
    def _all_valid_actions(self):
        """Helper to generate a fully valid action mask."""
        return np.ones(self.action_space_size, dtype=bool)
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.action_space_size, self.action_space_size)
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertIsNotNone(self.agent.q_network)
        self.assertIsNotNone(self.agent.target_network)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.memory)
    
    def test_select_action_random(self):
        """Test action selection in random mode (epsilon=1)."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        
        # With epsilon=1, should always be random
        actions = [self.agent.select_action(state) for _ in range(10)]
        
        # Should be valid actions
        for action in actions:
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.action_space_size)
    
    def test_select_action_greedy(self):
        """Test action selection in greedy mode (epsilon=0)."""
        self.agent.epsilon = 0.0
        state = np.random.randn(*self.state_shape).astype(np.float32)
        
        action = self.agent.select_action(state)
        
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_space_size)
    
    def test_select_action_with_valid_mask(self):
        """Test action selection with valid actions mask."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        valid_actions = np.zeros(self.action_space_size, dtype=bool)
        valid_actions[0:10] = True  # Only first 10 actions are valid
        
        self.agent.epsilon = 0.0  # Greedy mode
        action = self.agent.select_action(state, valid_actions)
        
        # Should be one of the valid actions
        self.assertIn(action, range(10))
    
    def test_remember(self):
        """Test storing experiences."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        action = 5
        reward = 1.0
        next_state = np.random.randn(*self.state_shape).astype(np.float32)
        done = False
        
        initial_len = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        self.assertEqual(len(self.agent.memory), initial_len + 1)
    
    def test_train_step_insufficient_samples(self):
        """Test training with insufficient samples."""
        # Should return None if not enough samples
        loss = self.agent.train_step()
        self.assertIsNone(loss)
    
    def test_train_step_sufficient_samples(self):
        """Test training with sufficient samples."""
        # Add enough experiences
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            action = i % self.action_space_size
            reward = np.random.randn()
            next_state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            next_state = np.random.randn(*self.state_shape).astype(np.float32)
            done = i % 10 == 0
            
            self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        
        # Train step
        loss = self.agent.train_step()
        
        # Should return a loss value
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)  # Loss should be non-negative
    
    def test_epsilon_decay(self):
        """Test epsilon decay."""
        initial_epsilon = self.agent.epsilon
        
        # Train multiple times to trigger epsilon decay
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train to trigger epsilon decay
        self.agent.train_step()
        
        # Epsilon should stay the same until decay is triggered explicitly
        self.assertEqual(self.agent.epsilon, initial_epsilon)
        
        # After calling decay, epsilon should decrease
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_end)
    
    def test_save_and_load(self):
        """Test saving and loading agent."""
        # Add some experiences and train
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        self.agent.train_step()
        initial_epsilon = self.agent.epsilon
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            tmp_path = tmp.name
        
        try:
            self.agent.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Create new agent and load
            new_agent = DQNAgent(
                state_channels=STATE_CHANNELS,
                action_space_size=self.action_space_size
            )
            new_agent.load(tmp_path)
            
            # Check that epsilon was loaded
            self.assertEqual(new_agent.epsilon, initial_epsilon)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_target_network_update(self):
        """Test target network update."""
        initial_update_counter = self.agent.update_counter
        
        # Train multiple times to trigger target update
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train enough times to trigger target update
        for _ in range(self.agent.target_update + 1):
            if len(self.agent.memory) >= self.agent.batch_size:
                self.agent.train_step()
        
        # Update counter should have increased
        self.assertGreater(self.agent.update_counter, initial_update_counter)


class TestDQNAgentIntegration(unittest.TestCase):
    """Integration tests for DQNAgent with Environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MinesweeperEnvironment("medium")
        self.agent = DQNAgent(
            state_channels=1,
            action_space_size=self.env.action_space_size,
            buffer_size=100,
            batch_size=16
        )
    
    def test_agent_environment_interaction(self):
        """Test interaction between agent and environment."""
        state = self.env.reset()
        
        # Agent selects action
        valid_actions = self.env.get_valid_actions()
        action = self.agent.select_action(state, valid_actions)
        
        # Environment executes action
        next_state, reward, done, info = self.env.step(action)
        next_valid_actions = self.env.get_valid_actions()
        
        # Agent remembers experience
        self.agent.remember(state, action, reward, next_state, done, next_valid_actions)
        
        # Check that experience was stored
        self.assertGreater(len(self.agent.memory), 0)


if __name__ == "__main__":
    unittest.main()


```

```python
"""Tests for DQNAgent and ReplayBuffer."""

import unittest
import numpy as np
import torch
import os
import tempfile
from src.reinforcement_learning.dqn_agent import DQNAgent, ReplayBuffer
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class TestReplayBuffer(unittest.TestCase):
    """Test cases for ReplayBuffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ReplayBuffer(capacity=100)
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
    
    def _random_state(self):
        return np.random.randn(STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
    
    def test_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(len(self.buffer), 0)
    
    def test_push(self):
        """Test pushing experiences."""
        state = self._random_state()
        action = 0
        reward = 1.0
        next_state = self._random_state()
        done = False
        next_valid = np.ones(self.action_space_size, dtype=bool)
        
        self.buffer.push(state, action, reward, next_state, done, next_valid)
        self.assertEqual(len(self.buffer), 1)
    
    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)
        
        for i in range(15):
            state = self._random_state()
            next_valid = np.ones(self.action_space_size, dtype=bool)
            buffer.push(state, 0, 1.0, state, False, next_valid)
        
        # Should not exceed capacity
        self.assertEqual(len(buffer), 10)
    
    def test_sample(self):
        """Test sampling from buffer."""
        # Add some experiences
        for i in range(32):
            state = self._random_state()
            next_valid = np.ones(self.action_space_size, dtype=bool)
            self.buffer.push(state, i % 10, float(i), state, i % 2 == 0, next_valid)
        
        # Sample batch
        batch_size = 16
        states, actions, rewards, next_states, dones, next_valid_actions = self.buffer.sample(batch_size)
        
        # Check types
        self.assertIsInstance(states, torch.Tensor)
        self.assertIsInstance(actions, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)
        self.assertIsInstance(next_valid_actions, torch.Tensor)
        
        # Check shapes
        self.assertEqual(states.shape[0], batch_size)
        self.assertEqual(actions.shape[0], batch_size)
        self.assertEqual(rewards.shape[0], batch_size)
        self.assertEqual(next_states.shape[0], batch_size)
        self.assertEqual(dones.shape[0], batch_size)
        self.assertEqual(next_valid_actions.shape[0], batch_size)


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQNAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
        self.agent = DQNAgent(
            state_channels=STATE_CHANNELS,
            action_space_size=self.action_space_size,
            lr=0.001,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=1000,
            batch_size=32,
            target_update=100
        )
        self.state_shape = (STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH)
    
    def _all_valid_actions(self):
        """Helper to generate a fully valid action mask."""
        return np.ones(self.action_space_size, dtype=bool)
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.action_space_size, self.action_space_size)
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertIsNotNone(self.agent.q_network)
        self.assertIsNotNone(self.agent.target_network)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.memory)
    
    def test_select_action_random(self):
        """Test action selection in random mode (epsilon=1)."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        
        # With epsilon=1, should always be random
        actions = [self.agent.select_action(state) for _ in range(10)]
        
        # Should be valid actions
        for action in actions:
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.action_space_size)
    
    def test_select_action_greedy(self):
        """Test action selection in greedy mode (epsilon=0)."""
        self.agent.epsilon = 0.0
        state = np.random.randn(*self.state_shape).astype(np.float32)
        
        action = self.agent.select_action(state)
        
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_space_size)
    
    def test_select_action_with_valid_mask(self):
        """Test action selection with valid actions mask."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        valid_actions = np.zeros(self.action_space_size, dtype=bool)
        valid_actions[0:10] = True  # Only first 10 actions are valid
        
        self.agent.epsilon = 0.0  # Greedy mode
        action = self.agent.select_action(state, valid_actions)
        
        # Should be one of the valid actions
        self.assertIn(action, range(10))
    
    def test_remember(self):
        """Test storing experiences."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        action = 5
        reward = 1.0
        next_state = np.random.randn(*self.state_shape).astype(np.float32)
        done = False
        
        initial_len = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        self.assertEqual(len(self.agent.memory), initial_len + 1)
    
    def test_train_step_insufficient_samples(self):
        """Test training with insufficient samples."""
        # Should return None if not enough samples
        loss = self.agent.train_step()
        self.assertIsNone(loss)
    
    def test_train_step_sufficient_samples(self):
        """Test training with sufficient samples."""
        # Add enough experiences
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            action = i % self.action_space_size
            reward = np.random.randn()
            next_state = np.random.randn(1, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
            next_state = np.random.randn(*self.state_shape).astype(np.float32)
            done = i % 10 == 0
            
            self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        
        # Train step
        loss = self.agent.train_step()
        
        # Should return a loss value
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)  # Loss should be non-negative
    
    def test_epsilon_decay(self):
        """Test epsilon decay."""
        initial_epsilon = self.agent.epsilon
        
        # Train multiple times to trigger epsilon decay
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train to trigger epsilon decay
        self.agent.train_step()
        
        # Epsilon should stay the same until decay is triggered explicitly
        self.assertEqual(self.agent.epsilon, initial_epsilon)
        
        # After calling decay, epsilon should decrease
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_end)
    
    def test_save_and_load(self):
        """Test saving and loading agent."""
        # Add some experiences and train
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        self.agent.train_step()
        initial_epsilon = self.agent.epsilon
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            tmp_path = tmp.name
        
        try:
            self.agent.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Create new agent and load
            new_agent = DQNAgent(
                state_channels=STATE_CHANNELS,
                action_space_size=self.action_space_size
            )
            new_agent.load(tmp_path)
            
            # Check that epsilon was loaded
            self.assertEqual(new_agent.epsilon, initial_epsilon)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_target_network_update(self):
        """Test target network update."""
        initial_update_counter = self.agent.update_counter
        
        # Train multiple times to trigger target update
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train enough times to trigger target update
        for _ in range(self.agent.target_update + 1):
            if len(self.agent.memory) >= self.agent.batch_size:
                self.agent.train_step()
        
        # Update counter should have increased
        self.assertGreater(self.agent.update_counter, initial_update_counter)


class TestDQNAgentIntegration(unittest.TestCase):
    """Integration tests for DQNAgent with Environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MinesweeperEnvironment("medium")
        self.agent = DQNAgent(
            state_channels=1,
            action_space_size=self.env.action_space_size,
            buffer_size=100,
            batch_size=16
        )
    
    def test_agent_environment_interaction(self):
        """Test interaction between agent and environment."""
        state = self.env.reset()
        
        # Agent selects action
        valid_actions = self.env.get_valid_actions()
        action = self.agent.select_action(state, valid_actions)
        
        # Environment executes action
        next_state, reward, done, info = self.env.step(action)
        next_valid_actions = self.env.get_valid_actions()
        
        # Agent remembers experience
        self.agent.remember(state, action, reward, next_state, done, next_valid_actions)
        
        # Check that experience was stored
        self.assertGreater(len(self.agent.memory), 0)


if __name__ == "__main__":
    unittest.main()


```

```python
"""Tests for DQNAgent and ReplayBuffer."""

import unittest
import numpy as np
import torch
import os
import tempfile
from src.reinforcement_learning.dqn_agent import DQNAgent, ReplayBuffer
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class TestReplayBuffer(unittest.TestCase):
    """Test cases for ReplayBuffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ReplayBuffer(capacity=100)
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
    
    def _random_state(self):
        return np.random.randn(STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
    
    def test_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(len(self.buffer), 0)
    
    def test_push(self):
        """Test pushing experiences."""
        state = self._random_state()
        action = 0
        reward = 1.0
        next_state = self._random_state()
        done = False
        next_valid = np.ones(self.action_space_size, dtype=bool)
        
        self.buffer.push(state, action, reward, next_state, done, next_valid)
        self.assertEqual(len(self.buffer), 1)
    
    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)
        
        for i in range(15):
            state = self._random_state()
            next_valid = np.ones(self.action_space_size, dtype=bool)
            buffer.push(state, 0, 1.0, state, False, next_valid)
        
        # Should not exceed capacity
        self.assertEqual(len(buffer), 10)
    
    def test_sample(self):
        """Test sampling from buffer."""
        # Add some experiences
        for i in range(32):
            state = self._random_state()
            next_valid = np.ones(self.action_space_size, dtype=bool)
            self.buffer.push(state, i % 10, float(i), state, i % 2 == 0, next_valid)
        
        # Sample batch
        batch_size = 16
        states, actions, rewards, next_states, dones, next_valid_actions = self.buffer.sample(batch_size)
        
        # Check types
        self.assertIsInstance(states, torch.Tensor)
        self.assertIsInstance(actions, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)
        self.assertIsInstance(next_valid_actions, torch.Tensor)
        
        # Check shapes
        self.assertEqual(states.shape[0], batch_size)
        self.assertEqual(actions.shape[0], batch_size)
        self.assertEqual(rewards.shape[0], batch_size)
        self.assertEqual(next_states.shape[0], batch_size)
        self.assertEqual(dones.shape[0], batch_size)
        self.assertEqual(next_valid_actions.shape[0], batch_size)


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQNAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
        self.agent = DQNAgent(
            state_channels=STATE_CHANNELS,
            action_space_size=self.action_space_size,
            lr=0.001,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=1000,
            batch_size=32,
            target_update=100
        )
        self.state_shape = (STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH)
    
    def _all_valid_actions(self):
        """Helper to generate a fully valid action mask."""
        return np.ones(self.action_space_size, dtype=bool)
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.action_space_size, self.action_space_size)
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertIsNotNone(self.agent.q_network)
        self.assertIsNotNone(self.agent.target_network)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.memory)
    
    def test_select_action_random(self):
        """Test action selection in random mode (epsilon=1)."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        
        # With epsilon=1, should always be random
        actions = [self.agent.select_action(state) for _ in range(10)]
        
        # Should be valid actions
        for action in actions:
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.action_space_size)
    
    def test_select_action_greedy(self):
        """Test action selection in greedy mode (epsilon=0)."""
        self.agent.epsilon = 0.0
        state = np.random.randn(*self.state_shape).astype(np.float32)
        
        action = self.agent.select_action(state)
        
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_space_size)
    
    def test_select_action_with_valid_mask(self):
        """Test action selection with valid actions mask."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        valid_actions = np.zeros(self.action_space_size, dtype=bool)
        valid_actions[0:10] = True  # Only first 10 actions are valid
        
        self.agent.epsilon = 0.0  # Greedy mode
        action = self.agent.select_action(state, valid_actions)
        
        # Should be one of the valid actions
        self.assertIn(action, range(10))
    
    def test_remember(self):
        """Test storing experiences."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        action = 5
        reward = 1.0
        next_state = np.random.randn(*self.state_shape).astype(np.float32)
        done = False
        
        initial_len = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        self.assertEqual(len(self.agent.memory), initial_len + 1)
    
    def test_train_step_insufficient_samples(self):
        """Test training with insufficient samples."""
        # Should return None if not enough samples
        loss = self.agent.train_step()
        self.assertIsNone(loss)
    
    def test_train_step_sufficient_samples(self):
        """Test training with sufficient samples."""
        # Add enough experiences
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            action = i % self.action_space_size
            reward = np.random.randn()
            next_state = np.random.randn(*self.state_shape).astype(np.float32)
            done = i % 10 == 0
            
            self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        
        # Train step
        loss = self.agent.train_step()
        
        # Should return a loss value
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)  # Loss should be non-negative
    
    def test_epsilon_decay(self):
        """Test epsilon decay."""
        initial_epsilon = self.agent.epsilon
        
        # Train multiple times to trigger epsilon decay
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train to trigger epsilon decay
        self.agent.train_step()
        
        # Epsilon should stay the same until decay is triggered explicitly
        self.assertEqual(self.agent.epsilon, initial_epsilon)
        
        # After calling decay, epsilon should decrease
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_end)
    
    def test_save_and_load(self):
        """Test saving and loading agent."""
        # Add some experiences and train
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        self.agent.train_step()
        initial_epsilon = self.agent.epsilon
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            tmp_path = tmp.name
        
        try:
            self.agent.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Create new agent and load
            new_agent = DQNAgent(
                state_channels=STATE_CHANNELS,
                action_space_size=self.action_space_size
            )
            new_agent.load(tmp_path)
            
            # Check that epsilon was loaded
            self.assertEqual(new_agent.epsilon, initial_epsilon)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_target_network_update(self):
        """Test target network update."""
        initial_update_counter = self.agent.update_counter
        
        # Train multiple times to trigger target update
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train enough times to trigger target update
        for _ in range(self.agent.target_update + 1):
            if len(self.agent.memory) >= self.agent.batch_size:
                self.agent.train_step()
        
        # Update counter should have increased
        self.assertGreater(self.agent.update_counter, initial_update_counter)


class TestDQNAgentIntegration(unittest.TestCase):
    """Integration tests for DQNAgent with Environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MinesweeperEnvironment("medium")
        self.agent = DQNAgent(
            state_channels=1,
            action_space_size=self.env.action_space_size,
            buffer_size=100,
            batch_size=16
        )
    
    def test_agent_environment_interaction(self):
        """Test interaction between agent and environment."""
        state = self.env.reset()
        
        # Agent selects action
        valid_actions = self.env.get_valid_actions()
        action = self.agent.select_action(state, valid_actions)
        
        # Environment executes action
        next_state, reward, done, info = self.env.step(action)
        next_valid_actions = self.env.get_valid_actions()
        
        # Agent remembers experience
        self.agent.remember(state, action, reward, next_state, done, next_valid_actions)
        
        # Check that experience was stored
        self.assertGreater(len(self.agent.memory), 0)


if __name__ == "__main__":
    unittest.main()


```

```python
"""Tests for DQNAgent and ReplayBuffer."""

import unittest
import numpy as np
import torch
import os
import tempfile
from src.reinforcement_learning.dqn_agent import DQNAgent, ReplayBuffer
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class TestReplayBuffer(unittest.TestCase):
    """Test cases for ReplayBuffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ReplayBuffer(capacity=100)
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
    
    def _random_state(self):
        return np.random.randn(STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
    
    def test_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(len(self.buffer), 0)
    
    def test_push(self):
        """Test pushing experiences."""
        state = self._random_state()
        action = 0
        reward = 1.0
        next_state = self._random_state()
        done = False
        next_valid = np.ones(self.action_space_size, dtype=bool)
        
        self.buffer.push(state, action, reward, next_state, done, next_valid)
        self.assertEqual(len(self.buffer), 1)
    
    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)
        
        for i in range(15):
            state = self._random_state()
            next_valid = np.ones(self.action_space_size, dtype=bool)
            buffer.push(state, 0, 1.0, state, False, next_valid)
        
        # Should not exceed capacity
        self.assertEqual(len(buffer), 10)
    
    def test_sample(self):
        """Test sampling from buffer."""
        # Add some experiences
        for i in range(32):
            state = self._random_state()
            next_valid = np.ones(self.action_space_size, dtype=bool)
            self.buffer.push(state, i % 10, float(i), state, i % 2 == 0, next_valid)
        
        # Sample batch
        batch_size = 16
        states, actions, rewards, next_states, dones, next_valid_actions = self.buffer.sample(batch_size)
        
        # Check types
        self.assertIsInstance(states, torch.Tensor)
        self.assertIsInstance(actions, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)
        self.assertIsInstance(next_valid_actions, torch.Tensor)
        
        # Check shapes
        self.assertEqual(states.shape[0], batch_size)
        self.assertEqual(actions.shape[0], batch_size)
        self.assertEqual(rewards.shape[0], batch_size)
        self.assertEqual(next_states.shape[0], batch_size)
        self.assertEqual(dones.shape[0], batch_size)
        self.assertEqual(next_valid_actions.shape[0], batch_size)


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQNAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT
        self.agent = DQNAgent(
            state_channels=STATE_CHANNELS,
            action_space_size=self.action_space_size,
            lr=0.001,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=1000,
            batch_size=32,
            target_update=100
        )
        self.state_shape = (STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH)
    
    def _all_valid_actions(self):
        """Helper to generate a fully valid action mask."""
        return np.ones(self.action_space_size, dtype=bool)
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.action_space_size, self.action_space_size)
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertIsNotNone(self.agent.q_network)
        self.assertIsNotNone(self.agent.target_network)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.memory)
    
    def test_select_action_random(self):
        """Test action selection in random mode (epsilon=1)."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        
        # With epsilon=1, should always be random
        actions = [self.agent.select_action(state) for _ in range(10)]
        
        # Should be valid actions
        for action in actions:
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.action_space_size)
    
    def test_select_action_greedy(self):
        """Test action selection in greedy mode (epsilon=0)."""
        self.agent.epsilon = 0.0
        state = np.random.randn(*self.state_shape).astype(np.float32)
        
        action = self.agent.select_action(state)
        
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_space_size)
    
    def test_select_action_with_valid_mask(self):
        """Test action selection with valid actions mask."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        valid_actions = np.zeros(self.action_space_size, dtype=bool)
        valid_actions[0:10] = True  # Only first 10 actions are valid
        
        self.agent.epsilon = 0.0  # Greedy mode
        action = self.agent.select_action(state, valid_actions)
        
        # Should be one of the valid actions
        self.assertIn(action, range(10))
    
    def test_remember(self):
        """Test storing experiences."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        action = 5
        reward = 1.0
        next_state = np.random.randn(*self.state_shape).astype(np.float32)
        done = False
        
        initial_len = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        self.assertEqual(len(self.agent.memory), initial_len + 1)
    
    def test_train_step_insufficient_samples(self):
        """Test training with insufficient samples."""
        # Should return None if not enough samples
        loss = self.agent.train_step()
        self.assertIsNone(loss)
    
    def test_train_step_sufficient_samples(self):
        """Test training with sufficient samples."""
        # Add enough experiences
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            action = i % self.action_space_size
            reward = np.random.randn()
            next_state = np.random.randn(*self.state_shape).astype(np.float32)
            done = i % 10 == 0
            
            self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        
        # Train step
        loss = self.agent.train_step()
        
        # Should return a loss value
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)  # Loss should be non-negative
    
    def test_epsilon_decay(self):
        """Test epsilon decay."""
        initial_epsilon = self.agent.epsilon
        
        # Train multiple times to trigger epsilon decay
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train to trigger epsilon decay
        self.agent.train_step()
        
        # Epsilon should stay the same until decay is triggered explicitly
        self.assertEqual(self.agent.epsilon, initial_epsilon)
        
        # After calling decay, epsilon should decrease
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_end)
    
    def test_save_and_load(self):
        """Test saving and loading agent."""
        # Add some experiences and train
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        self.agent.train_step()
        initial_epsilon = self.agent.epsilon
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            tmp_path = tmp.name
        
        try:
            self.agent.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Create new agent and load
            new_agent = DQNAgent(
                state_channels=STATE_CHANNELS,
                action_space_size=self.action_space_size
            )
            new_agent.load(tmp_path)
            
            # Check that epsilon was loaded
            self.assertEqual(new_agent.epsilon, initial_epsilon)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_target_network_update(self):
        """Test target network update."""
        initial_update_counter = self.agent.update_counter
        
        # Train multiple times to trigger target update
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train enough times to trigger target update
        for _ in range(self.agent.target_update + 1):
            if len(self.agent.memory) >= self.agent.batch_size:
                self.agent.train_step()
        
        # Update counter should have increased
        self.assertGreater(self.agent.update_counter, initial_update_counter)


class TestDQNAgentIntegration(unittest.TestCase):
    """Integration tests for DQNAgent with Environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MinesweeperEnvironment("medium")
        self.agent = DQNAgent(
            state_channels=STATE_CHANNELS,
            action_space_size=self.env.action_space_size,
            buffer_size=100,
            batch_size=16
        )
    
    def test_agent_environment_interaction(self):
        """Test interaction between agent and environment."""
        state = self.env.reset()
        
        # Agent selects action
        valid_actions = self.env.get_valid_actions()
        action = self.agent.select_action(state, valid_actions)
        
        # Environment executes action
        next_state, reward, done, info = self.env.step(action)
        next_valid_actions = self.env.get_valid_actions()
        
        # Agent remembers experience
        self.agent.remember(state, action, reward, next_state, done, next_valid_actions)
        
        # Check that experience was stored
        self.assertGreater(len(self.agent.memory), 0)


if __name__ == "__main__":
    unittest.main()


```

```markdown
*** Begin Patch
*** Update File: docs/RL_IMPLEMENTATION_GUIDE.md
@@
-#### State Representation (Zustandsrepräsentation)
-
-Der State ist ein 2D-Array mit normalisierten Werten:
-
-```python
-State Encoding:
-- -0.9: Verdeckte Zelle (hidden)
-- -0.5: Flagge gesetzt (flagged)
-- -1.0: Mine aufgedeckt (mine revealed)
-- 0.0 - 1.0: Anzahl benachbarter Minen (normalisiert auf 0-1)
-  - 0/8 = 0.0 (keine Minen)
-  - 1/8 = 0.125
-  - ...
-  - 8/8 = 1.0 (alle Nachbarn sind Minen)
-```
-
-**Warum diese Encoding?**
-- Negative Werte für verdeckte/flagge Zellen unterscheiden sie von aufgedeckten
-- Normalisierte Werte (0-1) für Zahlen verbessern das Training
-- Ein Kanal (1, height, width) für einfache CNN-Architektur
+#### State Representation (Zustandsrepräsentation)
+
+Der State besitzt **7 Kanäle** und gibt dem Agenten explizit Auskunft über sein Umfeld:
+
+1. **Basis-Encoding** (`state[0]`): alter kombinierter Wert (-0.9 Hidden, -0.5 Flag, -1 Mine, 0-1 Zahlen) inkl. Nachbarschaftsbonus/-strafe  
+2. **Hidden-Maske** (`state[1]`): 1 für verdeckt, sonst 0  
+3. **Flag-Maske** (`state[2]`): 1 für gesetzte Flagge, sonst 0  
+4. **Aufgedeckte Zahlen** (`state[3]`): normierte Zahl (0-1) oder -1 bei Mine  
+5. **Verdeckte Nachbarn** (`state[4]`): Anteil verdeckter Nachbarn (0-1)  
+6. **Flag-Nachbarn** (`state[5]`): Anteil geflaggter Nachbarn (0-1)  
+7. **Hinweis-Summe** (`state[6]`): normierter Summenwert aller aufgedeckten Nachbarzahlen (0-1)
+
+Damit kennt der Agent nicht nur den Status einer Zelle, sondern auch die unmittelbare Umgebung (Frontier, Flaggen, vorhandene Hinweise).
@@
-```python
-if random() < epsilon:
-    # Exploration: Zufällige Aktion
-    action = random(valid_actions)
-else:
-    # Exploitation: Beste Aktion nach Q-Network
-    action = argmax(Q(state))
-```
-
-**Warum Epsilon-Greedy?**
-- **Exploration**: Agent lernt neue Strategien
-- **Exploitation**: Agent nutzt gelerntes Wissen
-- **Balance**: Epsilon decay von 1.0 → 0.01 über Training
+**Gezielte Exploration**
+
+- Zufällige Aktionen wählen bevorzugt **Frontier-Zellen** (verdeckte Felder neben bereits bekannten Zahlen)  
+- Falls keine Frontier verfügbar ist, werden Felder mit dem höchsten Hinweis-Summenwert gewählt  
+- Epsilon wird weiterhin per Episode decayed, aber Exploration fokussiert sich auf informative Bereiche
@@
-1. **Batch Sampling**: Zufällige Erfahrungen aus Replay Buffer
+1. **Batch Sampling**: Zufällige Erfahrungen inkl. Aktionsmasken aus Replay Buffer
@@
-4. **TD Target**: `target = reward + gamma * max(Q(s', a'))`
+4. **TD Target**: `target = reward + gamma * max(Q(s', a') | a' gültig)`
@@
-8. **Epsilon Decay**: Reduziert Exploration über Zeit
+8. **Episodenweises Epsilon Decay**: Nach jeder Episode via `agent.decay_epsilon()`
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

#### Action Space

- **Größe**: 600 Aktionen (20 × 30 = alle Zellen)
- **Action Index**: `action = row * BOARD_WIDTH + col`
- **Umkehrung**: `row = action // BOARD_WIDTH`, `col = action % BOARD_WIDTH`

#### Reward System

Das Reward-System ist **sparse** mit **shaped rewards**:

```python
Rewards:
- +100.0: Spiel gewonnen (große Belohnung)
- -10.0:  Mine getroffen (große Strafe)
- +0.1 × N: N Zellen aufgedeckt (kleine Belohnung pro Zelle)
- -0.1: Ungültige Aktion (kleine Strafe)
```

**Design-Entscheidungen:**
- **Sparse Hauptrewards**: Gewinn/Verlust sind die wichtigsten Signale
- **Shaped Rewards**: Fortschritt (aufgedeckte Zellen) wird belohnt
- **Penalties**: Verhindern ungültige Aktionen

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
Input: (batch, 1, 20, 30)  # State representation

Conv Layers:
- Conv2d(1 → 32, kernel=3, padding=1)  # Erste Feature-Ebene
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
2. **Target Network**: Stabilisiert Training durch verzögerte Updates
3. **Epsilon-Greedy**: Exploration vs. Exploitation
4. **Experience Replay**: Speichert und nutzt vergangene Erfahrungen

**Hyperparameter:**

```python
- Learning Rate: 0.001 (Adam Optimizer)
- Gamma (Discount): 0.99 (langfristige Belohnung)
- Epsilon Start: 1.0 (100% Exploration)
- Epsilon End: 0.01 (1% Exploration)
- Epsilon Decay: 0.995 (exponentielles Decay)
- Buffer Size: 10,000 Erfahrungen
- Batch Size: 32 Erfahrungen pro Training
- Target Update: Alle 100 Steps
```

**Epsilon-Greedy Strategie:**

```python
if random() < epsilon:
    # Exploration: Zufällige Aktion
    action = random(valid_actions)
else:
    # Exploitation: Beste Aktion nach Q-Network
    action = argmax(Q(state))
```

**Warum Epsilon-Greedy?**
- **Exploration**: Agent lernt neue Strategien
- **Exploitation**: Agent nutzt gelerntes Wissen
- **Balance**: Epsilon decay von 1.0 → 0.01 über Training

**Training Process (train_step):**

1. **Batch Sampling**: Zufällige Erfahrungen inkl. Aktionsmasken aus Replay Buffer
2. **Current Q-Values**: Q-Network schätzt Q(s, a)
3. **Target Q-Values**: Target Network schätzt Q(s', a') und maskiert ungültige Aktionen
4. **TD Target**: `target = reward + gamma * max(Q(s', a') | a' gültig)`
5. **Loss**: Mean Squared Error zwischen Current und Target
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

#### Action Space

- **Größe**: 600 Aktionen (20 × 30 = alle Zellen)
- **Action Index**: `action = row * BOARD_WIDTH + col`
- **Umkehrung**: `row = action // BOARD_WIDTH`, `col = action % BOARD_WIDTH`

#### Reward System

Das Reward-System ist **sparse** mit **shaped rewards**:

```python
Rewards:
- +100.0: Spiel gewonnen (große Belohnung)
- -10.0:  Mine getroffen (große Strafe)
- +0.1 × N: N Zellen aufgedeckt (kleine Belohnung pro Zelle)
- -0.1: Ungültige Aktion (kleine Strafe)
```

**Design-Entscheidungen:**
- **Sparse Hauptrewards**: Gewinn/Verlust sind die wichtigsten Signale
- **Shaped Rewards**: Fortschritt (aufgedeckte Zellen) wird belohnt
- **Penalties**: Verhindern ungültige Aktionen

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
2. **Target Network**: Stabilisiert Training durch verzögerte Updates
3. **Epsilon-Greedy**: Exploration vs. Exploitation
4. **Experience Replay**: Speichert und nutzt vergangene Erfahrungen

**Hyperparameter:**

```python
- Learning Rate: 0.001 (Adam Optimizer)
- Gamma (Discount): 0.99 (langfristige Belohnung)
- Epsilon Start: 1.0 (100% Exploration)
- Epsilon End: 0.01 (1% Exploration)
- Epsilon Decay: 0.995 (exponentielles Decay)
- Buffer Size: 10,000 Erfahrungen
- Batch Size: 32 Erfahrungen pro Training
- Target Update: Alle 100 Steps
```

**Epsilon-Greedy Strategie:**

```python
if random() < epsilon:
    # Exploration: Zufällige Aktion
    action = random(valid_actions)
else:
    # Exploitation: Beste Aktion nach Q-Network
    action = argmax(Q(state))
```

**Warum Epsilon-Greedy?**
- **Exploration**: Agent lernt neue Strategien
- **Exploitation**: Agent nutzt gelerntes Wissen
- **Balance**: Epsilon decay von 1.0 → 0.01 über Training

**Training Process (train_step):**

1. **Batch Sampling**: Zufällige Erfahrungen inkl. Aktionsmasken aus Replay Buffer
2. **Current Q-Values**: Q-Network schätzt Q(s, a)
3. **Target Q-Values**: Target Network schätzt Q(s', a') und maskiert ungültige Aktionen
4. **TD Target**: `target = reward + gamma * max(Q(s', a') | a' gültig)`
5. **Loss**: Mean Squared Error zwischen Current und Target
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

#### Action Space

- **Größe**: 600 Aktionen (20 × 30 = alle Zellen)
- **Action Index**: `action = row * BOARD_WIDTH + col`
- **Umkehrung**: `row = action // BOARD_WIDTH`, `col = action % BOARD_WIDTH`

#### Reward System

Das Reward-System ist **sparse** mit **shaped rewards**:

```python
Rewards:
- +100.0: Spiel gewonnen (große Belohnung)
- -10.0:  Mine getroffen (große Strafe)
- +0.1 × N: N Zellen aufgedeckt (kleine Belohnung pro Zelle)
- -0.1: Ungültige Aktion (kleine Strafe)
```

**Design-Entscheidungen:**
- **Sparse Hauptrewards**: Gewinn/Verlust sind die wichtigsten Signale
- **Shaped Rewards**: Fortschritt (aufgedeckte Zellen) wird belohnt
- **Penalties**: Verhindern ungültige Aktionen

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
2. **Target Network**: Stabilisiert Training durch verzögerte Updates
3. **Epsilon-Greedy**: Exploration vs. Exploitation
4. **Experience Replay**: Speichert und nutzt vergangene Erfahrungen

**Hyperparameter:**

```python
- Learning Rate: 0.001 (Adam Optimizer)
- Gamma (Discount): 0.99 (langfristige Belohnung)
- Epsilon Start: 1.0 (100% Exploration)
- Epsilon End: 0.01 (1% Exploration)
- Epsilon Decay: 0.995 (exponentielles Decay)
- Buffer Size: 10,000 Erfahrungen
- Batch Size: 32 Erfahrungen pro Training
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
3. **Target Q-Values**: Target Network schätzt Q(s', a') und maskiert ungültige Aktionen
4. **TD Target**: `target = reward + gamma * max(Q(s', a') | a' gültig)`
5. **Loss**: Mean Squared Error zwischen Current und Target
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

```python
"""Neural network for DQN."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        # Convolutional layers for spatial features with batch norm for better training
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Use adaptive pooling to handle different input sizes
        # This converts any spatial size to a fixed size (e.g., 8x8)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Calculate flattened size after adaptive pooling
        # After pooling: 8x8, with 128 channels
        flattened_size = 128 * 8 * 8
        
        # Fully connected layers with dropout for regularization
        self.fc1 = nn.Linear(flattened_size, 512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, num_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Q-values for each action
        """
        # Convolutional layers with batch norm and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Adaptive pooling to handle different sizes
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout (only during training)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x



```

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
        
        # Convolutional layers for spatial features with batch norm for better training
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Use adaptive pooling to handle different input sizes
        # This converts any spatial size to a fixed size (e.g., 8x8)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Calculate flattened size after adaptive pooling
        # After pooling: 8x8, with 128 channels
        flattened_size = 128 * 8 * 8
        
        # Fully connected layers with dropout for regularization
        self.fc1 = nn.Linear(flattened_size, 512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(256, num_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Q-values for each action
        """
        # Convolutional layers with batch norm and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Adaptive pooling to handle different sizes
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout (only during training)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x



```

## Änderungen
- **State-Encoding auf 7 Kanäle erweitert:** Das Environment liefert jetzt pro Zelle Basiswert, Hidden-/Flag-Masken sowie Nachbarschaftsstatistiken (verdeckte/flagged Nachbarn und Hinweis-Summe). Damit steht dem Agenten das unmittelbare Umfeld explizit zur Verfügung; Tests und Doku wurden entsprechend angepasst (`tests/reinforcement_learning/test_environment.py`, `tests/reinforcement_learning/test_network.py`, `tests/reinforcement_learning/test_dqn_agent.py`, `docs/RL_IMPLEMENTATION_GUIDE.md`).  
```91:134:src/reinforcement_learning/environment.py
state = np.zeros((STATE_CHANNELS, self.height, self.width), dtype=np.float32)
...
state[4, row, col] = hidden_neighbors / 8.0
state[5, row, col] = flagged_neighbors / 8.0
state[6, row, col] = min(1.0, adjacent_hint_sum / 64.0)
```
- **Gezielte Exploration implementiert:** Zufällige Schritte wählen nun bevorzugt Frontier-Zellen (versteckte Felder neben bekannten Zahlen) bzw. die stärksten Hinweiskandidaten, bevor auf komplett zufällige Aktionen zurückgegriffen wird. Dadurch bleibt Exploration fokussiert und nutzt das neue Encoding aus.  
```133:269:src/reinforcement_learning/dqn_agent.py
if random.random() < self.epsilon:
    if valid_actions is not None:
        frontier_action = self._sample_frontier_action(state, valid_actions)
        if frontier_action is not None:
            return frontier_action
...
def _sample_frontier_action(...):
    ...
    if frontier_indices.size > 0:
        return int(random.choice(frontier_indices.tolist()))
    if state.shape[0] > 6:
        hint_channel = state[6].reshape(-1)
        ...
```
- **Agent/Trainer/Doku synchronisiert:** `DQNAgent`, `DQNNetwork` und Trainer verwenden automatisch `env.state_channels`, Replay-Buffer speichert weiterhin Masken, und der Guide beschreibt das neue Encoding sowie das Frontier-Sampling.

## Tests
- `python -m pytest tests\reinforcement_learning` *(fehlgeschlagen: Modul `pytest` ist nicht installiert)*

Bitte `pip install pytest` (oder `python -m pip install pytest`) ausführen, falls du die Tests lokal verifizieren möchtest.

```python
state = np.zeros((STATE_CHANNELS, self.height, self.width), dtype=np.float32)
...
state[4, row, col] = hidden_neighbors / 8.0
state[5, row, col] = flagged_neighbors / 8.0
state[6, row, col] = min(1.0, adjacent_hint_sum / 64.0)
```

```python
if random.random() < self.epsilon:
    if valid_actions is not None:
        frontier_action = self._sample_frontier_action(state, valid_actions)
        if frontier_action is not None:
            return frontier_action
...
def _sample_frontier_action(...):
    ...
    if frontier_indices.size > 0:
        return int(random.choice(frontier_indices.tolist()))
    if state.shape[0] > 6:
        hint_channel = state[6].reshape(-1)
        ...
```

---

**User**

Lasse den Agenten auch flaggen setzten, sobald er eine Bombe trifft, soll er anschauen, wo er richtige Flagen gesetzt hat. Für jede richtig gesetzte Flagge solle er noch einen zusätzlichen reward bekommen, für falsch gesetzte Flagen soll er bestraft werden. Macht diese überlegung sinn oder muss noch etwas daran geändert werden?

---

**Cursor**

Die Idee ist grundsätzlich sinnvoll, aber ein paar Punkte solltest du vorher klären:

- **Aktionsraum erweitern**: Momentan hat der Agent nur „aufdecken“-Aktionen. Damit er Flaggen setzen/entfernen kann, brauchst du pro Feld mindestens zwei Aktionen (aufdecken/flaggen) oder eine separate Aktionsklasse fürs Flaggen. Ohne diese Erweiterung kann der Agent keine Flaggenentscheidungen lernen.

- **Zeitpunkt der Belohnung**: Wenn du erst beim Verlust prüfst, ob Flaggen korrekt waren, kommt das Feedback sehr spät (Credit Assignment Problem). Besser: immer dann belohnen/ bestrafen, wenn eine Flagge gesetzt oder wieder entfernt wird und sich später bestätigt. Alternativ kannst du Flaggen-Rewards beim Aufdecken angrenzender Zahlen vergeben (z. B. wenn eine Zahl genau so viele Flaggen wie Minen anzeigt).

- **Belohnungen balancieren**: Zusätzliche Rewards sollten kleiner ausfallen als ein Sieg, sonst optimiert der Agent nur auf Flaggen statt auf sichere Züge. Eine sinnvolle Reihenfolge wäre: Sieg ≫ korrektes Aufdecken ≫ Flaggen-Feedback ≫ kleine Strafe für Fehlschläge.

- **State & Valid-Actions aktualisieren**: Wenn Flaggen Teil der Policy sind, braucht der Agent im Zustand Informationen darüber, welche Zellen er selbst markiert hat (haben wir bereits). Zusätzlich musst du gültige Flag-Aktionen maskieren (z. B. keine Flagge auf bereits aufgedeckte Zelle, Entfernen nur wenn Flag vorhanden).

- **Konsequenzen bei Verlust/Gewinn**: Eine schnelle Analyse „wie viele Flaggen waren korrekt“ kann als Bonus/Strafe dienen, aber sie sollte das unmittelbare Lernsignal beim Flaggen nicht ersetzen.

Fazit: Die Überlegung ist gut, solange du den Aktionsraum erweiterst und das Feedback zeitnah gibst. Ich empfehle: (1) Flag-Aktion pro Feld hinzufügen, (2) Belohnung sofort beim Setzen/Entfernen vergeben, (3) optional einen kleinen Bonus am Episodenende für korrekt markierte Minen. Sag Bescheid, wenn ich dir bei der Umsetzung helfen soll.

---

**User**

Setzte mir alles um, beziehe dabei die Punkte mit ein, welche du vorher geschrieben hast.

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
        
        if action_type == "reveal":
            valid = self.game.reveal_cell(row, col)
            reward += self._calculate_reward(prev_revealed, valid)
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
            State array of shape (1, height, width)
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
    
    def _calculate_reward(self, prev_revealed: int, valid_action: bool) -> float:
        """
        Calculate reward for the action.
        
        Args:
            prev_revealed: Number of revealed cells before action
            valid_action: Whether the action was valid
            
        Returns:
            Reward value
        """
        if not valid_action:
            return -0.5  # Penalty for invalid action
        
        if self.game.is_lost():
            # Penalty for hitting a mine - balanced to allow learning
            # Too large penalty prevents learning from mistakes
            # Scale with board size: smaller boards = less penalty (fewer cells to learn from)
            base_penalty = -100.0
            size_factor = (self.width * self.height) / 600.0  # Normalize to 30x20
            penalty_scale = max(0.5, size_factor)  # Minimum 50% penalty on tiny boards
            return base_penalty * penalty_scale
        
        if self.game.is_won():
            # Reward based on board size - larger boards = more reward
            base_reward = 1000.0
            size_bonus = (self.width * self.height) / 10.0
            return base_reward + size_bonus
        
        # Reward for revealing cells - LINEAR scaling, not quadratic
        cells_revealed = self.game.revealed_count - prev_revealed
        if cells_revealed > 0:
            # Linear reward: small base reward per cell, capped bonus for multiple cells
            # This prevents quadratic scaling
            base_reward_per_cell = 0.1
            # Small bonus for revealing multiple cells at once (capped)
            multi_cell_bonus = min(0.5, 0.02 * cells_revealed)  # Cap at 0.5
            
            # Linear reward: just sum of cells (no multiplication!)
            reward = base_reward_per_cell * cells_revealed + multi_cell_bonus
            
            # Small progress bonus (linear, not quadratic)
            progress = self.game.revealed_count / max(1, self.game.total_safe_cells)
            progress_bonus = 0.1 * progress  # Much smaller progress bonus
            
            return reward + progress_bonus
        
        # Small negative reward for no progress (to encourage exploration)
        return -0.01
    
    def get_valid_actions(self) -> np.ndarray:
        """
        Get mask of valid actions (cells that can be clicked).
        
        Returns:
            Boolean array of valid actions
        """
        valid = np.zeros(self.action_space_size, dtype=bool)
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                action_idx = row * self.width + col
                
                # Valid if cell is hidden and not flagged
                if cell.is_hidden() and not cell.is_flagged():
                    valid[action_idx] = True
        
        return valid
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get action mask for masking invalid actions in Q-values.
        
        Returns:
            Mask array (1.0 for valid, -inf for invalid)
        """
        valid_actions = self.get_valid_actions()
        mask = np.where(valid_actions, 0.0, -np.inf)
        return mask



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
        
        if action_type == "reveal":
            valid = self.game.reveal_cell(row, col)
            reward += self._calculate_reward(prev_revealed, valid)
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
            State array of shape (1, height, width)
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
    
    def _calculate_reward(self, prev_revealed: int, valid_action: bool) -> float:
        """
        Calculate reward for the action.
        
        Args:
            prev_revealed: Number of revealed cells before action
            valid_action: Whether the action was valid
            
        Returns:
            Reward value
        """
        if not valid_action:
            return -0.5  # Penalty for invalid action
        
        if self.game.is_lost():
            # Penalty for hitting a mine - balanced to allow learning
            # Too large penalty prevents learning from mistakes
            # Scale with board size: smaller boards = less penalty (fewer cells to learn from)
            base_penalty = -100.0
            size_factor = (self.width * self.height) / 600.0  # Normalize to 30x20
            penalty_scale = max(0.5, size_factor)  # Minimum 50% penalty on tiny boards
            return base_penalty * penalty_scale
        
        if self.game.is_won():
            # Reward based on board size - larger boards = more reward
            base_reward = 1000.0
            size_bonus = (self.width * self.height) / 10.0
            return base_reward + size_bonus
        
        # Reward for revealing cells - LINEAR scaling, not quadratic
        cells_revealed = self.game.revealed_count - prev_revealed
        if cells_revealed > 0:
            # Linear reward: small base reward per cell, capped bonus for multiple cells
            # This prevents quadratic scaling
            base_reward_per_cell = 0.1
            # Small bonus for revealing multiple cells at once (capped)
            multi_cell_bonus = min(0.5, 0.02 * cells_revealed)  # Cap at 0.5
            
            # Linear reward: just sum of cells (no multiplication!)
            reward = base_reward_per_cell * cells_revealed + multi_cell_bonus
            
            # Small progress bonus (linear, not quadratic)
            progress = self.game.revealed_count / max(1, self.game.total_safe_cells)
            progress_bonus = 0.1 * progress  # Much smaller progress bonus
            
            return reward + progress_bonus
        
        # Small negative reward for no progress (to encourage exploration)
        return -0.01
    
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
        valid = np.zeros(self.action_space_size, dtype=bool)
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                action_idx = row * self.width + col
                
                # Valid if cell is hidden and not flagged
                if cell.is_hidden() and not cell.is_flagged():
                    valid[action_idx] = True
        
        return valid
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get action mask for masking invalid actions in Q-values.
        
        Returns:
            Mask array (1.0 for valid, -inf for invalid)
        """
        valid_actions = self.get_valid_actions()
        mask = np.where(valid_actions, 0.0, -np.inf)
        return mask



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
        
        if action_type == "reveal":
            valid = self.game.reveal_cell(row, col)
            reward += self._calculate_reward(prev_revealed, valid)
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
            State array of shape (1, height, width)
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
    
    def _calculate_reward(self, prev_revealed: int, valid_action: bool) -> float:
        """
        Calculate reward for the action.
        
        Args:
            prev_revealed: Number of revealed cells before action
            valid_action: Whether the action was valid
            
        Returns:
            Reward value
        """
        if not valid_action:
            return -0.5  # Penalty for invalid action
        
        if self.game.is_lost():
            # Penalty for hitting a mine - balanced to allow learning
            # Too large penalty prevents learning from mistakes
            # Scale with board size: smaller boards = less penalty (fewer cells to learn from)
            base_penalty = -100.0
            size_factor = (self.width * self.height) / 600.0  # Normalize to 30x20
            penalty_scale = max(0.5, size_factor)  # Minimum 50% penalty on tiny boards
            return base_penalty * penalty_scale
        
        if self.game.is_won():
            # Reward based on board size - larger boards = more reward
            base_reward = 1000.0
            size_bonus = (self.width * self.height) / 10.0
            return base_reward + size_bonus
        
        # Reward for revealing cells - LINEAR scaling, not quadratic
        cells_revealed = self.game.revealed_count - prev_revealed
        if cells_revealed > 0:
            # Linear reward: small base reward per cell, capped bonus for multiple cells
            # This prevents quadratic scaling
            base_reward_per_cell = 0.1
            # Small bonus for revealing multiple cells at once (capped)
            multi_cell_bonus = min(0.5, 0.02 * cells_revealed)  # Cap at 0.5
            
            # Linear reward: just sum of cells (no multiplication!)
            reward = base_reward_per_cell * cells_revealed + multi_cell_bonus
            
            # Small progress bonus (linear, not quadratic)
            progress = self.game.revealed_count / max(1, self.game.total_safe_cells)
            progress_bonus = 0.1 * progress  # Much smaller progress bonus
            
            return reward + progress_bonus
        
        # Small negative reward for no progress (to encourage exploration)
        return -0.01
    
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
        
        if action_type == "reveal":
            valid = self.game.reveal_cell(row, col)
            reward += self._calculate_reward(prev_revealed, valid)
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
    
    def _calculate_reward(self, prev_revealed: int, valid_action: bool) -> float:
        """
        Calculate reward for the action.
        
        Args:
            prev_revealed: Number of revealed cells before action
            valid_action: Whether the action was valid
            
        Returns:
            Reward value
        """
        if not valid_action:
            return -0.5  # Penalty for invalid action
        
        if self.game.is_lost():
            # Penalty for hitting a mine - balanced to allow learning
            # Too large penalty prevents learning from mistakes
            # Scale with board size: smaller boards = less penalty (fewer cells to learn from)
            base_penalty = -100.0
            size_factor = (self.width * self.height) / 600.0  # Normalize to 30x20
            penalty_scale = max(0.5, size_factor)  # Minimum 50% penalty on tiny boards
            return base_penalty * penalty_scale
        
        if self.game.is_won():
            # Reward based on board size - larger boards = more reward
            base_reward = 1000.0
            size_bonus = (self.width * self.height) / 10.0
            return base_reward + size_bonus
        
        # Reward for revealing cells - LINEAR scaling, not quadratic
        cells_revealed = self.game.revealed_count - prev_revealed
        if cells_revealed > 0:
            # Linear reward: small base reward per cell, capped bonus for multiple cells
            # This prevents quadratic scaling
            base_reward_per_cell = 0.1
            # Small bonus for revealing multiple cells at once (capped)
            multi_cell_bonus = min(0.5, 0.02 * cells_revealed)  # Cap at 0.5
            
            # Linear reward: just sum of cells (no multiplication!)
            reward = base_reward_per_cell * cells_revealed + multi_cell_bonus
            
            # Small progress bonus (linear, not quadratic)
            progress = self.game.revealed_count / max(1, self.game.total_safe_cells)
            progress_bonus = 0.1 * progress  # Much smaller progress bonus
            
            return reward + progress_bonus
        
        # Small negative reward for no progress (to encourage exploration)
        return -0.01
    
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
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
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
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.masked_fill(~next_valid_actions, -1e9)
            next_q = next_q_values.max(1)[0]
            target_q = rewards + (1 - dones.float()) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
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
    
    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.difficulty, "medium")
        self.assertEqual(self.env.action_space_size, BOARD_WIDTH * BOARD_HEIGHT * 2)
        self.assertIsNotNone(self.env.game)
    
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
        self.assertEqual(valid_actions.shape, (BOARD_WIDTH * BOARD_HEIGHT,))
        self.assertEqual(valid_actions.dtype, bool)
        
        # Initially all actions should be valid (all cells hidden)
        self.assertTrue(np.all(valid_actions))
        
        # After first action, some should become invalid
        self.env.step(0)
        valid_actions = self.env.get_valid_actions()
        self.assertFalse(np.all(valid_actions))  # At least some should be invalid
    
    def test_get_action_mask(self):
        """Test action masking."""
        state = self.env.reset()
        
        mask = self.env.get_action_mask()
        
        # Check shape
        self.assertEqual(mask.shape, (BOARD_WIDTH * BOARD_HEIGHT,))
        
        # Valid actions should be 0.0, invalid should be -inf
        valid_actions = self.env.get_valid_actions()
        self.assertTrue(np.all(mask[valid_actions] == 0.0))
        self.assertTrue(np.all(np.isinf(mask[~valid_actions])))
    
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
        self.env = MinesweeperEnvironment("medium")
    
    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.difficulty, "medium")
        self.assertEqual(self.env.action_space_size, BOARD_WIDTH * BOARD_HEIGHT * 2)
        self.assertIsNotNone(self.env.game)
    
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

Kombination aus Fortschritts- und Flaggen-Rewards:

- **Aufdecken**: identisch zum ursprünglichen System (lineare Belohnung für neue Felder, kleiner Fortschrittsbonus, -0.01 für Stillstand)
- **Flaggen (sofort)**:
  - +0.2, wenn eine Mine korrekt markiert wird
  - -0.2, wenn ein sicheres Feld fälschlich markiert wird
  - 0.0 beim Entfernen einer Flagge (kein Schwingen)
- **Flaggen (bei Verlust)**:
  - +0.5 pro korrekt gesetzter Flagge
  - -0.5 pro falscher Flagge
- **Spiel verloren**: Basisstrafe skaliert weiterhin mit Brettgröße
- **Spiel gewonnen**: Große positive Belohnung, abhängig von der Feldanzahl

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
2. **Target Network**: Stabilisiert Training durch verzögerte Updates
3. **Epsilon-Greedy**: Exploration vs. Exploitation
4. **Experience Replay**: Speichert und nutzt vergangene Erfahrungen

**Hyperparameter:**

```python
- Learning Rate: 0.001 (Adam Optimizer)
- Gamma (Discount): 0.99 (langfristige Belohnung)
- Epsilon Start: 1.0 (100% Exploration)
- Epsilon End: 0.01 (1% Exploration)
- Epsilon Decay: 0.995 (exponentielles Decay)
- Buffer Size: 10,000 Erfahrungen
- Batch Size: 32 Erfahrungen pro Training
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
3. **Target Q-Values**: Target Network schätzt Q(s', a') und maskiert ungültige Aktionen
4. **TD Target**: `target = reward + gamma * max(Q(s', a') | a' gültig)`
5. **Loss**: Mean Squared Error zwischen Current und Target
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

```python
"""Tests for DQNAgent and ReplayBuffer."""

import unittest
import numpy as np
import torch
import os
import tempfile
from src.reinforcement_learning.dqn_agent import DQNAgent, ReplayBuffer
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class TestReplayBuffer(unittest.TestCase):
    """Test cases for ReplayBuffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ReplayBuffer(capacity=100)
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT * 2
    
    def _random_state(self):
        return np.random.randn(STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
    
    def test_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(len(self.buffer), 0)
    
    def test_push(self):
        """Test pushing experiences."""
        state = self._random_state()
        action = 0
        reward = 1.0
        next_state = self._random_state()
        done = False
        next_valid = np.ones(self.action_space_size, dtype=bool)
        
        self.buffer.push(state, action, reward, next_state, done, next_valid)
        self.assertEqual(len(self.buffer), 1)
    
    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)
        
        for i in range(15):
            state = self._random_state()
            next_valid = np.ones(self.action_space_size, dtype=bool)
            buffer.push(state, 0, 1.0, state, False, next_valid)
        
        # Should not exceed capacity
        self.assertEqual(len(buffer), 10)
    
    def test_sample(self):
        """Test sampling from buffer."""
        # Add some experiences
        for i in range(32):
            state = self._random_state()
            next_valid = np.ones(self.action_space_size, dtype=bool)
            self.buffer.push(state, i % 10, float(i), state, i % 2 == 0, next_valid)
        
        # Sample batch
        batch_size = 16
        states, actions, rewards, next_states, dones, next_valid_actions = self.buffer.sample(batch_size)
        
        # Check types
        self.assertIsInstance(states, torch.Tensor)
        self.assertIsInstance(actions, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)
        self.assertIsInstance(next_valid_actions, torch.Tensor)
        
        # Check shapes
        self.assertEqual(states.shape[0], batch_size)
        self.assertEqual(actions.shape[0], batch_size)
        self.assertEqual(rewards.shape[0], batch_size)
        self.assertEqual(next_states.shape[0], batch_size)
        self.assertEqual(dones.shape[0], batch_size)
        self.assertEqual(next_valid_actions.shape[0], batch_size)


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQNAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT * 2
        self.agent = DQNAgent(
            state_channels=STATE_CHANNELS,
            action_space_size=self.action_space_size,
            lr=0.001,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=1000,
            batch_size=32,
            target_update=100
        )
        self.state_shape = (STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH)
    
    def _all_valid_actions(self):
        """Helper to generate a fully valid action mask."""
        return np.ones(self.action_space_size, dtype=bool)
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.action_space_size, self.action_space_size)
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertIsNotNone(self.agent.q_network)
        self.assertIsNotNone(self.agent.target_network)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.memory)
    
    def test_select_action_random(self):
        """Test action selection in random mode (epsilon=1)."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        
        # With epsilon=1, should always be random
        actions = [self.agent.select_action(state) for _ in range(10)]
        
        # Should be valid actions
        for action in actions:
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.action_space_size)
    
    def test_select_action_greedy(self):
        """Test action selection in greedy mode (epsilon=0)."""
        self.agent.epsilon = 0.0
        state = np.random.randn(*self.state_shape).astype(np.float32)
        
        action = self.agent.select_action(state)
        
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_space_size)
    
    def test_select_action_with_valid_mask(self):
        """Test action selection with valid actions mask."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        valid_actions = np.zeros(self.action_space_size, dtype=bool)
        valid_actions[0:10] = True  # Only first 10 actions are valid
        
        self.agent.epsilon = 0.0  # Greedy mode
        action = self.agent.select_action(state, valid_actions)
        
        # Should be one of the valid actions
        self.assertIn(action, range(10))
    
    def test_remember(self):
        """Test storing experiences."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        action = 5
        reward = 1.0
        next_state = np.random.randn(*self.state_shape).astype(np.float32)
        done = False
        
        initial_len = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        self.assertEqual(len(self.agent.memory), initial_len + 1)
    
    def test_train_step_insufficient_samples(self):
        """Test training with insufficient samples."""
        # Should return None if not enough samples
        loss = self.agent.train_step()
        self.assertIsNone(loss)
    
    def test_train_step_sufficient_samples(self):
        """Test training with sufficient samples."""
        # Add enough experiences
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            action = i % self.action_space_size
            reward = np.random.randn()
            next_state = np.random.randn(*self.state_shape).astype(np.float32)
            done = i % 10 == 0
            
            self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        
        # Train step
        loss = self.agent.train_step()
        
        # Should return a loss value
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)  # Loss should be non-negative
    
    def test_epsilon_decay(self):
        """Test epsilon decay."""
        initial_epsilon = self.agent.epsilon
        
        # Train multiple times to trigger epsilon decay
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train to trigger epsilon decay
        self.agent.train_step()
        
        # Epsilon should stay the same until decay is triggered explicitly
        self.assertEqual(self.agent.epsilon, initial_epsilon)
        
        # After calling decay, epsilon should decrease
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_end)
    
    def test_save_and_load(self):
        """Test saving and loading agent."""
        # Add some experiences and train
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        self.agent.train_step()
        initial_epsilon = self.agent.epsilon
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            tmp_path = tmp.name
        
        try:
            self.agent.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Create new agent and load
            new_agent = DQNAgent(
                state_channels=STATE_CHANNELS,
                action_space_size=self.action_space_size
            )
            new_agent.load(tmp_path)
            
            # Check that epsilon was loaded
            self.assertEqual(new_agent.epsilon, initial_epsilon)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_target_network_update(self):
        """Test target network update."""
        initial_update_counter = self.agent.update_counter
        
        # Train multiple times to trigger target update
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train enough times to trigger target update
        for _ in range(self.agent.target_update + 1):
            if len(self.agent.memory) >= self.agent.batch_size:
                self.agent.train_step()
        
        # Update counter should have increased
        self.assertGreater(self.agent.update_counter, initial_update_counter)


class TestDQNAgentIntegration(unittest.TestCase):
    """Integration tests for DQNAgent with Environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MinesweeperEnvironment("medium")
        self.agent = DQNAgent(
            state_channels=STATE_CHANNELS,
            action_space_size=self.env.action_space_size,
            buffer_size=100,
            batch_size=16
        )
    
    def test_agent_environment_interaction(self):
        """Test interaction between agent and environment."""
        state = self.env.reset()
        
        # Agent selects action
        valid_actions = self.env.get_valid_actions()
        action = self.agent.select_action(state, valid_actions)
        
        # Environment executes action
        next_state, reward, done, info = self.env.step(action)
        next_valid_actions = self.env.get_valid_actions()
        
        # Agent remembers experience
        self.agent.remember(state, action, reward, next_state, done, next_valid_actions)
        
        # Check that experience was stored
        self.assertGreater(len(self.agent.memory), 0)


if __name__ == "__main__":
    unittest.main()


```

## RL-Agent mit Flaggen-Aktionen

- `MinesweeperEnvironment`: Aktionsraum verdoppelt (Aufdecken + Flaggen je Zelle), Status liefert jetzt 7 Kanäle inkl. Nachbarschaftsinfos. Zusätzlich gibt es unmittelbare Flaggen-Rewards (+0.2 korrekt, ‑0.2 falsch) und einen Bonus/Strafterm beim Game-Over, wenn zuvor eine Mine getroffen wurde. `get_valid_actions` und `get_action_mask` liefern Masken für beide Aktionstypen, Tests decken die neuen Formen und Flaggen-Rewards ab (`tests/reinforcement_learning/test_environment.py`).
- `DQNAgent`: Kennt die neue Aktionsgröße, bevorzugt bei Exploration weiterhin Frontier-Reveal-Zellen, fällt aber sauber auf Flaggenaktionen zurück. Replay/Training bleibt unverändert, nutzt jedoch die längeren Masken und Channels (`src/reinforcement_learning/dqn_agent.py`).
- `DQNNetwork`, Trainer und GUI nutzen automatisch `env.state_channels`; Dokumentation erklärt nun die 7-kanalige Darstellung, das erweiterte Aktionsschema und das Flaggen-Rewarding (`docs/RL_IMPLEMENTATION_GUIDE.md`).
- Tests und Helfer klassisch angepasst (`tests/reinforcement_learning/test_network.py`, `tests/reinforcement_learning/test_dqn_agent.py`).

## Tests
- `python -m pytest tests\reinforcement_learning` *(scheitert, weil `pytest` im aktuellen Python nicht installiert ist)*

Bitte `python -m pip install pytest` ausführen und die Suite erneut laufen lassen, sobald möglich.

## Nächste Schritte
- Nach Installation von `pytest` komplette Testsuite laufen lassen.
- Kürzere Trainingsläufe (z. B. 200 Episoden, kleines Brett) durchführen, um die Wirkung der Flaggen-Belohnungen zu beobachten.
- Optional: Reward-Gewichte weiter feinjustieren, falls das Flaggen-Verhalten zu aggressiv oder zu vorsichtig ausfällt.

