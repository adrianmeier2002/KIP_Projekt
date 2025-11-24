"""RL-Visualisierer zur Darstellung des Agent-Gameplays."""

import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from src.minesweeper.game import Game
from src.gui.game_board import GameBoard
from src.reinforcement_learning.environment import MinesweeperEnvironment
from src.reinforcement_learning.dqn_agent import DQNAgent
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class RLVisualizer(QWidget):
    """Widget zur Visualisierung des RL-Agent-Gameplays."""
    
    episode_finished = Signal(bool, float, int)  # won, reward, steps
    
    def __init__(self, game: Game, game_board: GameBoard):
        """
        Initialisiert den RL-Visualisierer.
        
        Args:
            game: Spiel-Instanz
            game_board: GameBoard-Widget zur Aktualisierung
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
        """Richtet die Benutzeroberfläche für den RL-Visualisierer ein."""
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
        """Setzt den zu verwendenden RL-Agent."""
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
        
        # Agent selects action (pass game for hybrid agent solver)
        action = self.agent.select_action(self.current_state, valid_actions, game=self.env.game)
        
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

