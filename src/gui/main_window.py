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
        # Starte mit kleinem Spielfeld (15x10) für bessere Übersicht
        self.game = Game("easy", width=15, height=10)
        self.timer = QTimer()
        self.elapsed_time = 0
        self.rl_agent = None
        self.training_thread = None
        self._setup_ui()
        self._setup_timer()
    
    def _setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("Minesweeper mit Hybrid AI - Leicht (15x10)")
        self.setMinimumSize(800, 600)
        
        # Central widget mit Styling
        central_widget = QWidget()
        central_widget.setStyleSheet("""
            QWidget {
                background-color: #34495e;
            }
        """)
        self.setCentralWidget(central_widget)
        
        # Layout
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
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
        
        # Status bar mit verbessertem Styling
        self.status_label = QLabel("Minen: 0 | Zeit: 0s")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #2c3e50;
                color: white;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
                margin: 5px;
            }
        """)
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
        from src.reinforcement_learning.hybrid_agent import HybridAgent
        
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
            
            # Prüfe ob es ein Hybrid-Modell ist
            is_hybrid = checkpoint.get("is_hybrid", False)
            use_solver = checkpoint.get("use_solver", True)
            
            # Erstelle den richtigen Agent-Typ
            if is_hybrid:
                agent = HybridAgent(
                    state_channels=STATE_CHANNELS,
                    action_space_size=action_space_size,
                    board_height=board_height,
                    board_width=board_width,
                    lr=0.001,
                    gamma=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.01,
                    epsilon_decay=0.995,
                    use_solver=use_solver
                )
                agent_type = "Hybrid Agent (Solver + RL)"
            else:
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
                agent_type = "Pure DQN Agent"
            
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
                    f"Agent-Typ: {agent_type}\n"
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


