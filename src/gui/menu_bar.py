"""Menu bar for Minesweeper GUI."""

from PySide6.QtWidgets import QMenuBar, QMenu
from PySide6.QtCore import Signal


class MenuBar(QMenuBar):
    """Menu bar with game options."""
    
    new_game_easy = Signal()
    new_game_medium = Signal()
    new_game_hard = Signal()
    change_board_size = Signal()
    start_rl_training = Signal()
    load_rl_model = Signal()
    quit_requested = Signal()
    
    def __init__(self):
        """Initialize menu bar."""
        super().__init__()
        self._setup_menus()
    
    def _setup_menus(self):
        """Setup menu items."""
        # Game menu
        game_menu = self.addMenu("Spiel")
        
        new_game_menu = game_menu.addMenu("Neues Spiel")
        new_game_menu.addAction("Leicht", self.new_game_easy.emit)
        new_game_menu.addAction("Mittel", self.new_game_medium.emit)
        new_game_menu.addAction("Schwer", self.new_game_hard.emit)
        
        game_menu.addSeparator()
        game_menu.addAction("Spielfeldgröße ändern", self.change_board_size.emit)
        
        game_menu.addSeparator()
        
        # RL menu
        rl_menu = self.addMenu("Reinforcement Learning")
        rl_menu.addAction("Training starten (mit Visualisierung)", self.start_rl_training.emit)
        rl_menu.addAction("Modell laden und testen", self.load_rl_model.emit)
        
        game_menu.addSeparator()
        game_menu.addAction("Beenden", self.quit_requested.emit)


