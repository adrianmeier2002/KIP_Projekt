"""Game board widget for Minesweeper GUI."""

from PySide6.QtWidgets import QWidget, QGridLayout, QPushButton, QSizePolicy
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QFont
from src.minesweeper.game import Game


class CellButton(QPushButton):
    """Custom button for a game cell."""
    
    left_clicked = Signal(int, int)
    right_clicked = Signal(int, int)
    
    def __init__(self, row: int, col: int):
        """Initialize cell button."""
        super().__init__()
        self.row = row
        self.col = col
        # Expanding policy damit Buttons das ganze Fenster füllen
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(20, 20)
        self.setFont(QFont("Arial", 10, QFont.Bold))
        self.setText("")
        self.setCursor(Qt.PointingHandCursor)
        # Entferne alle möglichen Margins
        self.setContentsMargins(0, 0, 0, 0)
    
    def sizeHint(self):
        """Return preferred size for button."""
        # Return a square size hint
        size = 40  # Default size
        return QSize(size, size)
    
    def hasHeightForWidth(self):
        """Indicate that height depends on width."""
        return True
    
    def heightForWidth(self, width):
        """Return height for given width to maintain square aspect ratio."""
        return width
    
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.LeftButton:
            self.left_clicked.emit(self.row, self.col)
        elif event.button() == Qt.RightButton:
            self.right_clicked.emit(self.row, self.col)
        super().mousePressEvent(event)


class GameBoard(QWidget):
    """Widget for displaying and interacting with the game board."""
    
    game_won = Signal()
    game_lost = Signal()
    
    def __init__(self, game: Game):
        """
        Initialize game board widget.
        
        Args:
            game: Game instance
        """
        super().__init__()
        self.game = game
        self.buttons: list[list[CellButton]] = []
        self._setup_ui()
        self._update_display()
    
    def _setup_ui(self):
        """Setup the UI layout."""
        self.layout = QGridLayout()
        # Explizit beide Spacing-Werte auf 0 setzen
        self.layout.setHorizontalSpacing(0)
        self.layout.setVerticalSpacing(0)
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        # Hintergrundfarbe auf Rasengrün setzen, damit keine dunklen Linien sichtbar sind
        self.setStyleSheet("""
            QWidget {
                background-color: #4a7c59;
                padding: 0px;
                margin: 0px;
            }
        """)
        self._create_buttons()
    
    def _create_buttons(self):
        """Create buttons for current game size."""
        # Clear existing buttons
        for row in self.buttons:
            for button in row:
                button.deleteLater()
        self.buttons.clear()
        
        # Create new buttons based on game size
        height = self.game.height
        width = self.game.width
        
        for row in range(height):
            button_row = []
            for col in range(width):
                button = CellButton(row, col)
                button.left_clicked.connect(self._on_left_click)
                button.right_clicked.connect(self._on_right_click)
                self.layout.addWidget(button, row, col)
                button_row.append(button)
            self.buttons.append(button_row)
        
        # Setze Stretch-Faktoren auf 1 für gleichmäßige Verteilung
        for row in range(height):
            self.layout.setRowStretch(row, 1)
        for col in range(width):
            self.layout.setColumnStretch(col, 1)
    
    def resizeEvent(self, event):
        """Handle resize events to update font size."""
        super().resizeEvent(event)
        self._update_font_sizes()
    
    def _update_font_sizes(self):
        """Update font sizes based on button size."""
        if not self.buttons or not self.game:
            return
        
        # Buttons passen sich automatisch an durch Expanding Policy
        # Wir müssen nur die Schriftgröße anpassen
        if len(self.buttons) > 0 and len(self.buttons[0]) > 0:
            sample_button = self.buttons[0][0]
            button_size = min(sample_button.width(), sample_button.height())
            
            if button_size > 0:
                font_size = max(8, min(20, button_size // 2))
                for row_buttons in self.buttons:
                    for button in row_buttons:
                        font = button.font()
                        font.setPointSize(font_size)
                        button.setFont(font)
    
    def _on_left_click(self, row: int, col: int):
        """Handle left click on a cell."""
        # Always update display after reveal attempt (even if False, to show mines on loss)
        result = self.game.reveal_cell(row, col)
        self._update_display()
        
        # Check game state after update
        if self.game.is_won():
            self.game_won.emit()
        elif self.game.is_lost():
            self.game_lost.emit()
    
    def _on_right_click(self, row: int, col: int):
        """Handle right click on a cell (flag toggle)."""
        if self.game.toggle_flag(row, col):
            self._update_display()
    
    def _update_display(self):
        """Update the visual display of all cells."""
        height = self.game.height
        width = self.game.width
        
        for row in range(height):
            for col in range(width):
                cell = self.game.board.get_cell(row, col)
                if not cell or row >= len(self.buttons) or col >= len(self.buttons[row]):
                    continue
                
                button = self.buttons[row][col]
                
                # Schachbrettmuster für Rasen-Effekt (Hell- und Dunkelgrün)
                is_dark_square = (row + col) % 2 == 0
                grass_dark = "#4a7c59"   # Dunkelgrün
                grass_light = "#5d9c6c"  # Hellgrün
                
                if cell.is_revealed():
                    button.setEnabled(False)
                    if cell.is_mine:
                        button.setText("💣")
                        button.setStyleSheet("""
                            QPushButton {
                                background-color: #8b4513;
                                color: white;
                                font-weight: bold;
                                border: none;
                                outline: none;
                                padding: 0px;
                                margin: 0px;
                            }
                        """)
                    elif cell.adjacent_mines > 0:
                        button.setText(str(cell.adjacent_mines))
                        # Helle, kontrastreiche Farben für bessere Lesbarkeit auf Braun
                        colors = {
                            1: "#5dade2",  # Helles Blau
                            2: "#52d98c",  # Helles Grün
                            3: "#ff6b6b",  # Helles Rot
                            4: "#c39bd3",  # Helles Lila
                            5: "#f8b739",  # Helles Orange/Gelb
                            6: "#48c9b0",  # Helles Türkis
                            7: "#ecf0f1",  # Fast Weiß
                            8: "#bdc3c7"   # Hellgrau
                        }
                        color = colors.get(cell.adjacent_mines, "#ecf0f1")
                        button.setStyleSheet(f"""
                            QPushButton {{
                                background-color: #8b6914;
                                color: {color};
                                font-weight: bold;
                                border: none;
                                outline: none;
                                padding: 0px;
                                margin: 0px;
                            }}
                        """)
                    else:
                        button.setText("")
                        button.setStyleSheet("""
                            QPushButton {
                                background-color: #a0826d;
                                border: none;
                                outline: none;
                                padding: 0px;
                                margin: 0px;
                            }
                        """)
                elif cell.is_flagged():
                    # Flagge auf Rasen
                    grass_color = grass_dark if is_dark_square else grass_light
                    button.setText("🚩")
                    button.setStyleSheet(f"""
                        QPushButton {{
                            background-color: {grass_color};
                            color: white;
                            font-weight: bold;
                            border: none;
                            outline: none;
                            padding: 0px;
                            margin: 0px;
                        }}
                    """)
                    button.setEnabled(True)
                else:
                    # Verdecktes Feld - Rasen mit Schachbrettmuster
                    grass_color = grass_dark if is_dark_square else grass_light
                    grass_hover = "#6bb77b" if is_dark_square else "#7dd68d"
                    button.setText("")
                    button.setStyleSheet(f"""
                        QPushButton {{
                            background-color: {grass_color};
                            border: none;
                            outline: none;
                            padding: 0px;
                            margin: 0px;
                        }}
                        QPushButton:hover {{
                            background-color: {grass_hover};
                            outline: none;
                        }}
                        QPushButton:pressed {{
                            background-color: #3d6647;
                            outline: none;
                        }}
                    """)
                    button.setEnabled(True)
    
    def reset_game(self, game: Game):
        """Reset the game board with a new game."""
        # Check if size changed
        size_changed = (self.game.width != game.width or self.game.height != game.height)
        self.game = game
        
        if size_changed:
            self._create_buttons()
        
        self._update_display()
        # Update font sizes after reset
        self._update_font_sizes()


