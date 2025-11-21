"""Game board widget for Minesweeper GUI."""

from PySide6.QtWidgets import QWidget, QGridLayout, QPushButton, QSizePolicy, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QFont
from src.minesweeper.game import Game


class CellButton(QPushButton):
    """Custom button for a game cell."""
    
    left_clicked = Signal(int, int)
    right_clicked = Signal(int, int)
    hover_entered = Signal(int, int)
    hover_left = Signal(int, int)
    
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
        self.setMouseTracking(True)
    
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
    
    def enterEvent(self, event):
        """Handle mouse enter events."""
        self.hover_entered.emit(self.row, self.col)
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Handle mouse leave events."""
        self.hover_left.emit(self.row, self.col)
        super().leaveEvent(event)


class GameBoard(QWidget):
    """Widget for displaying and interacting with the game board."""
    
    game_won = Signal()
    game_lost = Signal()
    radar_status_changed = Signal()  # Signal für Radar-Status-Änderungen
    
    def __init__(self, game: Game):
        """
        Initialize game board widget.
        
        Args:
            game: Game instance
        """
        super().__init__()
        self.game = game
        self.buttons: list[list[CellButton]] = []
        self.radar_mode = False  # Radar-Auswahl-Modus
        self.scanner_mode = False  # Scanner-Auswahl-Modus
        self.radar_hover_cells = set()  # Zellen die im Radar-Hover-Bereich sind
        self.scanner_hover_cells = set()  # Zellen die im Scanner-Hover-Bereich sind
        self.tetris_hover_cells = set()  # Zellen im Tetris-Hover-Bereich
        # ✨ NEU: Training-Lock Callback
        self.is_training_callback = None  # Wird vom MainWindow gesetzt
        self._setup_ui()
        self._update_display()
    
    def _setup_ui(self):
        """Setup the UI layout."""
        # Hauptlayout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)
        self.setLayout(main_layout)
        
        # Power-Up Buttons oben
        powerup_button_layout = QHBoxLayout()
        powerup_button_layout.setContentsMargins(0, 0, 0, 0)
        powerup_button_layout.setSpacing(5)
        
        # Radar Button
        self.radar_button = QPushButton("📡 Radar (70P)")
        self.radar_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.radar_button.setMinimumHeight(40)
        self.radar_button.setCursor(Qt.PointingHandCursor)
        self.radar_button.clicked.connect(self._on_radar_button_clicked)
        self.radar_button.setEnabled(False)
        
        # Scanner Button
        self.scanner_button = QPushButton("🔍 Scanner (70P)")
        self.scanner_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.scanner_button.setMinimumHeight(40)
        self.scanner_button.setCursor(Qt.PointingHandCursor)
        self.scanner_button.clicked.connect(self._on_scanner_button_clicked)
        self.scanner_button.setEnabled(False)
        
        # Blitz Button
        self.blitz_button = QPushButton("⚡ Blitz (50P)")
        self.blitz_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.blitz_button.setMinimumHeight(40)
        self.blitz_button.setCursor(Qt.PointingHandCursor)
        self.blitz_button.clicked.connect(self._on_blitz_button_clicked)
        self.blitz_button.setEnabled(False)
        
        powerup_button_layout.addWidget(self.radar_button)
        powerup_button_layout.addWidget(self.scanner_button)
        powerup_button_layout.addWidget(self.blitz_button)
        main_layout.addLayout(powerup_button_layout)
        
        self._update_powerup_buttons()
        
        # Spielfeld-Container
        board_container = QWidget()
        self.board_layout = QGridLayout()
        # Explizit beide Spacing-Werte auf 0 setzen
        self.board_layout.setHorizontalSpacing(0)
        self.board_layout.setVerticalSpacing(0)
        self.board_layout.setSpacing(0)
        self.board_layout.setContentsMargins(0, 0, 0, 0)
        board_container.setLayout(self.board_layout)
        # Hintergrundfarbe auf Rasengrün setzen, damit keine dunklen Linien sichtbar sind
        board_container.setStyleSheet("""
            QWidget {
                background-color: #4a7c59;
                padding: 0px;
                margin: 0px;
            }
        """)
        main_layout.addWidget(board_container)
        
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
                button.hover_entered.connect(self._on_cell_hover_enter)
                button.hover_left.connect(self._on_cell_hover_leave)
                self.board_layout.addWidget(button, row, col)
                button_row.append(button)
            self.buttons.append(button_row)
        
        # Setze Stretch-Faktoren auf 1 für gleichmäßige Verteilung
        for row in range(height):
            self.board_layout.setRowStretch(row, 1)
        for col in range(width):
            self.board_layout.setColumnStretch(col, 1)
    
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
    
    def _on_radar_button_clicked(self):
        """Handle radar button click."""
        # ✨ NEU: Blockiere während Training
        if self.is_training_callback and self.is_training_callback():
            return
        
        if self.game.points < self.game.radar_cost:
            return
        
        # Deaktiviere andere Modi
        self.scanner_mode = False
        self.scanner_hover_cells.clear()
        
        # Toggle radar mode
        self.radar_mode = not self.radar_mode
        self._update_powerup_buttons()
        
    def _on_scanner_button_clicked(self):
        """Handle scanner button click."""
        # ✨ NEU: Blockiere während Training
        if self.is_training_callback and self.is_training_callback():
            return
        
        if self.game.points < self.game.scanner_cost:
            return
        
        # Deaktiviere andere Modi
        self.radar_mode = False
        self.radar_hover_cells.clear()
        
        # Toggle scanner mode
        self.scanner_mode = not self.scanner_mode
        self._update_powerup_buttons()
    
    def _on_blitz_button_clicked(self):
        """Handle blitz button click."""
        # ✨ NEU: Blockiere während Training
        if self.is_training_callback and self.is_training_callback():
            return
        
        if self.game.points < self.game.blitz_cost:
            return
        
        # Deaktiviere alle Modi
        self.radar_mode = False
        self.scanner_mode = False
        self.radar_hover_cells.clear()
        self.scanner_hover_cells.clear()
        
        # Führe Blitz aus
        fields_revealed = self.game.use_blitz()
        self._update_powerup_buttons()
        self._update_display()
        
        # Check game state after update
        if self.game.is_won():
            self.game_won.emit()
    
    def _update_powerup_buttons(self):
        """Update power-up button states and text."""
        points = self.game.points
        
        # Radar Button
        radar_can_afford = points >= self.game.radar_cost
        if self.radar_mode:
            self.radar_button.setText("📡 Radar-Modus")
            self.radar_button.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border: 2px solid #c0392b;
                    border-radius: 5px;
                    padding: 8px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #ec7063;
                }
            """)
        else:
            self.radar_button.setText(f"📡 Radar ({self.game.radar_cost}P)")
            self.radar_button.setEnabled(radar_can_afford)
            if radar_can_afford:
                self.radar_button.setStyleSheet("""
                    QPushButton {
                        background-color: #27ae60;
                        color: white;
                        border: 2px solid #229954;
                        border-radius: 5px;
                        padding: 8px;
                    }
                    QPushButton:hover {
                        background-color: #2ecc71;
                    }
                    QPushButton:pressed {
                        background-color: #1e8449;
                    }
                """)
            else:
                self.radar_button.setStyleSheet("""
                    QPushButton {
                        background-color: #34495e;
                        color: #95a5a6;
                        border: 2px solid #7f8c8d;
                        border-radius: 5px;
                        padding: 8px;
                    }
                """)
        
        # Scanner Button
        scanner_can_afford = points >= self.game.scanner_cost
        if self.scanner_mode:
            self.scanner_button.setText("🔍 Scanner-Modus")
            self.scanner_button.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border: 2px solid #c0392b;
                    border-radius: 5px;
                    padding: 8px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #ec7063;
                }
            """)
        else:
            self.scanner_button.setText(f"🔍 Scanner ({self.game.scanner_cost}P)")
            self.scanner_button.setEnabled(scanner_can_afford)
            if scanner_can_afford:
                self.scanner_button.setStyleSheet("""
                    QPushButton {
                        background-color: #3498db;
                        color: white;
                        border: 2px solid #2980b9;
                        border-radius: 5px;
                        padding: 8px;
                    }
                    QPushButton:hover {
                        background-color: #5dade2;
                    }
                    QPushButton:pressed {
                        background-color: #2471a3;
                    }
                """)
            else:
                self.scanner_button.setStyleSheet("""
                    QPushButton {
                        background-color: #34495e;
                        color: #95a5a6;
                        border: 2px solid #7f8c8d;
                        border-radius: 5px;
                        padding: 8px;
                    }
                """)
        
        # Blitz Button
        blitz_can_afford = points >= self.game.blitz_cost
        self.blitz_button.setText(f"⚡ Blitz ({self.game.blitz_cost}P)")
        self.blitz_button.setEnabled(blitz_can_afford)
        if blitz_can_afford:
            self.blitz_button.setStyleSheet("""
                QPushButton {
                    background-color: #f39c12;
                    color: white;
                    border: 2px solid #e67e22;
                    border-radius: 5px;
                    padding: 8px;
                }
                QPushButton:hover {
                    background-color: #f8b739;
                }
                QPushButton:pressed {
                    background-color: #d68910;
                }
            """)
        else:
            self.blitz_button.setStyleSheet("""
                QPushButton {
                    background-color: #34495e;
                    color: #95a5a6;
                    border: 2px solid #7f8c8d;
                    border-radius: 5px;
                    padding: 8px;
                }
            """)
        
        self.radar_status_changed.emit()
    
    def _on_left_click(self, row: int, col: int):
        """Handle left click on a cell."""
        # ✨ NEU: Blockiere Klicks während Training
        if self.is_training_callback and self.is_training_callback():
            return  # Ignoriere Klick während Training
        
        # Radar-Modus
        if self.radar_mode:
            if self.game.use_radar(row, col):
                self.radar_mode = False
                self._update_powerup_buttons()
                self._update_display()
                
                # Check game state after update
                if self.game.is_won():
                    self.game_won.emit()
            return
        
        # Scanner-Modus
        if self.scanner_mode:
            mine_count = self.game.use_scanner(row, col)
            if mine_count is not None:
                self.scanner_mode = False
                self._update_powerup_buttons()
                self._update_display()
            return
        
        # Tetris-Modus
        if self.game.tetris_active:
            if self.game.place_tetris_shape(row, col):
                self._update_powerup_buttons()
                self._update_display()
                
                # Check game state after update
                if self.game.is_won():
                    self.game_won.emit()
            return
        
        # Mystery-Feld Enthüllung
        if (row, col) in self.game.mystery_cells:
            cell = self.game.board.get_cell(row, col)
            if cell and cell.is_revealed():
                # Mystery-Feld wurde geklickt - versuche zu enthüllen
                if self.game.reveal_mystery_cell(row, col):
                    self._update_powerup_buttons()
                    self._update_display()
                return
        
        # Normaler Modus
        # Always update display after reveal attempt (even if False, to show mines on loss)
        result = self.game.reveal_cell(row, col)
        self._update_powerup_buttons()  # Update nach jedem Aufdecken
        self._update_display()
        
        # Check game state after update
        if self.game.is_won():
            self.game_won.emit()
        elif self.game.is_lost():
            self.game_lost.emit()
    
    def _on_right_click(self, row: int, col: int):
        """Handle right click on a cell (flag toggle)."""
        # ✨ NEU: Blockiere Klicks während Training
        if self.is_training_callback and self.is_training_callback():
            return  # Ignoriere Klick während Training
        
        if self.game.toggle_flag(row, col):
            self._update_display()
    
    def _on_cell_hover_enter(self, row: int, col: int):
        """Handle mouse hover enter on a cell."""
        if self.radar_mode:
            # Berechne 3x3-Bereich für Radar
            self.radar_hover_cells.clear()
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    r = row + dr
                    c = col + dc
                    if 0 <= r < self.game.height and 0 <= c < self.game.width:
                        self.radar_hover_cells.add((r, c))
            self._update_display()
        elif self.scanner_mode:
            # Berechne 3x3-Bereich für Scanner
            self.scanner_hover_cells.clear()
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    r = row + dr
                    c = col + dc
                    if 0 <= r < self.game.height and 0 <= c < self.game.width:
                        self.scanner_hover_cells.add((r, c))
            self._update_display()
        elif self.game.tetris_active and self.game.tetris_current_shape:
            # Berechne Tetris-Form-Bereich
            self.tetris_hover_cells.clear()
            for dr, dc in self.game.tetris_current_shape:
                r = row + dr
                c = col + dc
                if 0 <= r < self.game.height and 0 <= c < self.game.width:
                    self.tetris_hover_cells.add((r, c))
            self._update_display()
    
    def _on_cell_hover_leave(self, row: int, col: int):
        """Handle mouse hover leave on a cell."""
        if self.radar_mode:
            # Entferne Radar-Hover-Effekt
            self.radar_hover_cells.clear()
            self._update_display()
        elif self.scanner_mode:
            # Entferne Scanner-Hover-Effekt
            self.scanner_hover_cells.clear()
            self._update_display()
        elif self.game.tetris_active:
            # Entferne Tetris-Hover-Effekt
            self.tetris_hover_cells.clear()
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
                
                # Prüfe ob Zelle mit Radar gescannt wurde
                is_radar_scanned = (row, col) in self.game.radar_scanned_cells
                # Prüfe ob Zelle im Radar-Hover-Bereich ist
                is_radar_hover = (row, col) in self.radar_hover_cells
                # Prüfe ob Zelle im Scanner-Hover-Bereich ist
                is_scanner_hover = (row, col) in self.scanner_hover_cells
                # Prüfe ob Scanner-Ergebnis vorhanden
                is_scanner_center = (row, col) in self.game.scanner_result_cells
                # Prüfe ob Speed-Feld
                is_speed_field = (row, col) in self.game.speed_cells
                # Prüfe ob Tetris-Feld
                is_tetris_field = (row, col) in self.game.tetris_cells
                # Prüfe ob Tetris-Hover
                is_tetris_hover = (row, col) in self.tetris_hover_cells
                
                # Schachbrettmuster für Rasen-Effekt (Hell- und Dunkelgrün)
                is_dark_square = (row + col) % 2 == 0
                grass_dark = "#4a7c59"   # Dunkelgrün
                grass_light = "#5d9c6c"  # Hellgrün
                
                # Scanner-Ergebnis-Anzeige - zeigt zusätzliche Info, blockiert aber nicht
                scanner_overlay = ""
                if is_scanner_center:
                    mine_count = self.game.scanner_result_cells[(row, col)]
                    scanner_overlay = f" 🔍{mine_count}"
                
                if cell.is_revealed():
                    button.setEnabled(False)
                    if cell.is_mine:
                        button.setText("💣" + scanner_overlay)
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
                        # Prüfe ob Mystery-Feld
                        is_mystery = (row, col) in self.game.mystery_cells
                        
                        if is_mystery:
                            # Zeige Fragezeichen statt Zahl
                            button.setText("❓" + scanner_overlay)
                            button.setEnabled(True)  # Klickbar für Enthüllung
                            # Aufhellung bei Hover
                            if is_radar_hover or is_scanner_hover:
                                bg_color = "#c084fc"  # Lila-Pink beim Hover
                            elif is_radar_scanned:
                                bg_color = "#a89a50"
                            else:
                                bg_color = "#9333ea"  # Lila für Mystery
                            border_style = "border: 2px solid #7c3aed;" if not (is_radar_hover or is_scanner_hover) else "border: 2px solid #c084fc;"
                            button.setStyleSheet(f"""
                                QPushButton {{
                                    background-color: {bg_color};
                                    color: white;
                                    font-weight: bold;
                                    font-size: 20px;
                                    {border_style}
                                    outline: none;
                                    padding: 0px;
                                    margin: 0px;
                                }}
                                QPushButton:hover {{
                                    background-color: #c084fc;
                                    outline: none;
                                }}
                            """)
                        else:
                            # Normale Zahlenanzeige
                            text_display = str(cell.adjacent_mines) + scanner_overlay
                            if is_speed_field:
                                text_display = "⚡" + text_display  # Speed-Symbol hinzufügen
                            if is_tetris_field:
                                text_display = "🎮" + text_display  # Tetris-Symbol hinzufügen
                            button.setText(text_display)
                            
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
                            
                            # Speed-Feld Hintergrund
                            if is_speed_field:
                                # Gelb-Orange für Speed-Felder
                                bg_color = "#f39c12"
                                border_style = "border: 3px solid #e67e22;"
                            # Tetris-Feld Hintergrund
                            elif is_tetris_field:
                                # Lila für Tetris-Felder
                                bg_color = "#9b59b6"
                                border_style = "border: 3px solid #8e44ad;"
                            # Tetris-Hover
                            elif is_tetris_hover:
                                # Hell-Lila für Tetris-Hover
                                bg_color = "#bb7dd1"
                                border_style = "border: 3px solid #9b59b6;"
                            # Aufhellung bei Radar-Scan/Scanner-Hover oder Hover
                            elif is_radar_hover or is_scanner_hover:
                                bg_color = "#b8b060"  # Noch heller beim Hover
                                border_style = "border: 1px solid rgba(77, 208, 225, 0.4);" if is_radar_scanned else "border: none;"
                            elif is_radar_scanned:
                                bg_color = "#a89a50"
                                border_style = "border: 1px solid rgba(77, 208, 225, 0.4);"
                            else:
                                bg_color = "#8b6914"
                                border_style = "border: none;"
                            
                            button.setStyleSheet(f"""
                                QPushButton {{
                                    background-color: {bg_color};
                                    color: {color};
                                    font-weight: bold;
                                    {border_style}
                                    outline: none;
                                    padding: 0px;
                                    margin: 0px;
                                }}
                            """)
                    else:
                        text_display = "" + scanner_overlay
                        if is_speed_field:
                            text_display = "⚡" + text_display
                        if is_tetris_field:
                            text_display = "🎮" + text_display
                        button.setText(text_display)
                        # Aufhellung bei Radar-Scan/Scanner-Hover oder Hover
                        if is_radar_hover or is_scanner_hover:
                            bg_color = "#d4b88e"  # Noch heller beim Hover
                        elif is_radar_scanned:
                            bg_color = "#c4a87e"
                        else:
                            bg_color = "#a0826d"
                        border_style = "border: 1px solid rgba(77, 208, 225, 0.4);" if is_radar_scanned else "border: none;"
                        button.setStyleSheet(f"""
                            QPushButton {{
                                background-color: {bg_color};
                                {border_style}
                                outline: none;
                                padding: 0px;
                                margin: 0px;
                            }}
                        """)
                elif cell.is_flagged():
                    # Flagge auf Rasen
                    if is_radar_hover or is_scanner_hover:
                        grass_color = "#7dd68d"  # Heller beim Hover
                    else:
                        grass_color = grass_dark if is_dark_square else grass_light
                    # Radar-Highlighting für geflaggte Felder
                    border_style = "border: 1px solid rgba(77, 208, 225, 0.4);" if is_radar_scanned else "border: none;"
                    button.setText("🚩" + scanner_overlay)
                    button.setStyleSheet(f"""
                        QPushButton {{
                            background-color: {grass_color};
                            color: white;
                            font-weight: bold;
                            {border_style}
                            outline: none;
                            padding: 0px;
                            margin: 0px;
                        }}
                    """)
                    button.setEnabled(True)
                else:
                    # Verdecktes Feld - Rasen mit Schachbrettmuster
                    # Prüfe ob Mine unter Radar-Scan
                    if is_radar_scanned and cell.is_mine:
                        # Mine wurde vom Radar erkannt - zeige Warnung
                        bg_color = "#f39c12" if (is_radar_hover or is_scanner_hover) else "#e67e22"
                        button.setText("⚠️" + scanner_overlay)
                        button.setStyleSheet(f"""
                            QPushButton {{
                                background-color: {bg_color};
                                color: white;
                                font-weight: bold;
                                border: 1px solid rgba(211, 84, 0, 0.6);
                                outline: none;
                                padding: 0px;
                                margin: 0px;
                            }}
                            QPushButton:hover {{
                                background-color: #f39c12;
                                outline: none;
                            }}
                        """)
                    else:
                        grass_color = grass_dark if is_dark_square else grass_light
                        grass_hover = "#6bb77b" if is_dark_square else "#7dd68d"
                        # Radar-Highlighting für verdeckte Felder
                        border_style_radar = "border: 1px solid rgba(77, 208, 225, 0.4);" if is_radar_scanned else "border: none;"
                        # Scanner-Highlighting (blaue Umrandung)
                        border_style_scanner = "border: 2px solid rgba(52, 152, 219, 0.6);" if is_scanner_hover else ""
                        border_style = border_style_scanner if is_scanner_hover else border_style_radar
                        
                        # Hintergrundfarbe abhängig von Hover-Modi
                        if is_tetris_hover:
                            radar_bg = "#bb7dd1"  # Hell-Lila beim Tetris-Hover
                        elif is_scanner_hover:
                            radar_bg = "#7dc9e8"  # Blaugrün beim Scanner-Hover
                        elif is_radar_hover:
                            radar_bg = "#8dd79d"  # Hellgrün beim Radar-Hover
                        elif is_radar_scanned:
                            radar_bg = "#7dd68d"
                        else:
                            radar_bg = grass_color
                        button.setText("" + scanner_overlay)
                        button.setStyleSheet(f"""
                            QPushButton {{
                                background-color: {radar_bg};
                                {border_style}
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
        self.radar_mode = False
        self.scanner_mode = False
        self.radar_hover_cells = set()
        self.scanner_hover_cells = set()
        self.tetris_hover_cells = set()
        
        if size_changed:
            self._create_buttons()
        
        self._update_powerup_buttons()
        self._update_display()
        # Update font sizes after reset
        self._update_font_sizes()


