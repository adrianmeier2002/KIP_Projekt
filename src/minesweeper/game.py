"""Game-Klasse für die Minesweeper-Spiellogik."""

from typing import List, Tuple, Optional
from src.minesweeper.board import Board
from src.utils.constants import (
    BOARD_WIDTH, BOARD_HEIGHT,
    get_mine_count
)


class GameState:
    """Spielzustands-Enumeration."""
    PLAYING = "playing"
    WON = "won"
    LOST = "lost"


class Game:
    """Verwaltet die Minesweeper-Spiellogik."""
    
    def __init__(self, difficulty: str = "medium", width: int = BOARD_WIDTH, height: int = BOARD_HEIGHT):
        """
        Initialisiert ein neues Spiel.
        
        Args:
            difficulty: Schwierigkeitsgrad ("easy", "medium" oder "hard")
            width: Breite des Spielfelds
            height: Höhe des Spielfelds
        """
        self.difficulty = difficulty
        self.width = width
        self.height = height
        self.num_mines = get_mine_count(width, height, difficulty)
        self.board = Board(width, height, self.num_mines)
        self.state = GameState.PLAYING
        self.first_click = True
        self.revealed_count = 0
        self.flagged_count = 0
        self.total_safe_cells = width * height - self.num_mines
    
    def _get_mine_count(self, difficulty: str) -> int:
        """Gibt die Minenanzahl für den Schwierigkeitsgrad zurück."""
        return get_mine_count(self.width, self.height, difficulty)
    
    def reveal_cell(self, row: int, col: int) -> bool:
        """
        Deckt eine Zelle auf und behandelt die Spiellogik.
        
        Args:
            row: Zeilenindex
            col: Spaltenindex
            
        Returns:
            True wenn die Zelle aufgedeckt wurde, False sonst
        """
        if self.state != GameState.PLAYING:
            return False
        
        cell = self.board.get_cell(row, col)
        if not cell or cell.is_revealed() or cell.is_flagged():
            return False
        
        # Platziere Minen beim ersten Klick (ausgenommen die geklickte Zelle)
        if self.first_click:
            self.board.place_mines(row, col)
            self.first_click = False
        
        # Prüfe ob es eine Mine ist
        if cell.is_mine:
            # Decke die geklickte Mine auf
            cell.reveal()
            self.state = GameState.LOST
            self.board.reveal_all_mines()
            return True  # Return True damit GUI aktualisiert wird
        
        # Decke Zelle auf
        if cell.reveal():
            self.revealed_count += 1
            
            # Auto-Aufdeckung der Nachbarn wenn keine benachbarten Minen
            if cell.adjacent_mines == 0:
                self._auto_reveal_safe_neighbors(row, col)
            
            # Prüfe Gewinnbedingung
            if self.revealed_count >= self.total_safe_cells:
                self.state = GameState.WON
            
            return True
        
        return False
    
    def _auto_reveal_safe_neighbors(self, row: int, col: int):
        """
        Deckt automatisch sichere Nachbarn einer Zelle ohne benachbarte Minen auf.
        
        Args:
            row: Zeilenindex
            col: Spaltenindex
        """
        # Starte mit Nachbarn der aktuellen Zelle (die bereits aufgedeckt ist)
        initial_cell = self.board.get_cell(row, col)
        if not initial_cell or initial_cell.adjacent_mines != 0:
            return
        
        # Starte Prüfung von Nachbarn, nicht der aktuellen Zelle
        cells_to_check = []
        checked = set()
        checked.add((row, col))  # Markiere initiale Zelle als geprüft
        
        # Füge alle Nachbarn der initialen Zelle hinzu
        neighbors = self.board.get_neighbors(row, col)
        for neighbor in neighbors:
            if neighbor.is_hidden() and not neighbor.is_flagged():
                if (neighbor.row, neighbor.col) not in checked:
                    cells_to_check.append((neighbor.row, neighbor.col))
        
        while cells_to_check:
            r, c = cells_to_check.pop(0)
            if (r, c) in checked:
                continue
            checked.add((r, c))
            
            cell = self.board.get_cell(r, c)
            if not cell or cell.is_revealed() or cell.is_flagged():
                continue
            
            # Don't reveal mines
            if cell.is_mine:
                continue
            
            # Decke aktuelle Zelle auf
            if cell.reveal():
                self.revealed_count += 1
            
            # Wenn keine benachbarten Minen, füge Nachbarn zur Prüfung hinzu
            if cell.adjacent_mines == 0:
                neighbors = self.board.get_neighbors(r, c)
                for neighbor in neighbors:
                    if neighbor.is_hidden() and not neighbor.is_flagged():
                        if (neighbor.row, neighbor.col) not in checked:
                            cells_to_check.append((neighbor.row, neighbor.col))
    
    def toggle_flag(self, row: int, col: int) -> bool:
        """
        Schaltet die Flagge einer Zelle um.
        
        Args:
            row: Zeilenindex
            col: Spaltenindex
            
        Returns:
            True wenn die Flagge umgeschaltet wurde, False sonst
        """
        if self.state != GameState.PLAYING:
            return False
        
        cell = self.board.get_cell(row, col)
        if not cell or cell.is_revealed():
            return False
        
        if cell.flag():
            if cell.is_flagged():
                self.flagged_count += 1
            else:
                self.flagged_count -= 1
            return True
        
        return False
    
    def get_remaining_mines(self) -> int:
        """Gibt die verbleibende Minenanzahl zurück (gesamt - geflaggt)."""
        return self.num_mines - self.flagged_count
    
    def is_game_over(self) -> bool:
        """Prüft, ob das Spiel beendet ist (gewonnen oder verloren)."""
        return self.state in [GameState.WON, GameState.LOST]
    
    def is_won(self) -> bool:
        """Prüft, ob das Spiel gewonnen wurde."""
        return self.state == GameState.WON
    
    def is_lost(self) -> bool:
        """Prüft, ob das Spiel verloren wurde."""
        return self.state == GameState.LOST
    
    def new_game(self, difficulty: Optional[str] = None, width: Optional[int] = None, height: Optional[int] = None):
        """
        Startet ein neues Spiel.
        
        Args:
            difficulty: Optionaler Schwierigkeitsgrad
            width: Optionale Breite des Spielfelds
            height: Optionale Höhe des Spielfelds
        """
        if difficulty:
            self.difficulty = difficulty
        if width:
            self.width = width
        if height:
            self.height = height
        
        self.num_mines = self._get_mine_count(self.difficulty)
        self.board = Board(self.width, self.height, self.num_mines)
        self.state = GameState.PLAYING
        self.first_click = True
        self.revealed_count = 0
        self.flagged_count = 0
        self.total_safe_cells = self.width * self.height - self.num_mines


