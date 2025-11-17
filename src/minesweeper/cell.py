"""Cell-Klasse für das Minesweeper-Spiel."""

from src.utils.constants import (
    CELL_HIDDEN,
    CELL_REVEALED,
    CELL_FLAGGED,
    CELL_MINE
)


class Cell:
    """Repräsentiert eine einzelne Zelle im Minesweeper-Spielfeld."""
    
    def __init__(self, row: int, col: int):
        """
        Initialisiert eine Zelle.
        
        Args:
            row: Zeilenindex der Zelle
            col: Spaltenindex der Zelle
        """
        self.row = row
        self.col = col
        self.is_mine = False
        self.adjacent_mines = 0
        self.state = CELL_HIDDEN
    
    def set_mine(self):
        """Markiert diese Zelle als Mine."""
        self.is_mine = True
    
    def set_adjacent_mines(self, count: int):
        """Setzt die Anzahl der benachbarten Minen."""
        self.adjacent_mines = count
    
    def reveal(self):
        """Deckt die Zelle auf."""
        if self.state == CELL_HIDDEN:
            self.state = CELL_REVEALED
            return True
        return False
    
    def flag(self):
        """Schaltet die Flagge der Zelle um."""
        if self.state == CELL_HIDDEN:
            self.state = CELL_FLAGGED
            return True
        elif self.state == CELL_FLAGGED:
            self.state = CELL_HIDDEN
            return True
        return False
    
    def is_revealed(self) -> bool:
        """Prüft, ob die Zelle aufgedeckt ist."""
        return self.state == CELL_REVEALED
    
    def is_flagged(self) -> bool:
        """Prüft, ob die Zelle geflaggt ist."""
        return self.state == CELL_FLAGGED
    
    def is_hidden(self) -> bool:
        """Prüft, ob die Zelle verdeckt ist."""
        return self.state == CELL_HIDDEN
    
    def __repr__(self):
        return f"Cell({self.row}, {self.col}, mine={self.is_mine}, state={self.state})"


