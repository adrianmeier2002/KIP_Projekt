"""Board-Klasse für das Minesweeper-Spiel."""

import random
from typing import List, Tuple, Set
from src.minesweeper.cell import Cell
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class Board:
    """Verwaltet das Minesweeper-Spielfeld."""
    
    def __init__(self, width: int = BOARD_WIDTH, height: int = BOARD_HEIGHT, num_mines: int = 0):
        """
        Initialisiert das Spielfeld.
        
        Args:
            width: Breite des Spielfelds
            height: Höhe des Spielfelds
            num_mines: Anzahl der zu platzierenden Minen
        """
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.cells: List[List[Cell]] = []
        self.mine_positions: Set[Tuple[int, int]] = set()
        self._initialize_board()
    
    def _initialize_board(self):
        """Initialisiert ein leeres Spielfeld mit Zellen."""
        self.cells = [
            [Cell(row, col) for col in range(self.width)]
            for row in range(self.height)
        ]
    
    def place_mines(self, first_row: int, first_col: int):
        """
        Platziert Minen zufällig auf dem Spielfeld, ausgenommen die erste geklickte Zelle
        und alle ihre Nachbarn. Dies garantiert, dass der erste Klick ein Autosolve-Feld ist.
        
        Args:
            first_row: Zeile der ersten geklickten Zelle
            first_col: Spalte der ersten geklickten Zelle
        """
        # Berechne ausgeschlossene Positionen (erste Zelle + alle Nachbarn)
        excluded_positions = {(first_row, first_col)}
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1),
                       (0, -1),           (0, 1),
                       (1, -1),  (1, 0),  (1, 1)]:
            nr, nc = first_row + dr, first_col + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                excluded_positions.add((nr, nc))
        
        available_positions = [
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
            if (row, col) not in excluded_positions
        ]
        
        mine_positions = random.sample(available_positions, min(self.num_mines, len(available_positions)))
        self.mine_positions = set(mine_positions)
        
        for row, col in mine_positions:
            self.cells[row][col].set_mine()
        
        # Berechne die Anzahl benachbarter Minen für alle Zellen
        self._calculate_adjacent_mines()
    
    def _calculate_adjacent_mines(self):
        """Berechnet die Anzahl benachbarter Minen für jede Zelle."""
        for row in range(self.height):
            for col in range(self.width):
                if not self.cells[row][col].is_mine:
                    count = 0
                    for dr, dc in [(-1, -1), (-1, 0), (-1, 1),
                                   (0, -1),           (0, 1),
                                   (1, -1),  (1, 0),  (1, 1)]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < self.height and 0 <= nc < self.width:
                            if self.cells[nr][nc].is_mine:
                                count += 1
                    self.cells[row][col].set_adjacent_mines(count)
    
    def get_cell(self, row: int, col: int) -> Cell:
        """Gibt die Zelle an der angegebenen Position zurück."""
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.cells[row][col]
        return None
    
    def get_neighbors(self, row: int, col: int) -> List[Cell]:
        """Gibt eine Liste der benachbarten Zellen zurück."""
        neighbors = []
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1),
                       (0, -1),           (0, 1),
                       (1, -1),  (1, 0),  (1, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                neighbors.append(self.cells[nr][nc])
        return neighbors
    
    def reveal_all_mines(self):
        """Deckt alle Minenpositionen auf (bei Spielende)."""
        for row, col in self.mine_positions:
            self.cells[row][col].reveal()


