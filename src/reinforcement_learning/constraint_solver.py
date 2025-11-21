"""Constraint Solver für Minesweeper.

Dieser Solver nutzt Minesweeper-Logik um 100% sichere Züge zu finden.
Funktioniert OHNE Flags durch Constraint Satisfaction!
"""

from typing import List, Tuple, Set, Optional
import numpy as np


class MinesweeperSolver:
    """Regelbasierter Solver zum Finden sicherer Zellen."""
    
    def __init__(self, game):
        """
        Initialisiert den Solver.
        
        Args:
            game: Minesweeper-Spiel-Instanz
        """
        self.game = game
        self.board = game.board
        self.width = game.board.width
        self.height = game.board.height
        self._neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                                  (0, -1),           (0, 1),
                                  (1, -1),  (1, 0),  (1, 1)]
    
    def get_safe_moves(self) -> List[Tuple[int, int]]:
        """
        Findet alle Zellen die 100% sicher aufzudecken sind.
        
        Strategien:
        1. Einfaches Pattern: Flags erfüllt → Rest ist sicher
        2. Cross-Constraint: Mehrere Zellen teilen Nachbarn
        3. Subtraktions-Pattern: Zelle A - Zelle B Constraints
        
        WICHTIG: Ignoriert Mystery-Zellen (Herausforderungs-Feature)!
        WICHTIG: Blockiert bei Tetris-aktiv Modus (Spezial-Platzierung erforderlich)!
        
        Returns:
            Liste von (row, col) Tupeln für sichere Zellen (leer bei Tetris-Modus)
        """
        # NEU: Wenn Tetris aktiv ist, darf Solver keine normalen Züge empfehlen
        if self.game.tetris_active:
            return []
        
        safe_cells = set()
        mine_cells = self._find_certain_mines()
        
        # Pass 1: Simple patterns with known mines
        for row in range(self.height):
            for col in range(self.width):
                cell = self.board.get_cell(row, col)
                
                # Nur aufgedeckte Zellen mit Zahlen prüfen
                if not cell.is_revealed() or cell.is_mine:
                    continue
                
                # NEU: Mystery-Zellen ignorieren (ihre Zahl ist für Spieler versteckt!)
                if (row, col) in self.game.mystery_cells:
                    continue
                
                # Sammle Nachbar-Informationen
                neighbors = self._get_neighbors(row, col)
                flagged = [n for n in neighbors if self.board.get_cell(n[0], n[1]).is_flagged()]
                hidden = [n for n in neighbors if self.board.get_cell(n[0], n[1]).is_hidden()]
                
                if len(hidden) == 0:
                    continue
                
                # Zähle bekannte Minen (geflaggt oder erkannt)
                known_mines = [n for n in hidden if n in mine_cells]
                total_known_mines = len(flagged) + len(known_mines)
                
                # Pattern: Alle Minen bekannt → Rest ist sicher
                if total_known_mines == cell.adjacent_mines:
                    for h in hidden:
                        if h not in mine_cells:
                            safe_cells.add(h)
        
        # Pass 2: Cross-Constraint-Analyse
        safe_cells.update(self._find_safe_by_constraint_satisfaction())
        
        # Entferne bereits aufgedeckte, Minen oder geflaggte Zellen
        result = []
        for r, c in safe_cells:
            cell = self.board.get_cell(r, c)
            if not cell.is_revealed() and not cell.is_flagged() and (r, c) not in mine_cells:
                result.append((r, c))
        
        return result
    
    def _find_certain_mines(self) -> Set[Tuple[int, int]]:
        """
        Find cells that are 100% certainly mines (even without flags).
        
        Pattern: If cell shows N and has exactly N hidden neighbors
                → All hidden neighbors are mines
        
        WICHTIG: Ignoriert Mystery-Zellen (Herausforderungs-Feature)!
        
        Returns:
            Set of (row, col) coordinates that are certain mines
        """
        certain_mines = set()
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.board.get_cell(row, col)
                
                if not cell.is_revealed() or cell.is_mine:
                    continue
                
                # NEU: Mystery-Zellen ignorieren (ihre Zahl ist für Spieler versteckt!)
                if (row, col) in self.game.mystery_cells:
                    continue
                
                neighbors = self._get_neighbors(row, col)
                flagged = [n for n in neighbors if self.board.get_cell(n[0], n[1]).is_flagged()]
                hidden = [n for n in neighbors if self.board.get_cell(n[0], n[1]).is_hidden()]
                
                # If remaining hidden cells = remaining mines → all are mines
                remaining_mines = cell.adjacent_mines - len(flagged)
                if remaining_mines > 0 and len(hidden) == remaining_mines:
                    certain_mines.update(hidden)
        
        return certain_mines
    
    def _find_safe_by_constraint_satisfaction(self) -> Set[Tuple[int, int]]:
        """
        IMPROVED: Find safe cells using advanced constraint satisfaction.
        
        Strategies:
        1. Subset analysis: If constraint A ⊆ B and same mines → difference is safe
        2. Difference analysis: If A has X mines in N cells, B has Y mines in overlapping cells
        3. Iterative propagation: Repeat until no new information
        
        Returns:
            Set of (row, col) coordinates that are safe
        """
        safe_cells = set()
        mine_cells = set()
        
        # OPTIMIZED: Iterative constraint propagation (up to 2 passes - balance between speed and accuracy)
        for iteration in range(2):
            new_info = False
            
            # Collect all constraints
            constraints = self._build_constraints()
            
            if not constraints:
                break
            
            # Pass 1: Simple subset relationships
            for i, c1 in enumerate(constraints):
                for c2 in constraints[i+1:]:
                    # Skip if no overlap
                    if not c1['hidden'] or not c2['hidden']:
                        continue
                    
                    # Case 1: c1 ⊆ c2
                    if c1['hidden'].issubset(c2['hidden']):
                        diff = c2['hidden'] - c1['hidden']
                        diff_mines = c2['mines'] - c1['mines']
                        
                        if diff_mines == 0:  # No mines in difference → safe
                            if diff:
                                safe_cells.update(diff)
                                new_info = True
                        elif diff_mines == len(diff) and len(diff) > 0:  # All difference are mines
                            mine_cells.update(diff)
                            new_info = True
                    
                    # Case 2: c2 ⊆ c1
                    elif c2['hidden'].issubset(c1['hidden']):
                        diff = c1['hidden'] - c2['hidden']
                        diff_mines = c1['mines'] - c2['mines']
                        
                        if diff_mines == 0:  # No mines in difference → safe
                            if diff:
                                safe_cells.update(diff)
                                new_info = True
                        elif diff_mines == len(diff) and len(diff) > 0:  # All difference are mines
                            mine_cells.update(diff)
                            new_info = True
                    
                    # Case 3: Overlapping constraints
                    else:
                        overlap = c1['hidden'] & c2['hidden']
                        if overlap:
                            only_c1 = c1['hidden'] - c2['hidden']
                            only_c2 = c2['hidden'] - c1['hidden']
                            
                            # Advanced logic: check if constraints force conclusions
                            # If c1 needs all its mines in overlap, only_c1 is safe
                            if c1['mines'] <= len(overlap) and only_c1:
                                max_mines_in_overlap_from_c1 = c1['mines']
                                min_mines_in_overlap_from_c2 = max(0, c2['mines'] - len(only_c2))
                                
                                if max_mines_in_overlap_from_c1 == min_mines_in_overlap_from_c2 == len(overlap):
                                    # All overlap are mines
                                    if only_c1 and c1['mines'] == len(overlap):
                                        safe_cells.update(only_c1)
                                        new_info = True
                                    if only_c2 and c2['mines'] == len(overlap):
                                        safe_cells.update(only_c2)
                                        new_info = True
            
            # Update known mines and safe cells for next iteration
            self._update_virtual_knowledge(safe_cells, mine_cells)
            
            if not new_info:
                break
        
        return safe_cells
    
    def _build_constraints(self) -> List[dict]:
        """
        Build list of constraints from revealed cells.
        
        WICHTIG: Ignoriert Mystery-Zellen (Herausforderungs-Feature)!
        """
        constraints = []
        mine_cells = self._find_certain_mines()
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.board.get_cell(row, col)
                
                if not cell.is_revealed() or cell.is_mine:
                    continue
                
                # NEU: Mystery-Zellen ignorieren (ihre Zahl ist für Spieler versteckt!)
                if (row, col) in self.game.mystery_cells:
                    continue
                
                neighbors = self._get_neighbors(row, col)
                flagged = [n for n in neighbors if self.board.get_cell(n[0], n[1]).is_flagged()]
                hidden = [n for n in neighbors if self.board.get_cell(n[0], n[1]).is_hidden()]
                
                # Count certain mines
                known_mines = [n for n in hidden if n in mine_cells]
                remaining_hidden = [n for n in hidden if n not in mine_cells]
                
                if len(remaining_hidden) > 0:
                    remaining_mines = cell.adjacent_mines - len(flagged) - len(known_mines)
                    constraints.append({
                        'position': (row, col),
                        'hidden': set(remaining_hidden),
                        'mines': remaining_mines
                    })
        
        return constraints
    
    def _update_virtual_knowledge(self, safe_cells: Set, mine_cells: Set):
        """Virtual update for iterative constraint propagation (doesn't modify board)."""
        # This is a placeholder for tracking temporary knowledge within iterations
        # The actual board state is not modified
        pass
    
    def get_mine_cells(self) -> List[Tuple[int, int]]:
        """
        Find all cells that are 100% mines.
        
        Logic:
        - If cell shows N and has exactly N hidden neighbors → all are mines
        
        Returns:
            List of (row, col) tuples for mine cells
        """
        return list(self._find_certain_mines())
    
    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get list of neighbor coordinates."""
        neighbors = []
        for dr, dc in self._neighbor_offsets:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                neighbors.append((nr, nc))
        return neighbors
    
    def get_mine_probabilities(self) -> dict:
        """
        OPTIMIZED: Calculate mine probability for frontier cells (simplified for speed).
        
        WICHTIG: Ignoriert Mystery-Zellen (Herausforderungs-Feature)!
        
        Returns:
            Dict mapping (row, col) → probability (0.0 to 1.0)
        """
        probabilities = {}
        
        # OPTIMIZED: Simplified probability calculation (much faster)
        for row in range(self.height):
            for col in range(self.width):
                cell = self.board.get_cell(row, col)
                
                if not cell.is_revealed() or cell.is_mine:
                    continue
                
                # NEU: Mystery-Zellen ignorieren (ihre Zahl ist für Spieler versteckt!)
                if (row, col) in self.game.mystery_cells:
                    continue
                
                neighbors = self._get_neighbors(row, col)
                flagged = [n for n in neighbors if self.board.get_cell(n[0], n[1]).is_flagged()]
                hidden = [n for n in neighbors if self.board.get_cell(n[0], n[1]).is_hidden()]
                
                if len(hidden) > 0:
                    remaining_mines = cell.adjacent_mines - len(flagged)
                    prob = remaining_mines / len(hidden)
                    
                    # Assign probability to each hidden neighbor
                    for h_pos in hidden:
                        # If cell already has a probability, average them
                        if h_pos in probabilities:
                            probabilities[h_pos] = (probabilities[h_pos] + prob) / 2
                        else:
                            probabilities[h_pos] = prob
        
        return probabilities
    
    def get_best_guess(self, valid_actions: np.ndarray) -> Optional[int]:
        """
        IMPROVED: Find best guess using mine probability calculation.
        
        Strategy:
        1. Calculate mine probabilities for all frontier cells
        2. Choose cell with LOWEST mine probability
        3. Fallback to heuristic if no frontier cells
        
        WICHTIG: Blockiert bei Tetris-aktiv Modus!
        
        Args:
            valid_actions: Boolean array of valid actions
            
        Returns:
            Action index or None
        """
        # NEU: Wenn Tetris aktiv ist, keine Guesses empfehlen
        if self.game.tetris_active:
            return None
        
        if not np.any(valid_actions):
            return None
        
        # Get mine probabilities
        probabilities = self.get_mine_probabilities()
        
        if probabilities:
            # Find valid action with lowest mine probability
            best_prob = float('inf')
            best_action = None
            
            valid_indices = np.where(valid_actions)[0]
            for action in valid_indices:
                row = action // self.width
                col = action % self.width
                
                if (row, col) in probabilities:
                    prob = probabilities[(row, col)]
                    if prob < best_prob:
                        best_prob = prob
                        best_action = action
            
            if best_action is not None:
                return best_action
        
        # Fallback: Heuristic-based guess
        valid_indices = np.where(valid_actions)[0]
        best_score = -float('inf')
        best_action = None
        
        for action in valid_indices:
            row = action // self.width
            col = action % self.width
            
            cell = self.board.get_cell(row, col)
            if not cell or not cell.is_hidden() or cell.is_flagged():
                continue
            
            # Score based on revealed neighbors
            neighbors = self._get_neighbors(row, col)
            revealed_count = sum(1 for r, c in neighbors 
                               if self.board.get_cell(r, c).is_revealed())
            
            # Prefer frontier cells (cells with revealed neighbors)
            if revealed_count > 0:
                # Sum of numbers in revealed neighbors (lower is better)
                danger_sum = sum(self.board.get_cell(r, c).adjacent_mines 
                               for r, c in neighbors 
                               if self.board.get_cell(r, c).is_revealed() 
                               and not self.board.get_cell(r, c).is_mine)
                
                # Score: favor many revealed neighbors, low danger
                score = revealed_count * 10 - danger_sum
            else:
                # Non-frontier cell: lower priority but still valid
                score = -100
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action

