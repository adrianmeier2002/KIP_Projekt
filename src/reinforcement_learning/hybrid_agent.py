"""Hybrid Agent: Constraint Solver + Deep Q-Network.

Dieser Agent kombiniert regelbasiertes Constraint Solving mit RL:
1. Zuerst: Nutze Constraint Solver für 100% sichere Züge
2. Wenn keine sicheren Züge: Nutze DQN für intelligente Vermutungen
3. Lerne aus Vermutungs-Situationen um sich zu verbessern
"""

import random
import numpy as np
from typing import Optional, Tuple
from src.reinforcement_learning.dqn_agent import DQNAgent
from src.reinforcement_learning.constraint_solver import MinesweeperSolver


class HybridAgent(DQNAgent):
    """
    Hybrid Agent der Constraint Solving mit RL kombiniert.
    
    Strategie:
    - Nutzt Constraint Solver für garantiert sichere Züge
    - Nutzt DQN nur wenn Vermutungen nötig sind
    - Vermeidet dadurch vermeidbare Fehler
    """
    
    def __init__(self, *args, use_solver: bool = True, **kwargs):
        """
        Initialisiert den Hybrid Agent.
        
        Args:
            use_solver: Ob Constraint Solver genutzt wird (True = Hybrid, False = Pure RL)
            *args, **kwargs: Werden an DQNAgent weitergegeben
        """
        super().__init__(*args, **kwargs)
        self.use_solver = use_solver
        self.stats = {
            'solver_moves': 0,
            'rl_moves': 0,
            'total_moves': 0
        }
    
    def select_action(
        self, 
        state: np.ndarray, 
        valid_actions: Optional[np.ndarray] = None,
        game = None
    ) -> int:
        """
        Wählt Aktion mit Hybrid-Strategie + intelligenter Joker-Nutzung.
        
        Strategie:
        1. ✨ NEU: Prüfe ob Blitz/Joker sinnvoll sind
        2. Versuche zuerst Constraint Solver
        3. Wenn kein sicherer Zug: Nutze RL für Vermutung
        4. Während Training: Manchmal Solver überspringen für Exploration
        
        Args:
            state: Aktueller Zustand
            valid_actions: Boolean-Array valider Aktionen
            game: Spiel-Instanz (benötigt für Solver)
            
        Returns:
            Gewählter Aktionsindex
        """
        self.stats['total_moves'] += 1
        
        # Option to use pure RL (for comparison)
        if not self.use_solver:
            self.stats['rl_moves'] += 1
            return super().select_action(state, valid_actions)
        
        # ✨ NEU: Intelligente Joker-Nutzung (wenn Training abgeschlossen / low epsilon)
        if game is not None and self.epsilon < 0.2:
            joker_used = self._try_use_powerups(game)
            if joker_used:
                # Joker wurde genutzt, mache trotzdem normalen Zug
                pass
        
        # Balance zwischen Solver und RL
        # Während Training: Manchmal Solver überspringen für Exploration
        skip_solver = random.random() < (self.epsilon * 0.2)
        
        if not skip_solver and game is not None:
            # Versuche zuerst Constraint Solver
            solver = MinesweeperSolver(game)
            safe_moves = solver.get_safe_moves()
            
            if safe_moves:
                # Sicherer Zug gefunden! Konvertiere zu Aktion
                row, col = random.choice(safe_moves)
                action = row * self.board_width + col
                self.stats['solver_moves'] += 1
                return action
            
            # ✨ NEU: Wenn keine sicheren Züge UND genug Punkte → Nutze Blitz!
            if game.points >= game.blitz_cost and len(safe_moves) == 0:
                fields_revealed = game.use_blitz()
                if fields_revealed > 0:
                    # Blitz hat Felder aufgedeckt, versuche erneut Solver
                    safe_moves = solver.get_safe_moves()
                    if safe_moves:
                        row, col = random.choice(safe_moves)
                        action = row * self.board_width + col
                        self.stats['solver_moves'] += 1
                        return action
            
            # Nutze probabilistische beste Vermutung nur bei niedrigem Epsilon
            if self.epsilon < 0.15 and valid_actions is not None:
                if random.random() > 0.3:
                    best_guess = solver.get_best_guess(valid_actions)
                    if best_guess is not None:
                        self.stats['solver_moves'] += 1
                        return best_guess
        
        # Kein sicherer Zug gefunden oder Solver übersprungen → RL für Vermutung
        self.stats['rl_moves'] += 1
        return super().select_action(state, valid_actions)
    
    def _try_use_powerups(self, game) -> bool:
        """
        Intelligente Joker-Nutzung basierend auf Spielsituation.
        
        Strategien:
        1. Mystery-Zellen enthüllen wenn sinnvoll
        2. Blitz nutzen wenn festgefahren
        
        Returns:
            True wenn Joker genutzt wurde
        """
        # Nur nutzen wenn nicht im Tetris-Modus
        if game.tetris_active:
            return False
        
        solver = MinesweeperSolver(game)
        safe_moves = solver.get_safe_moves()
        
        # ✨ STRATEGIE 1: Mystery-Zellen intelligent enthüllen
        # Wenn Mystery-Zellen existieren und genug Punkte
        if len(game.mystery_cells) > 0 and game.points >= game.mystery_reveal_cost:
            # Enthülle Mystery-Zelle wenn sie wichtig sein könnte
            # (z.B. hat viele verdeckte Nachbarn)
            for mystery_row, mystery_col in game.mystery_cells:
                cell = game.board.get_cell(mystery_row, mystery_col)
                if not cell or not cell.is_revealed():
                    continue
                
                # Zähle verdeckte Nachbarn
                hidden_neighbors = 0
                for dr, dc in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                    nr, nc = mystery_row + dr, mystery_col + dc
                    if 0 <= nr < game.height and 0 <= nc < game.width:
                        neighbor = game.board.get_cell(nr, nc)
                        if neighbor and neighbor.is_hidden():
                            hidden_neighbors += 1
                
                # Enthülle wenn viele verdeckte Nachbarn (könnte helfen!)
                if hidden_neighbors >= 3 and random.random() < 0.5:
                    if game.reveal_mystery_cell(mystery_row, mystery_col):
                        return True
        
        # ✨ STRATEGIE 2: Blitz wenn festgefahren (keine sicheren Züge)
        if len(safe_moves) == 0 and game.points >= game.blitz_cost:
            # Nur mit 30% Chance nutzen (nicht zu aggressiv)
            if random.random() < 0.3:
                fields = game.use_blitz()
                return fields > 0
        
        return False
    
    def get_stats(self) -> dict:
        """Gibt Statistiken über Zug-Quellen zurück."""
        total = self.stats['total_moves']
        if total == 0:
            return {
                'solver_percentage': 0.0,
                'rl_percentage': 0.0,
                **self.stats
            }
        
        return {
            'solver_percentage': (self.stats['solver_moves'] / total) * 100,
            'rl_percentage': (self.stats['rl_moves'] / total) * 100,
            **self.stats
        }
    
    def reset_stats(self):
        """Setzt Zug-Statistiken zurück."""
        self.stats = {
            'solver_moves': 0,
            'rl_moves': 0,
            'total_moves': 0
        }
    
    def save(self, filepath: str):
        """Save model to file with hybrid agent info."""
        import torch
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'board_height': self.board_height,
            'board_width': self.board_width,
            'action_space_size': self.action_space_size,
            'is_hybrid': True,  # Markiere als Hybrid-Modell
            'use_solver': self.use_solver
        }, filepath)

