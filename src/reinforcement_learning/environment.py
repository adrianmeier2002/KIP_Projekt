"""RL Environment wrapper for Minesweeper."""

import numpy as np
from typing import Tuple, Optional
from src.minesweeper.game import Game
from src.utils.constants import (
    BOARD_WIDTH, BOARD_HEIGHT,
    CELL_HIDDEN, CELL_REVEALED, CELL_FLAGGED, CELL_MINE
)

STATE_CHANNELS = 12  # Erhöht auf 12: +Frontier +Safe-Cell +Mystery +Speed +Tetris Kanäle
NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]

FLAG_SET_REWARD = 0.2
FLAG_SET_PENALTY = 0.2
FLAG_FINAL_CORRECT_BONUS = 0.5
FLAG_FINAL_INCORRECT_PENALTY = 0.5


class MinesweeperEnvironment:
    """Environment wrapper for reinforcement learning."""
    
    def __init__(
        self,
        difficulty: str = "medium",
        width: int = BOARD_WIDTH,
        height: int = BOARD_HEIGHT,
        use_flag_actions: bool = False,
        enable_challenges: bool = True  # AKTIVIERT: Modell soll mit Features umgehen!
    ):
        """
        Initialize the environment.
        
        Args:
            difficulty: Game difficulty level
            width: Board width
            height: Board height
            use_flag_actions: Whether to use flag actions
            enable_challenges: Whether to enable challenges (Mystery/Speed/Tetris) - Default TRUE!
        """
        self.difficulty = difficulty
        self.width = width
        self.height = height
        self.enable_challenges = enable_challenges
        self.game = Game(difficulty, width, height, enable_challenges=enable_challenges)
        self.cell_count = width * height
        self.use_flag_actions = use_flag_actions
        self.action_space_size = (
            self.cell_count if not use_flag_actions else self.cell_count * 2
        )  # Reveal (+ optional Flag) per cell
        self.state_channels = STATE_CHANNELS
        self.reward_scale = max(1.0, (self.width * self.height) / 100.0)
        self.progress_scale = max(0.5, (self.width * self.height) / 300.0)
        self.first_move_done = False  # Track if first move has been made
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment and return initial state.
        
        Returns:
            Initial state observation
        """
        self.game.new_game(self.difficulty, self.width, self.height, enable_challenges=self.enable_challenges)
        self.first_move_done = False  # Reset first move flag
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action index (0 to action_space_size - 1)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # ✨ NEU: Tetris-Handling - Auto-Platzierung wenn aktiv
        if self.game.tetris_active:
            valid_placed = self._handle_tetris_auto_placement()
            if valid_placed:
                # Tetris platziert, weitermachen mit normalem Zug
                pass
            else:
                # Kein gültiger Tetris-Platz → kleiner Reward-Abzug aber weitermachen
                pass
        
        # Convert action to row, col
        cell_index = action % self.cell_count if self.use_flag_actions and action >= self.cell_count else action
        row = cell_index // self.width
        col = cell_index % self.width
        action_type = "flag" if (self.use_flag_actions and action >= self.cell_count) else "reveal"
        
        # Get previous state
        prev_revealed = self.game.revealed_count
        reward = 0.0
        valid = False
        
        move_context = self._get_move_context(row, col)
        
        # ✨ NEU: Prüfe ob diese Zelle als sicher markiert ist
        current_state = self._get_state()
        is_safe_cell = current_state[8, row, col] > 0.5 if current_state.shape[0] > 8 else False
        
        if action_type == "reveal":
            valid = self.game.reveal_cell(row, col)
            reward += self._calculate_reward(
                prev_revealed,
                valid,
                action_type=action_type,
                move_context=move_context,
                is_first_move=not self.first_move_done,
                was_safe_cell=is_safe_cell
            )
            if valid:
                self.first_move_done = True  # Mark first move as done after successful reveal
        else:
            valid, flag_reward = self._handle_flag_action(row, col)
            reward += flag_reward
        
        # Check if done
        done = self.game.is_game_over()
        
        # Get next state
        next_state = self._get_state()
        
        # Info
        info = {
            "won": self.game.is_won(),
            "lost": self.game.is_lost(),
            "valid_action": valid
        }
        
        # Additional rewards/penalties when the game ends after this move
        if done and self.use_flag_actions and self.game.is_lost():
            reward += self._calculate_flag_end_bonus()
        
        return next_state, reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation with improved features.
        
        State channels (12 total):
        0: Basis-Encoding (kombinierter Wert für schnelle Übersicht)
        1: Hidden-Maske (1.0 für verdeckte Zellen)
        2: Flag-Maske (1.0 für geflaggte Zellen)
        3: Aufgedeckte Zahl (normiert 0-1, -1 für Mine)
        4: Verdeckte Nachbarn (0-1)
        5: Geflaggte Nachbarn (0-1)
        6: Hinweis-Summe benachbarter Zellen (0-1)
        7: Frontier-Maske (1.0 wenn verdeckt UND neben aufgedeckter Zelle)
        8: Safe-Cell-Maske (1.0 wenn mathematisch 100% sicher) ✨
        9: Mystery-Maske (1.0 wenn Mystery-Zelle) ✨ NEU!
        10: Speed-Aktiv-Maske (1.0 überall wenn Speed-Timer läuft) ✨ NEU!
        11: Tetris-Aktiv-Maske (1.0 überall wenn Tetris-Modus aktiv) ✨ NEU!
        
        Returns:
            State array of shape (STATE_CHANNELS, height, width)
        """
        state = np.zeros((STATE_CHANNELS, self.height, self.width), dtype=np.float32)
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                adjacent_revealed = 0
                adjacent_hint_sum = 0
                hidden_neighbors = 0
                flagged_neighbors = 0
                
                for dr, dc in NEIGHBOR_OFFSETS:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        neighbor = self.game.board.get_cell(nr, nc)
                        if neighbor.is_hidden():
                            hidden_neighbors += 1
                        elif neighbor.is_flagged():
                            flagged_neighbors += 1
                        elif neighbor.is_revealed():
                            adjacent_revealed += 1
                            if not neighbor.is_mine:
                                adjacent_hint_sum += neighbor.adjacent_mines
                
                # Channel 0: Basis-Encoding (vereinfacht, klarer)
                if cell.is_revealed():
                    if cell.is_mine:
                        state[0, row, col] = -1.0
                        state[3, row, col] = -1.0
                    else:
                        value = cell.adjacent_mines / 8.0
                        state[0, row, col] = value
                        state[3, row, col] = value
                elif cell.is_flagged():
                    state[0, row, col] = -0.5
                    state[2, row, col] = 1.0
                else:  # Hidden cell - VEREINFACHTES ENCODING
                    # Einfacher Wert: -1.0 für komplett unbekannt
                    # Wird durch andere Kanäle ergänzt (Frontier, Nachbarn)
                    state[0, row, col] = -0.8
                    state[1, row, col] = 1.0
                
                # Channels 4-6: Nachbarschaftsinformationen
                state[4, row, col] = hidden_neighbors / 8.0
                state[5, row, col] = flagged_neighbors / 8.0
                state[6, row, col] = min(1.0, adjacent_hint_sum / 64.0)
                
                # Channel 7: FRONTIER-MASKE (NEU!) ✨
                # Eine Zelle ist "Frontier", wenn sie verdeckt ist UND
                # mindestens einen aufgedeckten Nachbarn hat
                if cell.is_hidden() and not cell.is_flagged() and adjacent_revealed > 0:
                    state[7, row, col] = 1.0
                else:
                    state[7, row, col] = 0.0
        
        # Channel 8: SAFE-CELL-MASKE (KRITISCH!) ✨✨✨
        # Berechne welche Zellen mathematisch 100% sicher sind
        # Logik: Wenn eine aufgedeckte Zelle N hat und N Minen geflaggt sind,
        #        dann sind alle anderen verdeckten Nachbarn SICHER!
        state[8] = self._calculate_safe_cells()
        
        # Channel 9: MYSTERY-MASKE ✨ NEU!
        # Markiert Mystery-Zellen (Fragezeichen statt Zahl)
        for mystery_pos in self.game.mystery_cells:
            r, c = mystery_pos
            state[9, r, c] = 1.0
        
        # Channel 10: SPEED-AKTIV-MASKE ✨ NEU!
        # Wenn Speed-Timer läuft, markiere ALLE Zellen (globaler Zustand)
        if self.game.speed_active:
            state[10, :, :] = self.game.speed_time_remaining / self.game.speed_time_limit
        
        # Channel 11: TETRIS-AKTIV-MASKE ✨ NEU!
        # Wenn Tetris aktiv, markiere ALLE Zellen (globaler Zustand)
        if self.game.tetris_active:
            state[11, :, :] = 1.0
        
        return state
    
    def _calculate_safe_cells(self) -> np.ndarray:
        """
        Berechne welche verdeckten Zellen mathematisch 100% sicher sind.
        
        VEREINFACHTE LOGIK (funktioniert OHNE Flags):
        
        Einfache Patterns die der Agent lernen kann:
        1. Zelle zeigt 0 → Alle verdeckten Nachbarn SICHER (sollte auto-revealed sein)
        2. Zelle zeigt N, hat N+K Nachbarn, nur N sind verdeckt → Alle N UNSICHER
        3. Cross-Pattern: Kombinierte Constraints aus mehreren Zellen
        
        WICHTIG: Ignoriert Mystery-Zellen (Herausforderungs-Feature)!
        
        Returns:
            2D Array (height, width) mit 1.0 für sichere Zellen, 0.0 sonst
        """
        safe_cells = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Pass 1: Einfache Patterns
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                
                # Nur aufgedeckte Zellen mit Zahlen
                if not cell.is_revealed() or cell.is_mine:
                    continue
                
                # NEU: Mystery-Zellen ignorieren (ihre Zahl ist für Spieler versteckt!)
                if (row, col) in self.game.mystery_cells:
                    continue
                
                # Sammle Nachbar-Informationen
                flagged_count = 0
                hidden_neighbors = []
                revealed_neighbors = []
                
                for dr, dc in NEIGHBOR_OFFSETS:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        neighbor = self.game.board.get_cell(nr, nc)
                        if neighbor.is_flagged():
                            flagged_count += 1
                        elif neighbor.is_hidden():
                            hidden_neighbors.append((nr, nc))
                        elif neighbor.is_revealed():
                            revealed_neighbors.append((nr, nc))
                
                # Pattern 1: Zelle zeigt 0 → ALLE Nachbarn sicher
                if cell.adjacent_mines == 0 and len(hidden_neighbors) > 0:
                    for hr, hc in hidden_neighbors:
                        safe_cells[hr, hc] = 1.0
                
                # Pattern 2: Flags erfüllen bereits die Anzahl → Rest ist sicher
                if flagged_count == cell.adjacent_mines and len(hidden_neighbors) > 0:
                    for hr, hc in hidden_neighbors:
                        safe_cells[hr, hc] = 1.0
                
                # Pattern 3: Nur 1 verdeckter Nachbar und Zahl ist 1
                # Aber wir wissen nicht ob es eine Mine ist → NICHT sicher!
                # (Dieser Fall hilft nicht)
        
        # Pass 2: Cross-Pattern Detection (fortgeschritten)
        # Beispiel: Zwei benachbarte aufgedeckte Zellen mit überlappenden Nachbarn
        # können zusammen mehr Information geben
        safe_cells = self._advanced_safe_cell_detection(safe_cells)
        
        return safe_cells
    
    def _advanced_safe_cell_detection(self, safe_cells: np.ndarray) -> np.ndarray:
        """
        Fortgeschrittene Pattern-Erkennung für sichere Zellen.
        
        Nutzt kombinierte Constraints aus mehreren aufgedeckten Zellen.
        
        WICHTIG: Ignoriert Mystery-Zellen!
        """
        # Durchlaufe alle Paare von benachbarten aufgedeckten Zellen
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                
                if not cell.is_revealed() or cell.is_mine or cell.adjacent_mines == 0:
                    continue
                
                # NEU: Mystery-Zellen ignorieren
                if (row, col) in self.game.mystery_cells:
                    continue
                
                # Prüfe alle Nachbarn dieser Zelle
                for dr, dc in NEIGHBOR_OFFSETS:
                    nr, nc = row + dr, col + dc
                    if not (0 <= nr < self.height and 0 <= nc < self.width):
                        continue
                    
                    neighbor_cell = self.game.board.get_cell(nr, nc)
                    if not neighbor_cell.is_revealed() or neighbor_cell.is_mine or neighbor_cell.adjacent_mines == 0:
                        continue
                    
                    # NEU: Mystery-Zellen auch bei Nachbarn ignorieren
                    if (nr, nc) in self.game.mystery_cells:
                        continue
                    
                    # Jetzt haben wir zwei benachbarte aufgedeckte Zellen
                    # Analysiere ihre gemeinsamen und exklusiven Nachbarn
                    safe_cells = self._analyze_cell_pair(
                        row, col, nr, nc, cell, neighbor_cell, safe_cells
                    )
        
        return safe_cells
    
    def _analyze_cell_pair(self, r1: int, c1: int, r2: int, c2: int,
                           cell1, cell2, safe_cells: np.ndarray) -> np.ndarray:
        """
        Analysiere zwei benachbarte aufgedeckte Zellen für sichere Nachbarn.
        
        Beispiel-Pattern:
        Cell1 zeigt 1, hat 3 verdeckte Nachbarn: A, B, C
        Cell2 zeigt 1, hat 3 verdeckte Nachbarn: A, B, D
        → Wenn A oder B eine Mine ist, ist D bzw. C sicher!
        (Zu komplex für ersten Ansatz, skip)
        """
        # Für jetzt: Zu komplex, return unchanged
        return safe_cells
    
    def _calculate_reward(
        self,
        prev_revealed: int,
        valid_action: bool,
        action_type: str,
        move_context: Optional[dict] = None,
        is_first_move: bool = False,
        was_safe_cell: bool = False
    ) -> float:
        """
        Calculate reward for the action - VERBESSERTE VERSION V3.
        
        Wichtige Änderungen V3:
        - Sichere Zellen (100% garantiert) werden EXTREM stark belohnt! ✨
        - Erster Zug wird speziell behandelt (keine Guess-Penalty)
        - Stärkere Reward-Skalierung unabhängig von Board-Größe
        - Klarere Signale für Frontier vs. Guess
        
        Args:
            prev_revealed: Number of revealed cells before action
            valid_action: Whether the action was valid
            action_type: Type of action ("reveal" or "flag")
            move_context: Context information about the move (neighbors etc.)
            is_first_move: Whether this is the first move of the episode
            was_safe_cell: Whether this cell was marked as 100% safe
            
        Returns:
            Reward value
        """
        if action_type != "reveal":
            return 0.0
        
        if not valid_action:
            return -0.6 * self.reward_scale  # Penalty for invalid action
        
        board_scale = self.reward_scale
        cells_revealed = self.game.revealed_count - prev_revealed
        progress_ratio = self.game.revealed_count / max(1, self.game.total_safe_cells)
        
        # Bestimme, ob es ein Frontier-Move oder ein Guess war
        is_guess = move_context.get("adjacent_revealed", 0) == 0 if move_context else True
        frontier_neighbors = move_context.get("adjacent_revealed", 0) if move_context else 0
        frontier_factor = frontier_neighbors / 8.0
        
        # ✨ KORREKTUR: Erster Zug hat IMMER keine Frontier → nicht als Guess behandeln!
        if is_first_move:
            is_guess = False  # Erster Zug ist neutral/notwendig
        
        # ========== SPIEL VERLOREN ========== 
        if self.game.is_lost():
            base_penalty = -12.0 * board_scale
            
            # ✨ NEU: Reduzierte Strafe wenn durch Speed-Timer verloren!
            # Speed-Timeout ist eine Spielmechanik, keine schlechte Entscheidung
            if self.game.speed_active and self.game.speed_time_remaining <= 0:
                return base_penalty * 0.5  # 50% Strafe
            
            # ✨ KORREKTUR: Frontier-Moves werden weniger bestraft!
            # Ein Frontier-Move war eine informierte Entscheidung, auch wenn es zur Mine führte
            if not is_guess and frontier_neighbors > 0:
                # Reduziere die Strafe für Frontier-Moves um 40%
                penalty_reduction = 0.4
                return base_penalty * (1.0 - penalty_reduction)
            else:
                # Volle Strafe für zufällige Guesses
                return base_penalty
        
        # ========== SPIEL GEWONNEN ==========
        if self.game.is_won():
            # Erhöhter Win-Reward für stärkeres Signal
            # Minimum für kleine Boards, skaliert für große
            win_reward = max(50.0, 25.0 * board_scale) + 10.0 * progress_ratio
            
            # ✨ NEU: Bonus wenn mit aktiven Challenges gewonnen!
            if len(self.game.mystery_cells) > 0 or len(self.game.speed_cells) > 0 or len(self.game.tetris_cells) > 0:
                challenge_bonus = 10.0 * board_scale
                win_reward += challenge_bonus
            
            return win_reward
        
        # ========== NORMALE ZÜGE ==========
        if cells_revealed > 0:
            # Basis-Reward für Fortschritt
            base_reward = self.progress_scale * cells_revealed
            
            # Bonus für Kettenreaktionen (Auto-Reveal)
            chain_bonus = 0.5 * self.progress_scale * max(0, cells_revealed - 1)
            
            # ✨ NEU: Bonus wenn Speed-Challenge bewältigt!
            # Wenn ein Feld während Speed-Timer aufgedeckt wurde → Belohnung
            speed_bonus = 0.0
            if self.game.speed_active or (prev_revealed < self.game.revealed_count and len(self.game.speed_cells) > 0):
                # Agent hat unter Zeitdruck gehandelt - gut!
                speed_bonus = 2.0 * self.progress_scale
            
            # ✨✨✨ KRITISCH: SICHERE ZELLEN EXTREM BELOHNEN! ✨✨✨
            # Wenn Agent eine mathematisch sichere Zelle aufdeckt,
            # MASSIVER Bonus - das ist perfektes Minesweeper-Spiel!
            if was_safe_cell:
                # EXTREM HOHER Bonus für sichere Züge!
                # Das ist das BESTE was der Agent tun kann
                safe_cell_bonus = max(30.0, 15.0 * board_scale) * (1.0 + cells_revealed / 5.0)
                return base_reward + chain_bonus + safe_cell_bonus + speed_bonus
            
            # ✨ VERSTÄRKTER FRONTIER-BONUS V2
            # Stärkere Skalierung unabhängig von Board-Größe für klarere Signale
            # Belohne Frontier-Moves SEHR stark, bestrafe Guesses SEHR stark
            if is_guess:
                # SEHR starke Penalty für zufällige Züge (nicht erster Zug!)
                # Skaliert mit Board-Größe, aber mit Minimum für kleine Boards
                frontier_reward = -max(8.0, 3.0 * board_scale)
            else:
                # SEHR starker Bonus für Frontier-Moves
                # Skaliert mit Board-Größe und Anzahl Nachbarn
                frontier_reward = max(10.0, 4.0 * board_scale) * (1.0 + frontier_factor)
            
            # Gesamt-Reward
            total_reward = base_reward + chain_bonus + frontier_reward + speed_bonus
            return total_reward
        
        # Keine Zellen aufgedeckt -> leichte Strafe
        return -0.2 * self.progress_scale
    
    def _handle_flag_action(self, row: int, col: int) -> Tuple[bool, float]:
        """
        Handle flag toggling and return (valid, reward).
        """
        cell = self.game.board.get_cell(row, col)
        if not cell or cell.is_revealed():
            return False, -0.1
        
        toggled = self.game.toggle_flag(row, col)
        if not toggled:
            return False, -0.1
        
        if cell.is_flagged():
            if cell.is_mine:
                return True, FLAG_SET_REWARD
            return True, -FLAG_SET_PENALTY
        
        # Flag removed (no immediate reward to avoid oscillation)
        return True, 0.0
    
    def _calculate_flag_end_bonus(self) -> float:
        """Reward/punish based on flagged cells when the game ends."""
        correct_flags = 0
        incorrect_flags = 0
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                if cell and cell.is_flagged():
                    if cell.is_mine:
                        correct_flags += 1
                    else:
                        incorrect_flags += 1
        
        return (
            correct_flags * FLAG_FINAL_CORRECT_BONUS
            - incorrect_flags * FLAG_FINAL_INCORRECT_PENALTY
        )
    
    def get_valid_actions(self) -> np.ndarray:
        """
        Get mask of valid actions (cells that can be clicked).
        
        Returns:
            Boolean array of valid actions
        """
        valid_reveal = np.zeros(self.cell_count, dtype=bool)
        valid_flag = np.zeros(self.cell_count, dtype=bool) if self.use_flag_actions else None
        
        for row in range(self.height):
            for col in range(self.width):
                cell = self.game.board.get_cell(row, col)
                action_idx = row * self.width + col
                
                # Reveal valid if cell hidden and not flagged
                if cell.is_hidden() and not cell.is_flagged():
                    valid_reveal[action_idx] = True
                
                # Flag valid if cell hidden or currently flagged (toggle)
                if self.use_flag_actions and not cell.is_revealed():
                    if cell.is_hidden() or cell.is_flagged():
                        valid_flag[action_idx] = True
        
        if not self.use_flag_actions:
            return valid_reveal
        
        return np.concatenate([valid_reveal, valid_flag])
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get action mask for masking invalid actions in Q-values.
        
        Returns:
            Mask array (1.0 for valid, -inf for invalid)
        """
        valid_actions = self.get_valid_actions()
        mask = np.where(valid_actions, 0.0, -np.inf)
        return mask

    def _handle_tetris_auto_placement(self) -> bool:
        """
        INTELLIGENTE Tetris-Platzierung für RL-Training.
        
        Bevorzugt sichere Positionen (keine Minen) basierend auf:
        1. Solver-Informationen (welche Zellen sind sicher)
        2. Heuristiken (viele aufgedeckte Nachbarn = wahrscheinlich sicher)
        
        Returns:
            True wenn platziert, False wenn keine gültige Position gefunden
        """
        if not self.game.tetris_active or not self.game.tetris_current_shape:
            return False
        
        # Sammle alle gültigen Positionen mit Scores
        from src.reinforcement_learning.constraint_solver import MinesweeperSolver
        solver = MinesweeperSolver(self.game)
        safe_cells = set(solver.get_safe_moves())  # 100% sichere Zellen
        
        position_scores = []
        for row in range(self.height):
            for col in range(self.width):
                if not self.game._can_place_tetris_shape_at(self.game.tetris_current_shape, row, col):
                    continue
                
                # Berechne Score für diese Position
                score = 0
                cells_in_shape = []
                
                # Prüfe alle Zellen in der Form
                for dr, dc in self.game.tetris_current_shape:
                    r, c = row + dr, col + dc
                    cells_in_shape.append((r, c))
                    
                    # Sehr hoher Score wenn Zelle als sicher markiert ist
                    if (r, c) in safe_cells:
                        score += 100
                    
                    # Mittlerer Score für Frontier-Zellen (neben aufgedeckten)
                    cell = self.game.board.get_cell(r, c)
                    if cell:
                        revealed_neighbors = sum(
                            1 for dr2, dc2 in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
                            if 0 <= r+dr2 < self.height and 0 <= c+dc2 < self.width
                            and self.game.board.get_cell(r+dr2, c+dc2).is_revealed()
                        )
                        score += revealed_neighbors * 5
                
                # Bonus wenn ALLE Zellen der Form sicher sind
                if all(cell in safe_cells for cell in cells_in_shape):
                    score += 500  # Riesiger Bonus!
                
                position_scores.append((score, row, col))
        
        if not position_scores:
            # Keine gültige Position → Deaktiviere Tetris-Modus
            self.game.tetris_active = False
            self.game.tetris_current_shape = None
            return False
        
        # Sortiere nach Score (höchster zuerst)
        position_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Nutze beste Position (oder Top 3 für etwas Randomness)
        import random
        top_positions = position_scores[:min(3, len(position_scores))]
        _, row, col = random.choice(top_positions)
        
        success = self.game.place_tetris_shape(row, col)
        return success
    
    def _get_move_context(self, row: int, col: int) -> dict:
        """Collect contextual information for the selected move."""
        context = {
            "adjacent_revealed": 0,
            "adjacent_hidden": 0,
            "adjacent_flagged": 0
        }
        
        for dr, dc in NEIGHBOR_OFFSETS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                neighbor = self.game.board.get_cell(nr, nc)
                if not neighbor:
                    continue
                if neighbor.is_revealed():
                    context["adjacent_revealed"] += 1
                elif neighbor.is_flagged():
                    context["adjacent_flagged"] += 1
                elif neighbor.is_hidden():
                    context["adjacent_hidden"] += 1
        
        return context


