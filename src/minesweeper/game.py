"""Game-Klasse für die Minesweeper-Spiellogik."""

import random
from typing import List, Tuple, Optional
from src.minesweeper.board import Board
from src.utils.constants import (
    BOARD_WIDTH, BOARD_HEIGHT,
    get_mine_count
)
from src.reinforcement_learning.constraint_solver import MinesweeperSolver


class GameState:
    """Spielzustands-Enumeration."""
    PLAYING = "playing"
    WON = "won"
    LOST = "lost"


class Game:
    """Verwaltet die Minesweeper-Spiellogik."""
    
    def __init__(self, difficulty: str = "medium", width: int = BOARD_WIDTH, height: int = BOARD_HEIGHT, 
                 enable_challenges: bool = True):
        """
        Initialisiert ein neues Spiel.
        
        Args:
            difficulty: Schwierigkeitsgrad ("easy", "medium" oder "hard")
            width: Breite des Spielfelds
            height: Höhe des Spielfelds
            enable_challenges: Ob Herausforderungen (Mystery/Speed/Tetris) aktiviert sind (für Training: False)
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
        
        # NEU: Feature-Toggle für Training/Performance
        self.enable_challenges = enable_challenges
        
        # Punktesystem
        self.points = 0  # Verfügbare Punkte
        self.points_per_cell = 1  # Punkte pro aufgedecktem Feld
        # Power-Up Kosten
        self.radar_cost = 70
        self.scanner_cost = 30
        self.blitz_cost = 50
        # Power-Up Effekte
        self.radar_scanned_cells = set()  # Speichert Zellen, die mit Radar gescannt wurden
        self.scanner_result_cells = {}  # Speichert Scanner-Ergebnisse {(row, col): mine_count}
        
        # Herausforderungen (können deaktiviert werden für Training)
        self.mystery_cells = set()  # Zellen mit Fragezeichen statt echter Zahl
        self.mystery_reveal_cost = 20  # Kosten um Mystery-Zahl zu enthüllen
        # ANGEPASST: Seltener triggern (vorher 4-8, jetzt 15-25)
        self.mystery_next_trigger = random.randint(15, 25) if enable_challenges else 999999
        self.mystery_counter = 0  # Zähler für Mystery-Trigger
        
        # Speed-Felder
        self.speed_cells = set()  # Zellen die Speed-Felder sind
        self.speed_active = False  # Ob gerade ein Speed-Timer läuft
        self.speed_time_remaining = 0.0  # Verbleibende Zeit in Sekunden
        self.speed_time_limit = 5.0  # 5 Sekunden Zeit
        # ANGEPASST: Seltener triggern (vorher 6-12, jetzt 20-30)
        self.speed_next_trigger = random.randint(20, 30) if enable_challenges else 999999
        self.speed_counter = 0  # Zähler für Speed-Trigger
        
        # Tetris-Felder
        self.tetris_cells = set()  # Zellen die Tetris-Felder sind
        self.tetris_active = False  # Ob gerade eine Tetris-Form platziert werden muss
        self.tetris_current_shape = None  # Aktuelle Tetris-Form (Liste von (dr, dc) Offsets)
        # ANGEPASST: Seltener triggern (vorher 10-15, jetzt 25-40)
        self.tetris_next_trigger = random.randint(25, 40) if enable_challenges else 999999
        self.tetris_counter = 0  # Zähler für Tetris-Trigger
        # Tetris-Formen (relative Koordinaten vom Zentrum)
        self.tetris_shapes = {
            'I': [(0, -1), (0, 0), (0, 1), (0, 2)],     # ■■■■
            'O': [(0, 0), (0, 1), (1, 0), (1, 1)],      # ■■
                                                          # ■■
            'T': [(0, -1), (0, 0), (0, 1), (1, 0)],     # ■■■
                                                          #  ■
            'S': [(0, 0), (0, 1), (1, -1), (1, 0)],     #  ■■
                                                          # ■■
            'Z': [(0, -1), (0, 0), (1, 0), (1, 1)],     # ■■
                                                          #  ■■
            'L': [(0, -1), (0, 0), (0, 1), (1, -1)],    # ■■■
                                                          # ■
            'J': [(0, -1), (0, 0), (0, 1), (1, 1)]      # ■■■
                                                          #    ■
        }
    
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
            self.points += self.points_per_cell  # Punkte vergeben
            
            # Deaktiviere Speed-Timer wenn ein Feld aufgedeckt wurde
            if self.speed_active:
                self.speed_active = False
                self.speed_time_remaining = 0.0
            
            # NEU: Herausforderungen nur wenn aktiviert
            if self.enable_challenges:
                # Prüfe ob dieses Feld bereits eine Herausforderung ist
                is_already_challenge = (
                    (row, col) in self.mystery_cells or
                    (row, col) in self.speed_cells or
                    (row, col) in self.tetris_cells
                )
                
                # Prüfe ob dieses Feld ein Mystery-Feld wird (nur Zahlenfelder)
                if cell.adjacent_mines > 0 and not is_already_challenge:
                    self.mystery_counter += 1
                    if self.mystery_counter >= self.mystery_next_trigger:
                        self.mystery_cells.add((row, col))
                        # ANGEPASST: Seltener triggern (vorher 4-8, jetzt 15-25)
                        self.mystery_counter = 0
                        self.mystery_next_trigger = random.randint(15, 25)
                        # Markiere als Herausforderung
                        is_already_challenge = True
                
                # Prüfe ob dieses Feld ein Speed-Feld wird (nur wenn noch keine Herausforderung)
                if not is_already_challenge:
                    self.speed_counter += 1
                    if self.speed_counter >= self.speed_next_trigger:
                        self.speed_cells.add((row, col))
                        # Aktiviere Speed-Timer
                        self.speed_active = True
                        self.speed_time_remaining = self.speed_time_limit
                        # ANGEPASST: Seltener triggern (vorher 6-12, jetzt 20-30)
                        self.speed_counter = 0
                        self.speed_next_trigger = random.randint(20, 30)
                        # Markiere als Herausforderung
                        is_already_challenge = True
                
                # Prüfe ob dieses Feld ein Tetris-Feld wird (nur wenn noch keine Herausforderung)
                if not is_already_challenge:
                    self.tetris_counter += 1
                    if self.tetris_counter >= self.tetris_next_trigger:
                        self.tetris_cells.add((row, col))
                        # Aktiviere Tetris-Modus mit gültiger Form
                        valid_shape = self._get_valid_tetris_shape()
                        if valid_shape:
                            self.tetris_active = True
                            self.tetris_current_shape = valid_shape
                        # ANGEPASST: Seltener triggern (vorher 10-15, jetzt 25-40)
                        self.tetris_counter = 0
                        self.tetris_next_trigger = random.randint(25, 40)
            
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
                self.points += self.points_per_cell  # Punkte für auto-aufgedeckte Zellen
            
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
    
    def use_radar(self, center_row: int, center_col: int) -> bool:
        """
        Verwendet ein Radar, um einen 3x3-Bereich aufzudecken.
        Minen in diesem Bereich werden markiert aber nicht ausgelöst.
        Kostet 70 Punkte.
        
        Args:
            center_row: Zentrale Zeile des Radar-Bereichs
            center_col: Zentrale Spalte des Radar-Bereichs
            
        Returns:
            True wenn Radar verwendet wurde, False sonst
        """
        if self.state != GameState.PLAYING:
            return False
        
        if self.points < self.radar_cost:
            return False
        
        # Platziere Minen beim ersten Klick (falls noch nicht geschehen)
        if self.first_click:
            self.board.place_mines(center_row, center_col)
            self.first_click = False
        
        # Decke 3x3-Bereich auf und markiere gescannte Zellen
        cells_revealed = 0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r = center_row + dr
                c = center_col + dc
                
                # Prüfe ob Position gültig ist
                if 0 <= r < self.height and 0 <= c < self.width:
                    # Markiere als gescannt für visuelles Feedback
                    self.radar_scanned_cells.add((r, c))
                    
                    cell = self.board.get_cell(r, c)
                    if cell and not cell.is_revealed() and not cell.is_flagged():
                        # Decke Zelle auf, aber nur wenn es keine Mine ist
                        if not cell.is_mine:
                            if cell.reveal():
                                self.revealed_count += 1
                                self.points += self.points_per_cell  # Punkte für aufgedeckte Zellen
                                cells_revealed += 1
                                
                                # Auto-Aufdeckung bei leeren Feldern
                                if cell.adjacent_mines == 0:
                                    self._auto_reveal_safe_neighbors(r, c)
        
        # Verbrauche Punkte
        self.points -= self.radar_cost
        
        # Prüfe Gewinnbedingung
        if self.revealed_count >= self.total_safe_cells:
            self.state = GameState.WON
        
        return True
    
    def use_scanner(self, center_row: int, center_col: int) -> Optional[int]:
        """
        Verwendet einen Scanner, um die Anzahl der Minen in einem 3x3-Bereich zu zählen.
        Kostet 70 Punkte.
        
        Args:
            center_row: Zentrale Zeile des Scanner-Bereichs
            center_col: Zentrale Spalte des Scanner-Bereichs
            
        Returns:
            Anzahl der Minen im Bereich, oder None wenn Scanner nicht verwendet werden konnte
        """
        if self.state != GameState.PLAYING:
            return None
        
        if self.points < self.scanner_cost:
            return None
        
        # Platziere Minen beim ersten Klick (falls noch nicht geschehen)
        if self.first_click:
            self.board.place_mines(center_row, center_col)
            self.first_click = False
        
        # Zähle Minen im 3x3-Bereich
        mine_count = 0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r = center_row + dr
                c = center_col + dc
                
                # Prüfe ob Position gültig ist
                if 0 <= r < self.height and 0 <= c < self.width:
                    cell = self.board.get_cell(r, c)
                    if cell and cell.is_mine:
                        mine_count += 1
        
        # Speichere Ergebnis für Anzeige
        self.scanner_result_cells[(center_row, center_col)] = mine_count
        
        # Verbrauche Punkte
        self.points -= self.scanner_cost
        
        return mine_count
    
    def use_blitz(self) -> int:
        """
        Verwendet einen Blitz, um automatisch 1-3 sichere Felder aufzudecken.
        Nutzt den Constraint-Solver um sichere Züge zu finden.
        Kostet 50 Punkte.
        
        Returns:
            Anzahl der aufgedeckten Felder, oder 0 wenn Blitz nicht verwendet werden konnte
        """
        if self.state != GameState.PLAYING:
            return 0
        
        if self.points < self.blitz_cost:
            return 0
        
        # Platziere Minen beim ersten Klick (falls noch nicht geschehen)
        if self.first_click:
            # Wähle zufällige Position für erste Mine-Platzierung
            import random
            row = random.randint(0, self.height - 1)
            col = random.randint(0, self.width - 1)
            self.board.place_mines(row, col)
            self.first_click = False
        
        # Erstelle Minesweeper-Solver
        solver = MinesweeperSolver(self)
        safe_cells = solver.get_safe_moves()
        
        # Decke bis zu 3 sichere Felder auf
        fields_revealed = 0
        max_fields = min(3, len(safe_cells))
        
        for i in range(max_fields):
            if i < len(safe_cells):
                row, col = safe_cells[i]
                cell = self.board.get_cell(row, col)
                if cell and not cell.is_revealed() and not cell.is_flagged():
                    if cell.reveal():
                        self.revealed_count += 1
                        self.points += self.points_per_cell  # Punkte für aufgedeckte Zellen
                        fields_revealed += 1
                        
                        # Auto-Aufdeckung bei leeren Feldern
                        if cell.adjacent_mines == 0:
                            self._auto_reveal_safe_neighbors(row, col)
        
        # Verbrauche Punkte nur wenn mindestens ein Feld aufgedeckt wurde
        if fields_revealed > 0:
            self.points -= self.blitz_cost
            
            # Prüfe Gewinnbedingung
            if self.revealed_count >= self.total_safe_cells:
                self.state = GameState.WON
        
        return fields_revealed
    
    def reveal_mystery_cell(self, row: int, col: int) -> bool:
        """
        Enthüllt die echte Zahl eines Mystery-Feldes.
        Kostet 20 Punkte.
        
        Args:
            row: Zeilenindex
            col: Spaltenindex
            
        Returns:
            True wenn Mystery-Feld enthüllt wurde, False sonst
        """
        if self.state != GameState.PLAYING:
            return False
        
        if (row, col) not in self.mystery_cells:
            return False
        
        if self.points < self.mystery_reveal_cost:
            return False
        
        # Entferne aus Mystery-Zellen
        self.mystery_cells.remove((row, col))
        
        # Verbrauche Punkte
        self.points -= self.mystery_reveal_cost
        
        return True
    
    def update_speed_timer(self, delta_time: float) -> bool:
        """
        Aktualisiert den Speed-Timer.
        
        Args:
            delta_time: Zeit seit letztem Update in Sekunden
            
        Returns:
            True wenn Timer abgelaufen (Game Over), False sonst
        """
        if not self.speed_active:
            return False
        
        self.speed_time_remaining -= delta_time
        
        if self.speed_time_remaining <= 0:
            # Zeit abgelaufen - Game Over
            self.state = GameState.LOST
            self.speed_active = False
            return True
        
        return False
    
    def _get_valid_tetris_shape(self) -> Optional[List[Tuple[int, int]]]:
        """
        Findet eine Tetris-Form die platzierbar ist.
        
        Returns:
            Liste von (dr, dc) Offsets oder None wenn keine Form passt
        """
        # Mische Formen für Zufälligkeit
        shape_names = list(self.tetris_shapes.keys())
        random.shuffle(shape_names)
        
        for shape_name in shape_names:
            shape = self.tetris_shapes[shape_name]
            # Prüfe ob diese Form irgendwo platziert werden kann
            if self._can_place_tetris_shape_anywhere(shape):
                return shape
        
        return None
    
    def _can_place_tetris_shape_anywhere(self, shape: List[Tuple[int, int]]) -> bool:
        """
        Prüft ob eine Tetris-Form irgendwo auf dem Feld platziert werden kann.
        
        Args:
            shape: Liste von (dr, dc) Offsets
            
        Returns:
            True wenn Form platzierbar ist
        """
        for row in range(self.height):
            for col in range(self.width):
                if self._can_place_tetris_shape_at(shape, row, col):
                    return True
        return False
    
    def _can_place_tetris_shape_at(self, shape: List[Tuple[int, int]], center_row: int, center_col: int) -> bool:
        """
        Prüft ob eine Tetris-Form an einer bestimmten Position platziert werden kann.
        
        Args:
            shape: Liste von (dr, dc) Offsets
            center_row: Zentrale Zeile
            center_col: Zentrale Spalte
            
        Returns:
            True wenn Form hier platzierbar ist (keine Minen, nicht aufgedeckt, im Feld)
        """
        for dr, dc in shape:
            r = center_row + dr
            c = center_col + dc
            
            # Prüfe ob Position im Feld ist
            if not (0 <= r < self.height and 0 <= c < self.width):
                return False
            
            cell = self.board.get_cell(r, c)
            if not cell:
                return False
            
            # Zelle muss verdeckt und keine Mine sein
            if cell.is_revealed() or cell.is_mine:
                return False
        
        return True
    
    def place_tetris_shape(self, center_row: int, center_col: int) -> bool:
        """
        Platziert die aktuelle Tetris-Form an der gegebenen Position.
        
        Args:
            center_row: Zentrale Zeile
            center_col: Zentrale Spalte
            
        Returns:
            True wenn Form platziert wurde, False sonst
        """
        if not self.tetris_active or not self.tetris_current_shape:
            return False
        
        # Prüfe ob Form hier platziert werden kann
        if not self._can_place_tetris_shape_at(self.tetris_current_shape, center_row, center_col):
            return False
        
        # Platziere Minen beim ersten Klick (falls noch nicht geschehen)
        if self.first_click:
            self.board.place_mines(center_row, center_col)
            self.first_click = False
        
        # Decke alle Felder der Form auf
        for dr, dc in self.tetris_current_shape:
            r = center_row + dr
            c = center_col + dc
            
            cell = self.board.get_cell(r, c)
            if cell and not cell.is_revealed():
                if cell.reveal():
                    self.revealed_count += 1
                    self.points += self.points_per_cell
                    
                    # Auto-Aufdeckung bei leeren Feldern
                    if cell.adjacent_mines == 0:
                        self._auto_reveal_safe_neighbors(r, c)
        
        # Deaktiviere Tetris-Modus
        self.tetris_active = False
        self.tetris_current_shape = None
        
        # Prüfe Gewinnbedingung
        if self.revealed_count >= self.total_safe_cells:
            self.state = GameState.WON
        
        return True
    
    def new_game(self, difficulty: Optional[str] = None, width: Optional[int] = None, height: Optional[int] = None,
                 enable_challenges: Optional[bool] = None):
        """
        Startet ein neues Spiel.
        
        Args:
            difficulty: Optionaler Schwierigkeitsgrad
            width: Optionale Breite des Spielfelds
            height: Optionale Höhe des Spielfelds
            enable_challenges: Optional - ob Herausforderungen aktiviert sind
        """
        if difficulty:
            self.difficulty = difficulty
        if width:
            self.width = width
        if height:
            self.height = height
        if enable_challenges is not None:
            self.enable_challenges = enable_challenges
        
        self.num_mines = self._get_mine_count(self.difficulty)
        self.board = Board(self.width, self.height, self.num_mines)
        self.state = GameState.PLAYING
        self.first_click = True
        self.revealed_count = 0
        self.flagged_count = 0
        self.total_safe_cells = self.width * self.height - self.num_mines
        self.points = 0
        self.radar_scanned_cells = set()
        self.scanner_result_cells = {}
        self.mystery_cells = set()
        # ANGEPASST: Seltener triggern + respektiert enable_challenges
        self.mystery_next_trigger = random.randint(15, 25) if self.enable_challenges else 999999
        self.mystery_counter = 0
        self.speed_cells = set()
        self.speed_active = False
        self.speed_time_remaining = 0.0
        # ANGEPASST: Seltener triggern + respektiert enable_challenges
        self.speed_next_trigger = random.randint(20, 30) if self.enable_challenges else 999999
        self.speed_counter = 0
        self.tetris_cells = set()
        self.tetris_active = False
        self.tetris_current_shape = None
        # ANGEPASST: Seltener triggern + respektiert enable_challenges
        self.tetris_next_trigger = random.randint(25, 40) if self.enable_challenges else 999999
        self.tetris_counter = 0


