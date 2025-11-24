"""Konstanten für das Minesweeper-Spiel."""

# Standard-Spielfelddimensionen
BOARD_WIDTH = 30
BOARD_HEIGHT = 20

# Vordefinierte Spielfeldgrößen
BOARD_SIZES = {
    "small": (15, 10),      # 15x10 = 150 Zellen
    "medium": (20, 15),     # 20x15 = 300 Zellen
    "large": (30, 20),      # 30x20 = 600 Zellen (Standard)
    "xlarge": (40, 25),     # 40x25 = 1000 Zellen
    "custom": None          # Benutzerdefinierte Größe
}

# Schwierigkeitsgrade (Prozentsatz der Minen)
DIFFICULTY_EASY = 0.10    # ~10% Minen
DIFFICULTY_MEDIUM = 0.15  # ~15% Minen (Standard)
DIFFICULTY_HARD = 0.20    # ~20% Minen

def get_mine_count(width: int, height: int, difficulty: str) -> int:
    """
    Berechnet die Minenanzahl für gegebene Spielfeldgröße und Schwierigkeit.
    
    Args:
        width: Breite des Spielfelds
        height: Höhe des Spielfelds
        difficulty: "easy", "medium" oder "hard"
        
    Returns:
        Anzahl der Minen
    """
    total_cells = width * height
    difficulty_map = {
        "easy": DIFFICULTY_EASY,
        "medium": DIFFICULTY_MEDIUM,
        "hard": DIFFICULTY_HARD
    }
    percentage = difficulty_map.get(difficulty, DIFFICULTY_MEDIUM)
    return int(total_cells * percentage)

# Legacy-Unterstützung (für Rückwärtskompatibilität)
TOTAL_CELLS = BOARD_WIDTH * BOARD_HEIGHT
MINES_EASY = get_mine_count(BOARD_WIDTH, BOARD_HEIGHT, "easy")
MINES_MEDIUM = get_mine_count(BOARD_WIDTH, BOARD_HEIGHT, "medium")
MINES_HARD = get_mine_count(BOARD_WIDTH, BOARD_HEIGHT, "hard")

# Zellzustände
CELL_HIDDEN = 0
CELL_REVEALED = 1
CELL_FLAGGED = 2
CELL_MINE = 3


