"""Constants for the Minesweeper game."""

# Default board dimensions
BOARD_WIDTH = 30
BOARD_HEIGHT = 20

# Predefined board sizes
BOARD_SIZES = {
    "small": (15, 10),      # 15x10 = 150 cells
    "medium": (20, 15),     # 20x15 = 300 cells
    "large": (30, 20),      # 30x20 = 600 cells (default)
    "xlarge": (40, 25),     # 40x25 = 1000 cells
    "custom": None          # Custom size
}

# Difficulty levels (percentage of mines)
DIFFICULTY_EASY = 0.10    # ~10% mines
DIFFICULTY_MEDIUM = 0.15  # ~15% mines (standard)
DIFFICULTY_HARD = 0.20    # ~20% mines

def get_mine_count(width: int, height: int, difficulty: str) -> int:
    """
    Calculate mine count for given board size and difficulty.
    
    Args:
        width: Board width
        height: Board height
        difficulty: "easy", "medium", or "hard"
        
    Returns:
        Number of mines
    """
    total_cells = width * height
    difficulty_map = {
        "easy": DIFFICULTY_EASY,
        "medium": DIFFICULTY_MEDIUM,
        "hard": DIFFICULTY_HARD
    }
    percentage = difficulty_map.get(difficulty, DIFFICULTY_MEDIUM)
    return int(total_cells * percentage)

# Legacy support (for backward compatibility)
TOTAL_CELLS = BOARD_WIDTH * BOARD_HEIGHT
MINES_EASY = get_mine_count(BOARD_WIDTH, BOARD_HEIGHT, "easy")
MINES_MEDIUM = get_mine_count(BOARD_WIDTH, BOARD_HEIGHT, "medium")
MINES_HARD = get_mine_count(BOARD_WIDTH, BOARD_HEIGHT, "hard")

# Cell states
CELL_HIDDEN = 0
CELL_REVEALED = 1
CELL_FLAGGED = 2
CELL_MINE = 3


