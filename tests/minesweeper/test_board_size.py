"""Tests for board size functionality."""

import unittest
from src.minesweeper.game import Game
from src.minesweeper.board import Board
from src.utils.constants import BOARD_SIZES, get_mine_count


class TestBoardSize(unittest.TestCase):
    """Test cases for board size functionality."""
    
    def test_game_with_different_sizes(self):
        """Test Game class with different board sizes."""
        # Test small board
        game_small = Game("medium", width=10, height=8)
        self.assertEqual(game_small.width, 10)
        self.assertEqual(game_small.height, 8)
        self.assertEqual(game_small.board.width, 10)
        self.assertEqual(game_small.board.height, 8)
        
        # Test large board
        game_large = Game("medium", width=40, height=30)
        self.assertEqual(game_large.width, 40)
        self.assertEqual(game_large.height, 30)
        self.assertEqual(game_large.board.width, 40)
        self.assertEqual(game_large.board.height, 30)
    
    def test_predefined_board_sizes(self):
        """Test predefined board sizes."""
        for size_name, size_value in BOARD_SIZES.items():
            if size_name == "custom" or size_value is None:
                continue  # Skip custom
            
            width, height = size_value
            game = Game("medium", width=width, height=height)
            self.assertEqual(game.width, width)
            self.assertEqual(game.height, height)
            self.assertEqual(game.board.width, width)
            self.assertEqual(game.board.height, height)
    
    def test_mine_count_calculation(self):
        """Test mine count calculation for different sizes."""
        # Test small board
        mines_small = get_mine_count(10, 8, "medium")
        self.assertGreater(mines_small, 0)
        self.assertLessEqual(mines_small, 10 * 8)
        
        # Test large board
        mines_large = get_mine_count(40, 30, "medium")
        self.assertGreater(mines_large, 0)
        self.assertLessEqual(mines_large, 40 * 30)
        self.assertGreater(mines_large, mines_small)  # More mines for larger board
    
    def test_difficulty_scaling(self):
        """Test that difficulty scales correctly with board size."""
        width, height = 20, 15
        
        mines_easy = get_mine_count(width, height, "easy")
        mines_medium = get_mine_count(width, height, "medium")
        mines_hard = get_mine_count(width, height, "hard")
        
        # Easy should have fewer mines than medium
        self.assertLess(mines_easy, mines_medium)
        # Medium should have fewer mines than hard
        self.assertLess(mines_medium, mines_hard)
    
    def test_new_game_with_size_change(self):
        """Test starting new game with different size."""
        game = Game("medium", width=20, height=15)
        original_width = game.width
        original_height = game.height
        
        # Change to different size
        game.new_game("medium", width=30, height=20)
        self.assertEqual(game.width, 30)
        self.assertEqual(game.height, 20)
        self.assertNotEqual(game.width, original_width)
        self.assertNotEqual(game.height, original_height)
    
    def test_board_cell_access_different_sizes(self):
        """Test accessing cells in boards of different sizes."""
        # Small board
        board_small = Board(10, 8, 5)
        cell = board_small.get_cell(5, 5)
        self.assertIsNotNone(cell)
        self.assertEqual(cell.row, 5)
        self.assertEqual(cell.col, 5)
        
        # Large board
        board_large = Board(40, 30, 50)
        cell = board_large.get_cell(20, 25)
        self.assertIsNotNone(cell)
        self.assertEqual(cell.row, 20)
        self.assertEqual(cell.col, 25)
    
    def test_board_boundary_checks(self):
        """Test boundary checks for different board sizes."""
        # Small board (width=10, height=8, so valid indices are 0-9 for col, 0-7 for row)
        board_small = Board(10, 8, 5)
        self.assertIsNone(board_small.get_cell(-1, 0))
        self.assertIsNone(board_small.get_cell(0, -1))
        self.assertIsNone(board_small.get_cell(8, 0))  # row >= height
        self.assertIsNone(board_small.get_cell(0, 10))  # col >= width
        self.assertIsNone(board_small.get_cell(8, 9))  # Both out of bounds
        
        # Large board (width=40, height=30, so valid indices are 0-39 for col, 0-29 for row)
        board_large = Board(40, 30, 50)
        self.assertIsNone(board_large.get_cell(-1, 0))
        self.assertIsNone(board_large.get_cell(0, -1))
        self.assertIsNone(board_large.get_cell(30, 0))  # row >= height
        self.assertIsNone(board_large.get_cell(0, 40))  # col >= width
        self.assertIsNone(board_large.get_cell(30, 39))  # Both out of bounds
    
    def test_adjacent_mines_calculation_different_sizes(self):
        """Test adjacent mines calculation for different board sizes."""
        # Small board
        board_small = Board(5, 5, 3)
        board_small.cells[0][0].set_mine()
        board_small.cells[0][1].set_mine()
        board_small.mine_positions = {(0, 0), (0, 1)}
        board_small._calculate_adjacent_mines()
        
        # Cell (1, 0) should have 2 adjacent mines
        cell = board_small.get_cell(1, 0)
        self.assertEqual(cell.adjacent_mines, 2)
        
        # Large board
        board_large = Board(20, 20, 10)
        board_large.cells[5][5].set_mine()
        board_large.cells[5][6].set_mine()
        board_large.cells[6][5].set_mine()
        board_large.mine_positions = {(5, 5), (5, 6), (6, 5)}
        board_large._calculate_adjacent_mines()
        
        # Cell (6, 6) should have 3 adjacent mines
        cell = board_large.get_cell(6, 6)
        self.assertEqual(cell.adjacent_mines, 3)
    
    def test_reveal_cell_different_sizes(self):
        """Test revealing cells in games of different sizes."""
        # Small board
        game_small = Game("medium", width=10, height=8)
        result = game_small.reveal_cell(0, 0)
        self.assertTrue(result)
        self.assertTrue(game_small.board.get_cell(0, 0).is_revealed())
        
        # Large board
        game_large = Game("medium", width=40, height=30)
        result = game_large.reveal_cell(20, 15)
        self.assertTrue(result)
        self.assertTrue(game_large.board.get_cell(20, 15).is_revealed())
    
    def test_total_safe_cells_calculation(self):
        """Test total safe cells calculation for different sizes."""
        # Small board
        game_small = Game("medium", width=10, height=8)
        total_cells = 10 * 8
        expected_safe = total_cells - game_small.num_mines
        self.assertEqual(game_small.total_safe_cells, expected_safe)
        
        # Large board
        game_large = Game("medium", width=40, height=30)
        total_cells = 40 * 30
        expected_safe = total_cells - game_large.num_mines
        self.assertEqual(game_large.total_safe_cells, expected_safe)


if __name__ == "__main__":
    unittest.main()

