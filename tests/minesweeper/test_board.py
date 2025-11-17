"""Tests for Board class."""

import unittest
from src.minesweeper.board import Board
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class TestBoard(unittest.TestCase):
    """Test cases for Board class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.board = Board(10, 10, 10)  # 10x10 board with 10 mines
    
    def test_initialization(self):
        """Test board initialization."""
        self.assertEqual(self.board.width, 10)
        self.assertEqual(self.board.height, 10)
        self.assertEqual(self.board.num_mines, 10)
        self.assertEqual(len(self.board.cells), 10)
        self.assertEqual(len(self.board.cells[0]), 10)
        self.assertEqual(len(self.board.mine_positions), 0)  # Mines not placed yet
    
    def test_get_cell_valid(self):
        """Test getting a valid cell."""
        cell = self.board.get_cell(5, 5)
        self.assertIsNotNone(cell)
        self.assertEqual(cell.row, 5)
        self.assertEqual(cell.col, 5)
    
    def test_get_cell_invalid(self):
        """Test getting an invalid cell."""
        cell = self.board.get_cell(-1, 5)
        self.assertIsNone(cell)
        
        cell = self.board.get_cell(15, 5)
        self.assertIsNone(cell)
        
        cell = self.board.get_cell(5, 15)
        self.assertIsNone(cell)
    
    def test_get_neighbors(self):
        """Test getting neighbors of a cell."""
        neighbors = self.board.get_neighbors(5, 5)
        # Middle cell should have 8 neighbors
        self.assertEqual(len(neighbors), 8)
        
        # Corner cell should have 3 neighbors
        neighbors = self.board.get_neighbors(0, 0)
        self.assertEqual(len(neighbors), 3)
        
        # Edge cell should have 5 neighbors
        neighbors = self.board.get_neighbors(0, 5)
        self.assertEqual(len(neighbors), 5)
    
    def test_place_mines(self):
        """Test placing mines on the board."""
        # Place mines excluding first click at (0, 0)
        self.board.place_mines(0, 0)
        
        # Check that mines were placed
        self.assertEqual(len(self.board.mine_positions), 10)
        
        # Check that first clicked cell is not a mine
        first_cell = self.board.get_cell(0, 0)
        self.assertFalse(first_cell.is_mine)
        
        # Verify mine count
        mine_count = sum(
            1 for row in self.board.cells
            for cell in row
            if cell.is_mine
        )
        self.assertEqual(mine_count, 10)
    
    def test_adjacent_mines_calculation(self):
        """Test calculation of adjacent mines."""
        # Create a small board for testing
        test_board = Board(3, 3, 2)
        
        # Manually place mines at (0, 0) and (0, 1)
        test_board.cells[0][0].set_mine()
        test_board.cells[0][1].set_mine()
        test_board.mine_positions = {(0, 0), (0, 1)}
        test_board._calculate_adjacent_mines()
        
        # Cell (0, 2) should have 1 adjacent mine (only (0, 1))
        cell = test_board.get_cell(0, 2)
        self.assertEqual(cell.adjacent_mines, 1)
        
        # Cell (1, 0) should have 2 adjacent mines ((0, 0) and (0, 1))
        cell = test_board.get_cell(1, 0)
        self.assertEqual(cell.adjacent_mines, 2)
        
        # Cell (1, 1) should have 2 adjacent mines ((0, 0) and (0, 1))
        cell = test_board.get_cell(1, 1)
        self.assertEqual(cell.adjacent_mines, 2)
        
        # Cell (2, 2) should have 0 adjacent mines
        cell = test_board.get_cell(2, 2)
        self.assertEqual(cell.adjacent_mines, 0)
        
        # Cell (1, 2) should have 1 adjacent mine (only (0, 1))
        cell = test_board.get_cell(1, 2)
        self.assertEqual(cell.adjacent_mines, 1)
    
    def test_reveal_all_mines(self):
        """Test revealing all mines."""
        self.board.place_mines(0, 0)
        self.board.reveal_all_mines()
        
        # All mine cells should be revealed
        for row, col in self.board.mine_positions:
            cell = self.board.get_cell(row, col)
            self.assertTrue(cell.is_revealed())


if __name__ == "__main__":
    unittest.main()

