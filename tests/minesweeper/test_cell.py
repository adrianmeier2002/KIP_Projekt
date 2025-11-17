"""Tests for Cell class."""

import unittest
from src.minesweeper.cell import Cell
from src.utils.constants import (
    CELL_HIDDEN, CELL_REVEALED, CELL_FLAGGED
)


class TestCell(unittest.TestCase):
    """Test cases for Cell class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cell = Cell(5, 10)
    
    def test_initialization(self):
        """Test cell initialization."""
        self.assertEqual(self.cell.row, 5)
        self.assertEqual(self.cell.col, 10)
        self.assertFalse(self.cell.is_mine)
        self.assertEqual(self.cell.adjacent_mines, 0)
        self.assertEqual(self.cell.state, CELL_HIDDEN)
        self.assertTrue(self.cell.is_hidden())
    
    def test_set_mine(self):
        """Test setting a mine."""
        self.cell.set_mine()
        self.assertTrue(self.cell.is_mine)
    
    def test_set_adjacent_mines(self):
        """Test setting adjacent mine count."""
        self.cell.set_adjacent_mines(3)
        self.assertEqual(self.cell.adjacent_mines, 3)
    
    def test_reveal(self):
        """Test revealing a cell."""
        # Initially hidden
        self.assertTrue(self.cell.is_hidden())
        
        # Reveal
        result = self.cell.reveal()
        self.assertTrue(result)
        self.assertTrue(self.cell.is_revealed())
        self.assertFalse(self.cell.is_hidden())
        
        # Try to reveal again (should fail)
        result = self.cell.reveal()
        self.assertFalse(result)
    
    def test_flag_toggle(self):
        """Test flagging and unflagging."""
        # Initially hidden
        self.assertTrue(self.cell.is_hidden())
        
        # Flag
        result = self.cell.flag()
        self.assertTrue(result)
        self.assertTrue(self.cell.is_flagged())
        self.assertFalse(self.cell.is_hidden())
        
        # Unflag
        result = self.cell.flag()
        self.assertTrue(result)
        self.assertFalse(self.cell.is_flagged())
        self.assertTrue(self.cell.is_hidden())
    
    def test_flag_revealed_cell(self):
        """Test that revealed cells cannot be flagged."""
        self.cell.reveal()
        result = self.cell.flag()
        self.assertFalse(result)
        self.assertFalse(self.cell.is_flagged())
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.cell)
        self.assertIn("Cell", repr_str)
        self.assertIn("5", repr_str)
        self.assertIn("10", repr_str)


if __name__ == "__main__":
    unittest.main()

