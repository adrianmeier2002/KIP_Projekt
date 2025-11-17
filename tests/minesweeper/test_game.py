"""Tests for Game class."""

import unittest
from src.minesweeper.game import Game, GameState
from src.utils.constants import (
    BOARD_WIDTH, BOARD_HEIGHT,
    MINES_EASY, MINES_MEDIUM, MINES_HARD
)


class TestGame(unittest.TestCase):
    """Test cases for Game class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.game = Game("medium")
    
    def test_initialization(self):
        """Test game initialization."""
        self.assertEqual(self.game.difficulty, "medium")
        self.assertEqual(self.game.num_mines, MINES_MEDIUM)
        self.assertEqual(self.game.state, GameState.PLAYING)
        self.assertTrue(self.game.first_click)
        self.assertEqual(self.game.revealed_count, 0)
        self.assertEqual(self.game.flagged_count, 0)
    
    def test_difficulty_levels(self):
        """Test different difficulty levels."""
        game_easy = Game("easy")
        self.assertEqual(game_easy.num_mines, MINES_EASY)
        
        game_medium = Game("medium")
        self.assertEqual(game_medium.num_mines, MINES_MEDIUM)
        
        game_hard = Game("hard")
        self.assertEqual(game_hard.num_mines, MINES_HARD)
    
    def test_first_click_places_mines(self):
        """Test that mines are placed on first click."""
        # Before first click, no mines placed
        self.assertEqual(len(self.game.board.mine_positions), 0)
        
        # First click should place mines
        self.game.reveal_cell(0, 0)
        self.assertGreater(len(self.game.board.mine_positions), 0)
        self.assertFalse(self.game.first_click)
    
    def test_reveal_cell(self):
        """Test revealing a cell."""
        # First click places mines and reveals cell
        result = self.game.reveal_cell(0, 0)
        self.assertTrue(result)
        self.assertGreater(self.game.revealed_count, 0)
        
        # Try to reveal same cell again (should fail)
        result = self.game.reveal_cell(0, 0)
        self.assertFalse(result)
    
    def test_toggle_flag(self):
        """Test flagging and unflagging cells."""
        # Flag a cell
        result = self.game.toggle_flag(5, 5)
        self.assertTrue(result)
        self.assertEqual(self.game.flagged_count, 1)
        
        # Unflag the cell
        result = self.game.toggle_flag(5, 5)
        self.assertTrue(result)
        self.assertEqual(self.game.flagged_count, 0)
    
    def test_flag_prevents_reveal(self):
        """Test that flagged cells cannot be revealed."""
        self.game.toggle_flag(5, 5)
        result = self.game.reveal_cell(5, 5)
        self.assertFalse(result)
    
    def test_auto_reveal_safe_neighbors(self):
        """Test automatic revealing of safe neighbors."""
        # Create a small test game
        test_game = Game("easy")
        
        # First click
        test_game.reveal_cell(0, 0)
        
        # Find a cell with 0 adjacent mines (should auto-reveal neighbors)
        found_zero = False
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                cell = test_game.board.get_cell(row, col)
                if cell.is_revealed() and cell.adjacent_mines == 0:
                    found_zero = True
                    # Check that some neighbors were auto-revealed
                    neighbors = test_game.board.get_neighbors(row, col)
                    revealed_neighbors = sum(1 for n in neighbors if n.is_revealed())
                    self.assertGreater(revealed_neighbors, 0)
                    break
            if found_zero:
                break
    
    def test_game_over_on_mine(self):
        """Test that game ends when a mine is clicked."""
        # We need to find a mine after first click
        self.game.reveal_cell(0, 0)  # First click
        
        # Find and click a mine
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                cell = self.game.board.get_cell(row, col)
                if cell.is_mine and not cell.is_revealed():
                    self.game.reveal_cell(row, col)
                    self.assertTrue(self.game.is_lost())
                    self.assertTrue(self.game.is_game_over())
                    return
    
    def test_get_remaining_mines(self):
        """Test remaining mines calculation."""
        initial_mines = self.game.num_mines
        self.assertEqual(self.game.get_remaining_mines(), initial_mines)
        
        # Flag some cells
        self.game.toggle_flag(0, 0)
        self.game.toggle_flag(0, 1)
        self.assertEqual(self.game.get_remaining_mines(), initial_mines - 2)
    
    def test_new_game(self):
        """Test starting a new game."""
        # Play a bit
        self.game.reveal_cell(0, 0)
        self.game.toggle_flag(5, 5)
        
        # Start new game
        self.game.new_game("hard")
        
        # Check reset
        self.assertEqual(self.game.difficulty, "hard")
        self.assertEqual(self.game.num_mines, MINES_HARD)
        self.assertEqual(self.game.state, GameState.PLAYING)
        self.assertTrue(self.game.first_click)
        self.assertEqual(self.game.revealed_count, 0)
        self.assertEqual(self.game.flagged_count, 0)


if __name__ == "__main__":
    unittest.main()

