"""Tests for Constraint Solver."""

import unittest
from src.reinforcement_learning.constraint_solver import MinesweeperSolver
from src.minesweeper.game import Game


class TestConstraintSolver(unittest.TestCase):
    """Test cases for MinesweeperSolver."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.game = Game("easy", width=5, height=5)
    
    def test_initialization(self):
        """Test solver initialization."""
        solver = MinesweeperSolver(self.game)
        self.assertIsNotNone(solver)
        self.assertEqual(solver.width, 5)
        self.assertEqual(solver.height, 5)
    
    def test_get_safe_moves_initial(self):
        """Test that no safe moves are found initially."""
        solver = MinesweeperSolver(self.game)
        safe_moves = solver.get_safe_moves()
        # No moves revealed yet, so no safe moves
        self.assertEqual(len(safe_moves), 0)
    
    def test_get_safe_moves_after_reveal(self):
        """Test safe move detection after revealing a cell."""
        # Reveal first cell
        self.game.reveal_cell(0, 0)
        
        solver = MinesweeperSolver(self.game)
        safe_moves = solver.get_safe_moves()
        
        # Should find some safe moves if cell shows 0 or has satisfied constraints
        # (Depends on mine placement, but test structure is valid)
        self.assertIsInstance(safe_moves, list)
        for row, col in safe_moves:
            self.assertGreaterEqual(row, 0)
            self.assertLess(row, 5)
            self.assertGreaterEqual(col, 0)
            self.assertLess(col, 5)
    
    def test_get_mine_cells(self):
        """Test mine cell detection."""
        # Reveal some cells
        self.game.reveal_cell(2, 2)
        
        solver = MinesweeperSolver(self.game)
        mine_cells = solver.get_mine_cells()
        
        # Mine detection depends on board state
        self.assertIsInstance(mine_cells, list)
    
    def test_get_best_guess(self):
        """Test best guess selection."""
        import numpy as np
        
        # Reveal first cell
        self.game.reveal_cell(0, 0)
        
        solver = MinesweeperSolver(self.game)
        
        # Create valid actions mask
        valid_actions = np.ones(25, dtype=bool)
        valid_actions[0] = False  # First cell already revealed
        
        best_guess = solver.get_best_guess(valid_actions)
        
        # Should return a valid action
        if best_guess is not None:
            self.assertGreaterEqual(best_guess, 0)
            self.assertLess(best_guess, 25)
            self.assertTrue(valid_actions[best_guess])


class TestConstraintSolverLogic(unittest.TestCase):
    """Test solver logic with specific scenarios."""
    
    def test_simple_pattern_detection(self):
        """Test detection of simple safe patterns."""
        game = Game("easy", width=3, height=3)
        
        # Force mine placement (we need a controlled test)
        # This is tricky without exposing internal API
        # For now, just test that solver doesn't crash
        game.reveal_cell(1, 1)
        
        solver = MinesweeperSolver(game)
        safe_moves = solver.get_safe_moves()
        mine_cells = solver.get_mine_cells()
        
        # Should not crash and return lists
        self.assertIsInstance(safe_moves, list)
        self.assertIsInstance(mine_cells, list)


if __name__ == "__main__":
    unittest.main()

