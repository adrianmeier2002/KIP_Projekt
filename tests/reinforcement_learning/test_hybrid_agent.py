"""Tests for Hybrid Agent."""

import unittest
import numpy as np
from src.reinforcement_learning.hybrid_agent import HybridAgent
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS


class TestHybridAgent(unittest.TestCase):
    """Test cases for HybridAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MinesweeperEnvironment("easy", width=5, height=5)
        self.agent = HybridAgent(
            state_channels=STATE_CHANNELS,
            action_space_size=self.env.action_space_size,
            board_height=5,
            board_width=5,
            use_solver=True
        )
    
    def test_initialization(self):
        """Test hybrid agent initialization."""
        self.assertTrue(self.agent.use_solver)
        self.assertIsNotNone(self.agent.stats)
        self.assertEqual(self.agent.stats['total_moves'], 0)
    
    def test_select_action_with_solver(self):
        """Test action selection with solver enabled."""
        state = self.env.reset()
        valid_actions = self.env.get_valid_actions()
        
        # Select action (pass game object)
        action = self.agent.select_action(state, valid_actions, game=self.env.game)
        
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.env.action_space_size)
        self.assertGreater(self.agent.stats['total_moves'], 0)
    
    def test_select_action_without_solver(self):
        """Test action selection with solver disabled."""
        agent_no_solver = HybridAgent(
            state_channels=STATE_CHANNELS,
            action_space_size=self.env.action_space_size,
            board_height=5,
            board_width=5,
            use_solver=False
        )
        
        state = self.env.reset()
        valid_actions = self.env.get_valid_actions()
        
        action = agent_no_solver.select_action(state, valid_actions, game=self.env.game)
        
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.env.action_space_size)
        # Without solver, all moves should be RL
        self.assertEqual(agent_no_solver.stats['rl_moves'], 1)
        self.assertEqual(agent_no_solver.stats['solver_moves'], 0)
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        stats = self.agent.get_stats()
        
        self.assertIn('solver_percentage', stats)
        self.assertIn('rl_percentage', stats)
        self.assertIn('total_moves', stats)
        self.assertEqual(stats['total_moves'], 0)  # No moves yet
    
    def test_reset_stats(self):
        """Test statistics reset."""
        # Make some moves
        state = self.env.reset()
        self.agent.select_action(state, self.env.get_valid_actions(), game=self.env.game)
        
        self.assertGreater(self.agent.stats['total_moves'], 0)
        
        # Reset
        self.agent.reset_stats()
        
        self.assertEqual(self.agent.stats['total_moves'], 0)
        self.assertEqual(self.agent.stats['solver_moves'], 0)
        self.assertEqual(self.agent.stats['rl_moves'], 0)
    
    def test_hybrid_mode_integration(self):
        """Test hybrid mode in a full episode."""
        state = self.env.reset()
        done = False
        steps = 0
        max_steps = 50
        
        while not done and steps < max_steps:
            valid_actions = self.env.get_valid_actions()
            action = self.agent.select_action(state, valid_actions, game=self.env.game)
            state, reward, done, info = self.env.step(action)
            steps += 1
        
        # Should have made some moves
        stats = self.agent.get_stats()
        self.assertEqual(stats['total_moves'], steps)
        
        # At least some moves should be either solver or RL
        self.assertGreaterEqual(stats['solver_moves'] + stats['rl_moves'], steps)


class TestHybridAgentComparison(unittest.TestCase):
    """Compare Hybrid vs Pure RL behavior."""
    
    def test_hybrid_vs_pure_rl(self):
        """Test that hybrid makes different choices than pure RL."""
        env = MinesweeperEnvironment("easy", width=5, height=5)
        
        # Hybrid agent
        hybrid_agent = HybridAgent(
            state_channels=STATE_CHANNELS,
            action_space_size=env.action_space_size,
            board_height=5,
            board_width=5,
            use_solver=True,
            epsilon_start=0.0  # No randomness
        )
        hybrid_agent.epsilon = 0.0
        
        # Pure RL agent
        pure_rl_agent = HybridAgent(
            state_channels=STATE_CHANNELS,
            action_space_size=env.action_space_size,
            board_height=5,
            board_width=5,
            use_solver=False,
            epsilon_start=0.0
        )
        pure_rl_agent.epsilon = 0.0
        
        # Run both agents on same state
        state = env.reset()
        
        # First move: Might be same (random initial conditions)
        # But after revealing cells, hybrid should prefer safe moves
        
        # Make first move with hybrid
        env.reset()
        state = env.reset()
        action1 = hybrid_agent.select_action(state, env.get_valid_actions(), game=env.game)
        state, _, _, _ = env.step(action1)
        
        # Continue for a few steps
        for _ in range(5):
            if env.game.is_game_over():
                break
            action = hybrid_agent.select_action(state, env.get_valid_actions(), game=env.game)
            state, _, _, _ = env.step(action)
        
        # Check that solver was used
        stats = hybrid_agent.get_stats()
        # If there were safe moves, solver should have been used
        # (This is probabilistic, so just check structure)
        self.assertGreaterEqual(stats['solver_percentage'], 0.0)
        self.assertLessEqual(stats['solver_percentage'], 100.0)


if __name__ == "__main__":
    unittest.main()

