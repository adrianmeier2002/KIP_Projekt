"""Tests for MinesweeperEnvironment."""

import unittest
import numpy as np
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class TestMinesweeperEnvironment(unittest.TestCase):
    """Test cases for MinesweeperEnvironment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MinesweeperEnvironment("medium", use_flag_actions=False)
        self.env_with_flags = MinesweeperEnvironment("medium", use_flag_actions=True)
    
    def test_initialization(self):
        """Test environment initialization."""
        self.assertEqual(self.env.difficulty, "medium")
        self.assertEqual(self.env.action_space_size, BOARD_WIDTH * BOARD_HEIGHT)
        self.assertIsNotNone(self.env.game)
        self.assertEqual(self.env_with_flags.action_space_size, BOARD_WIDTH * BOARD_HEIGHT * 2)
    
    def test_reset(self):
        """Test environment reset."""
        state = self.env.reset()
        
        # Check state shape
        self.assertEqual(state.shape, (STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH))
        self.assertEqual(state.dtype, np.float32)
        
        # Check that all cells are hidden initially (value should be -0.8 now)
        # Actually, first click hasn't happened yet, so state should be all hidden
        self.assertTrue(np.all(state[0] == -0.8))  # Updated from -0.9 to -0.8
        self.assertTrue(np.all(state[1] == 1.0))
        # Check that frontier channel is all zeros initially (no revealed cells yet)
        self.assertTrue(np.all(state[7] == 0.0))
        # Check that safe cell channel is all zeros initially
        self.assertTrue(np.all(state[8] == 0.0))
    
    def test_step_valid_action(self):
        """Test step with valid action."""
        state = self.env.reset()
        
        # First action (should be valid)
        action = 0  # Top-left cell
        next_state, reward, done, info = self.env.step(action)
        
        # Check return types
        self.assertIsInstance(next_state, np.ndarray)
        self.assertIsInstance(reward, (float, np.floating))
        self.assertIsInstance(done, (bool, np.bool_))
        self.assertIsInstance(info, dict)
        
        # Check state shape
        self.assertEqual(next_state.shape, (STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH))
        
        # Check info keys
        self.assertIn("won", info)
        self.assertIn("lost", info)
        self.assertIn("valid_action", info)
    
    def test_step_invalid_action(self):
        """Test step with invalid action (already revealed cell)."""
        state = self.env.reset()
        
        # First action
        action1 = 0
        next_state, reward1, done1, info1 = self.env.step(action1)
        
        # Try same action again (should be invalid)
        next_state2, reward2, done2, info2 = self.env.step(action1)
        
        # Should get negative reward for invalid action
        self.assertLess(reward2, 0)
        self.assertFalse(info2["valid_action"])
    
    def test_state_representation(self):
        """Test state representation encoding."""
        state = self.env.reset()
        
        # Make some moves
        self.env.step(0)
        state = self.env._get_state()
        
        # Check state values are in expected range
        # -1.0 = mine, -0.9 = hidden, -0.5 = flagged, 0-1 = numbers
        self.assertTrue(np.all(state >= -1.0))
        self.assertTrue(np.all(state <= 1.0))
    
    def test_reward_calculation(self):
        """Test reward calculation."""
        state = self.env.reset()
        
        # First action should give reward if cells are revealed
        _, reward, _, _ = self.env.step(0)
        # Reward could be positive (cells revealed) or negative (mine hit)
        self.assertIsInstance(reward, (float, np.floating))
    
    def test_reward_on_win(self):
        """Test reward when game is won."""
        # This is hard to test deterministically, but we can check the structure
        state = self.env.reset()
        
        # Make moves until game ends (if possible)
        # For now, just check that reward calculation handles win case
        # (This would require solving the entire game, which is complex)
        pass
    
    def test_reward_on_loss(self):
        """Test reward when mine is hit."""
        state = self.env.reset()
        
        # Try to find and hit a mine (unlikely but possible)
        # Loss penalty scales with board size (approx -12 * scale)
        for _ in range(100):  # Try up to 100 actions
            valid_actions = self.env.get_valid_actions()
            if not np.any(valid_actions):
                break
            
            action = np.random.choice(np.where(valid_actions)[0])
            _, reward, done, info = self.env.step(action)
            
            if info["lost"]:
                # Reward is now scaled by board size, so check range instead of exact value
                # For medium difficulty (30x20), should be around -100
                self.assertLess(reward, 0)
                self.assertGreaterEqual(reward, -200.0)  # Allow some flexibility
                break
    
    def test_get_valid_actions(self):
        """Test getting valid actions."""
        state = self.env.reset()
        
        cell_count = BOARD_WIDTH * BOARD_HEIGHT
        valid_reveal_only = self.env.get_valid_actions()
        self.assertEqual(valid_reveal_only.shape, (cell_count,))
        self.assertEqual(valid_reveal_only.dtype, bool)
        self.assertTrue(np.all(valid_reveal_only))
        
        # After first action, some should become invalid
        self.env.step(0)
        valid_reveal_only = self.env.get_valid_actions()
        self.assertFalse(np.all(valid_reveal_only))  # Some reveals invalid now
        
        # With flags enabled
        self.env_with_flags.reset()
        valid_with_flags = self.env_with_flags.get_valid_actions()
        self.assertEqual(valid_with_flags.shape, (cell_count * 2,))
        self.assertTrue(np.all(valid_with_flags[:cell_count]))
        self.assertTrue(np.all(valid_with_flags[cell_count:]))
    
    def test_get_action_mask(self):
        """Test action masking."""
        state = self.env.reset()
        
        mask_no_flags = self.env.get_action_mask()
        self.assertEqual(mask_no_flags.shape, (BOARD_WIDTH * BOARD_HEIGHT,))
        valid_nf = self.env.get_valid_actions()
        self.assertTrue(np.all(mask_no_flags[valid_nf] == 0.0))
        
        mask_flags = self.env_with_flags.get_action_mask()
        self.assertEqual(mask_flags.shape, (BOARD_WIDTH * BOARD_HEIGHT * 2,))
        valid_flags = self.env_with_flags.get_valid_actions()
        self.assertTrue(np.all(mask_flags[valid_flags] == 0.0))
        self.assertTrue(np.all(np.isinf(mask_flags[~valid_flags])))
    
    def test_multiple_episodes(self):
        """Test multiple episodes."""
        # Run multiple episodes
        for _ in range(3):
            state = self.env.reset()
            self.assertFalse(self.env.game.is_game_over())
            
            # Make a few moves
            for _ in range(5):
                if self.env.game.is_game_over():
                    break
                valid = self.env.get_valid_actions()
                if np.any(valid):
                    action = np.random.choice(np.where(valid)[0])
                    self.env.step(action)
    
    def test_flag_rewards(self):
        """Test rewards for flagging correct/incorrect cells."""
        env = self.env_with_flags
        env.reset()
        # Trigger mine placement
        env.step(0)
        
        mine_idx = None
        safe_idx = None
        
        for row in range(env.height):
            for col in range(env.width):
                cell = env.game.board.get_cell(row, col)
                if cell.is_revealed():
                    continue
                idx = row * env.width + col
                if cell.is_mine and mine_idx is None:
                    mine_idx = idx
                elif not cell.is_mine and safe_idx is None:
                    safe_idx = idx
            if mine_idx is not None and safe_idx is not None:
                break
        
        self.assertIsNotNone(mine_idx)
        self.assertIsNotNone(safe_idx)
        
        # Flag a mine -> positive reward
        mine_flag_action = mine_idx + env.cell_count
        _, reward_correct, _, _ = env.step(mine_flag_action)
        self.assertGreater(reward_correct, 0)
        
        # Flag a safe cell -> penalty
        safe_flag_action = safe_idx + env.cell_count
        _, reward_wrong, _, _ = env.step(safe_flag_action)
        self.assertLess(reward_wrong, 0)


if __name__ == "__main__":
    unittest.main()

