"""Tests for DQNAgent and ReplayBuffer."""

import unittest
import numpy as np
import torch
import os
import tempfile
from src.reinforcement_learning.dqn_agent import DQNAgent, ReplayBuffer
from src.reinforcement_learning.environment import MinesweeperEnvironment, STATE_CHANNELS
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class TestReplayBuffer(unittest.TestCase):
    """Test cases for ReplayBuffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ReplayBuffer(capacity=100)
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT * 2
    
    def _random_state(self):
        return np.random.randn(STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH).astype(np.float32)
    
    def test_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(len(self.buffer), 0)
    
    def test_push(self):
        """Test pushing experiences."""
        state = self._random_state()
        action = 0
        reward = 1.0
        next_state = self._random_state()
        done = False
        next_valid = np.ones(self.action_space_size, dtype=bool)
        
        self.buffer.push(state, action, reward, next_state, done, next_valid)
        self.assertEqual(len(self.buffer), 1)
    
    def test_capacity_limit(self):
        """Test that buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10)
        
        for i in range(15):
            state = self._random_state()
            next_valid = np.ones(self.action_space_size, dtype=bool)
            buffer.push(state, 0, 1.0, state, False, next_valid)
        
        # Should not exceed capacity
        self.assertEqual(len(buffer), 10)
    
    def test_sample(self):
        """Test sampling from buffer."""
        # Add some experiences
        for i in range(32):
            state = self._random_state()
            next_valid = np.ones(self.action_space_size, dtype=bool)
            self.buffer.push(state, i % 10, float(i), state, i % 2 == 0, next_valid)
        
        # Sample batch
        batch_size = 16
        states, actions, rewards, next_states, dones, next_valid_actions = self.buffer.sample(batch_size)
        
        # Check types
        self.assertIsInstance(states, torch.Tensor)
        self.assertIsInstance(actions, torch.Tensor)
        self.assertIsInstance(rewards, torch.Tensor)
        self.assertIsInstance(next_states, torch.Tensor)
        self.assertIsInstance(dones, torch.Tensor)
        self.assertIsInstance(next_valid_actions, torch.Tensor)
        
        # Check shapes
        self.assertEqual(states.shape[0], batch_size)
        self.assertEqual(actions.shape[0], batch_size)
        self.assertEqual(rewards.shape[0], batch_size)
        self.assertEqual(next_states.shape[0], batch_size)
        self.assertEqual(dones.shape[0], batch_size)
        self.assertEqual(next_valid_actions.shape[0], batch_size)


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQNAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.action_space_size = BOARD_WIDTH * BOARD_HEIGHT * 2
        self.agent = DQNAgent(
            state_channels=STATE_CHANNELS,
            action_space_size=self.action_space_size,
            lr=0.001,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=1000,
            batch_size=32,
            target_update=100
        )
        self.state_shape = (STATE_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH)
    
    def _all_valid_actions(self):
        """Helper to generate a fully valid action mask."""
        return np.ones(self.action_space_size, dtype=bool)
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.action_space_size, self.action_space_size)
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertIsNotNone(self.agent.q_network)
        self.assertIsNotNone(self.agent.target_network)
        self.assertIsNotNone(self.agent.optimizer)
        self.assertIsNotNone(self.agent.memory)
    
    def test_select_action_random(self):
        """Test action selection in random mode (epsilon=1)."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        
        # With epsilon=1, should always be random
        actions = [self.agent.select_action(state) for _ in range(10)]
        
        # Should be valid actions
        for action in actions:
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.action_space_size)
    
    def test_select_action_greedy(self):
        """Test action selection in greedy mode (epsilon=0)."""
        self.agent.epsilon = 0.0
        state = np.random.randn(*self.state_shape).astype(np.float32)
        
        action = self.agent.select_action(state)
        
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.action_space_size)
    
    def test_select_action_with_valid_mask(self):
        """Test action selection with valid actions mask."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        valid_actions = np.zeros(self.action_space_size, dtype=bool)
        valid_actions[0:10] = True  # Only first 10 actions are valid
        
        self.agent.epsilon = 0.0  # Greedy mode
        action = self.agent.select_action(state, valid_actions)
        
        # Should be one of the valid actions
        self.assertIn(action, range(10))
    
    def test_remember(self):
        """Test storing experiences."""
        state = np.random.randn(*self.state_shape).astype(np.float32)
        action = 5
        reward = 1.0
        next_state = np.random.randn(*self.state_shape).astype(np.float32)
        done = False
        
        initial_len = len(self.agent.memory)
        self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        self.assertEqual(len(self.agent.memory), initial_len + 1)
    
    def test_train_step_insufficient_samples(self):
        """Test training with insufficient samples."""
        # Should return None if not enough samples
        loss = self.agent.train_step()
        self.assertIsNone(loss)
    
    def test_train_step_sufficient_samples(self):
        """Test training with sufficient samples."""
        # Add enough experiences
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            action = i % self.action_space_size
            reward = np.random.randn()
            next_state = np.random.randn(*self.state_shape).astype(np.float32)
            done = i % 10 == 0
            
            self.agent.remember(state, action, reward, next_state, done, self._all_valid_actions())
        
        # Train step
        loss = self.agent.train_step()
        
        # Should return a loss value
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0.0)  # Loss should be non-negative
    
    def test_epsilon_decay(self):
        """Test epsilon decay."""
        initial_epsilon = self.agent.epsilon
        
        # Train multiple times to trigger epsilon decay
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train to trigger epsilon decay
        self.agent.train_step()
        
        # Epsilon should stay the same until decay is triggered explicitly
        self.assertEqual(self.agent.epsilon, initial_epsilon)
        
        # After calling decay, epsilon should decrease
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_end)
    
    def test_save_and_load(self):
        """Test saving and loading agent."""
        # Add some experiences and train
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        self.agent.train_step()
        initial_epsilon = self.agent.epsilon
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
            tmp_path = tmp.name
        
        try:
            self.agent.save(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Create new agent and load
            new_agent = DQNAgent(
                state_channels=STATE_CHANNELS,
                action_space_size=self.action_space_size
            )
            new_agent.load(tmp_path)
            
            # Check that epsilon was loaded
            self.assertEqual(new_agent.epsilon, initial_epsilon)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_target_network_update(self):
        """Test target network update."""
        initial_update_counter = self.agent.update_counter
        
        # Train multiple times to trigger target update
        for i in range(self.agent.batch_size + 10):
            state = np.random.randn(*self.state_shape).astype(np.float32)
            self.agent.remember(state, 0, 1.0, state, False, self._all_valid_actions())
        
        # Train enough times to trigger target update
        for _ in range(self.agent.target_update + 1):
            if len(self.agent.memory) >= self.agent.batch_size:
                self.agent.train_step()
        
        # Update counter should have increased
        self.assertGreater(self.agent.update_counter, initial_update_counter)


class TestDQNAgentIntegration(unittest.TestCase):
    """Integration tests for DQNAgent with Environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.env = MinesweeperEnvironment("medium")
        self.agent = DQNAgent(
            state_channels=STATE_CHANNELS,
            action_space_size=self.env.action_space_size,
            buffer_size=100,
            batch_size=16
        )
    
    def test_agent_environment_interaction(self):
        """Test interaction between agent and environment."""
        state = self.env.reset()
        
        # Agent selects action
        valid_actions = self.env.get_valid_actions()
        action = self.agent.select_action(state, valid_actions)
        
        # Environment executes action
        next_state, reward, done, info = self.env.step(action)
        next_valid_actions = self.env.get_valid_actions()
        
        # Agent remembers experience
        self.agent.remember(state, action, reward, next_state, done, next_valid_actions)
        
        # Check that experience was stored
        self.assertGreater(len(self.agent.memory), 0)


if __name__ == "__main__":
    unittest.main()

