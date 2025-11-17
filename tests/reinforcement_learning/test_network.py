"""Tests for DQNNetwork."""

import unittest
import torch
from src.reinforcement_learning.network import DQNNetwork
from src.reinforcement_learning.environment import STATE_CHANNELS
from src.utils.constants import BOARD_WIDTH, BOARD_HEIGHT


class TestDQNNetwork(unittest.TestCase):
    """Test cases for DQNNetwork."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_channels = STATE_CHANNELS
        self.num_actions = BOARD_WIDTH * BOARD_HEIGHT
        self.network = DQNNetwork(self.input_channels, self.num_actions)
    
    def test_initialization(self):
        """Test network initialization."""
        self.assertEqual(self.network.num_actions, self.num_actions)
        self.assertIsNotNone(self.network.conv_stack)
        self.assertEqual(len(self.network.conv_stack), 12)  # 4 * (Conv+BN+ReLU)
        self.assertIsNotNone(self.network.fc1)
        self.assertIsNotNone(self.network.fc2)
        self.assertIsNotNone(self.network.fc_out)
    
    def test_forward_pass_single(self):
        """Test forward pass with single sample."""
        # Create dummy input (batch_size=1, channels=1, height, width)
        batch_size = 1
        x = torch.randn(batch_size, self.input_channels, BOARD_HEIGHT, BOARD_WIDTH)
        
        # Forward pass
        output = self.network(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.num_actions))
        
        # Check output values are finite
        self.assertTrue(torch.all(torch.isfinite(output)))
    
    def test_forward_pass_batch(self):
        """Test forward pass with batch."""
        batch_size = 32
        x = torch.randn(batch_size, self.input_channels, BOARD_HEIGHT, BOARD_WIDTH)
        
        output = self.network(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.num_actions))
        self.assertTrue(torch.all(torch.isfinite(output)))
    
    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 4, 16, 64]:
            x = torch.randn(batch_size, self.input_channels, BOARD_HEIGHT, BOARD_WIDTH)
            output = self.network(x)
            self.assertEqual(output.shape, (batch_size, self.num_actions))
    
    def test_gradient_flow(self):
        """Test that gradients can flow through the network."""
        x = torch.randn(1, self.input_channels, BOARD_HEIGHT, BOARD_WIDTH, requires_grad=True)
        output = self.network(x)
        
        # Create dummy loss
        loss = output.mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(x.grad)
        
        # Check that network parameters have gradients
        for param in self.network.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_network_parameters_count(self):
        """Test that network has reasonable number of parameters."""
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
        # Should have millions of parameters for this network
        self.assertGreater(total_params, 100000)
        self.assertEqual(total_params, trainable_params)
    
    def test_network_eval_mode(self):
        """Test network in eval mode."""
        self.network.eval()
        
        x = torch.randn(1, self.input_channels, BOARD_HEIGHT, BOARD_WIDTH)
        
        with torch.no_grad():
            output = self.network(x)
        
        self.assertEqual(output.shape, (1, self.num_actions))


if __name__ == "__main__":
    unittest.main()

