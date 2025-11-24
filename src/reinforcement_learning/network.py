"""Neuronales Netzwerk für DQN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.reinforcement_learning.environment import STATE_CHANNELS


class DQNNetwork(nn.Module):
    """Deep Q-Network für Minesweeper."""
    
    def __init__(self, input_channels: int = STATE_CHANNELS, num_actions: int = 600, board_height: int = 20, board_width: int = 30):
        """
        Initialisiert das DQN-Netzwerk.
        
        Args:
            input_channels: Anzahl der Eingabekanäle (Features pro Zelle)
            num_actions: Anzahl möglicher Aktionen (alle Zellen)
            board_height: Höhe des Spielfelds
            board_width: Breite des Spielfelds
        """
        super(DQNNetwork, self).__init__()
        self.num_actions = num_actions
        self.board_height = board_height
        self.board_width = board_width
        
        # 4 Convolutional-Blöcke à 128 Filter (inspiriert von sdlee94)
        conv_layers = []
        in_channels = input_channels
        for _ in range(4):
            conv_layers.extend([
                nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ])
            in_channels = 128
        self.conv_stack = nn.Sequential(*conv_layers)
        
        # Adaptive Pooling auf feste Größe (8x8) für unterschiedliche Boards
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        flattened_size = 128 * 8 * 8
        
        # Zwei vollverbundene Schichten mit je 512 Neuronen + Dropout
        self.fc1 = nn.Linear(flattened_size, 512)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.25)
        self.fc_out = nn.Linear(512, num_actions)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward-Pass durch das Netzwerk.
        
        Args:
            x: Eingabe-Tensor
            
        Returns:
            Q-Werte für alle Aktionen
        """
        x = self.conv_stack(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc_out(x)
        return x


