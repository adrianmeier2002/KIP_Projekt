"""Dialog for RL training with visualization."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QComboBox, QFileDialog, QMessageBox, QProgressBar
)
from PySide6.QtCore import Qt


class RLTrainingDialog(QDialog):
    """Dialog for configuring and starting RL training."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RL Training starten")
        self.setMinimumWidth(400)
        
        self.training_thread = None
        self.visualization_callback = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout()
        
        # Episodes
        episodes_layout = QHBoxLayout()
        episodes_layout.addWidget(QLabel("Episoden:"))
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setMinimum(100)
        self.episodes_spin.setMaximum(100000)
        self.episodes_spin.setValue(1000)
        episodes_layout.addWidget(self.episodes_spin)
        episodes_layout.addStretch()
        layout.addLayout(episodes_layout)
        
        # Difficulty
        difficulty_layout = QHBoxLayout()
        difficulty_layout.addWidget(QLabel("Schwierigkeit:"))
        self.difficulty_combo = QComboBox()
        self.difficulty_combo.addItems(["Leicht", "Mittel", "Schwer"])
        self.difficulty_combo.setCurrentText("Mittel")
        difficulty_layout.addWidget(self.difficulty_combo)
        difficulty_layout.addStretch()
        layout.addLayout(difficulty_layout)
        
        # Save path
        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("Speicherort:"))
        self.save_path_edit = QLabel("models/dqn_model.pth")
        self.save_path_edit.setWordWrap(True)
        browse_button = QPushButton("Durchsuchen...")
        browse_button.clicked.connect(self._browse_save_path)
        save_layout.addWidget(self.save_path_edit)
        save_layout.addWidget(browse_button)
        layout.addLayout(save_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(QLabel("Fortschritt:"))
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Bereit zum Starten")
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Training starten")
        self.start_button.clicked.connect(self._start_training)
        self.cancel_button = QPushButton("Abbrechen")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _browse_save_path(self):
        """Browse for save path."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Modell speichern", "models/dqn_model.pth", "PyTorch Model (*.pth)"
        )
        if path:
            self.save_path_edit.setText(path)
    
    def _start_training(self):
        """Start training."""
        episodes = self.episodes_spin.value()
        difficulty_map = {
            "Leicht": "easy",
            "Mittel": "medium",
            "Schwer": "hard"
        }
        difficulty = difficulty_map[self.difficulty_combo.currentText()]
        save_path = self.save_path_edit.text()
        
        self.start_button.setEnabled(False)
        self.cancel_button.setText("Schließen")
        self.progress_bar.setMaximum(episodes)
        self.progress_bar.setValue(0)
        
        # Set callback for visualization
        self.visualization_callback = self._on_visualization_request
        
        # Start training thread
        self.training_thread = RLTrainingThread(
            episodes, difficulty, save_path, 100, self.visualization_callback
        )
        self.training_thread.training_progress.connect(self._on_progress)
        self.training_thread.episode_complete.connect(self._on_episode_complete)
        self.training_thread.visualization_ready.connect(self._on_visualization_ready)
        self.training_thread.training_complete.connect(self._on_training_complete)
        self.training_thread.start()
    
    def _on_progress(self, current, total):
        """Update progress bar."""
        self.progress_bar.setValue(current)
        self.status_label.setText(f"Training... Episode {current}/{total}")
    
    def _on_episode_complete(self, episode, avg_reward, avg_length, win_rate, epsilon):
        """Handle episode completion."""
        self.status_label.setText(
            f"Episode {episode} | Avg Reward: {avg_reward:.2f} | "
            f"Win Rate: {win_rate:.1f}% | Epsilon: {epsilon:.3f}"
        )
    
    def _on_visualization_ready(self, episode):
        """Handle visualization request."""
        # This will be handled by the main window
        pass
    
    def _on_visualization_request(self, agent, episode):
        """Callback for visualization requests."""
        # Emit signal that main window will handle
        self.visualization_ready.emit(episode)
        return agent
    
    def _on_training_complete(self, save_path):
        """Handle training completion."""
        self.status_label.setText(f"Training abgeschlossen! Modell gespeichert: {save_path}")
        self.start_button.setEnabled(True)
        QMessageBox.information(
            self,
            "Training abgeschlossen",
            f"Training erfolgreich beendet!\nModell gespeichert: {save_path}"
        )
    
    def get_visualization_callback(self):
        """Get visualization callback."""
        return self._on_visualization_request

