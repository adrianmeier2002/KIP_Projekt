"""Main entry point for the Minesweeper application."""

import sys
from PySide6.QtWidgets import QApplication
from src.gui.main_window import MainWindow


def main():
    """Run the Minesweeper application."""
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
