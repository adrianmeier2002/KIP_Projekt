"""Haupteinstiegspunkt für die Minesweeper-Anwendung."""

import sys
from PySide6.QtWidgets import QApplication
from src.gui.main_window import MainWindow


def main():
    """Startet die Minesweeper-Anwendung."""
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
