import sys
from pathlib import Path

from UI.pages.page_v2 import MainWindow
from PyQt6.QtWidgets import QApplication

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow(project_root=PROJECT_ROOT)
    window.show()
    sys.exit(app.exec())


