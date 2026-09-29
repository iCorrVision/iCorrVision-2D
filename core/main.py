import sys
from PySide6.QtWidgets import QApplication
from core.app_loader import AppLoader

if __name__ == "__main__":
    app = QApplication(sys.argv)
    loader = AppLoader(app)
    loader.start()
    sys.exit(app.exec())
