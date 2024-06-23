from PyQt6.QtWidgets import QApplication
from gui import main_page

if __name__ == '__main__':
    app = QApplication([])
    w= main_page.MainWindow()
    w.show()
    app.exec()