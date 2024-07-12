from PyQt6.QtWidgets import QApplication, QMainWindow, QScrollArea, QWidget, QHBoxLayout, QCheckBox, QSizePolicy
from PyQt6.QtGui import QDrag, QPixmap
from PyQt6.QtCore import Qt, pyqtSignal, QMimeData
from PyQt6.QtGui import QCursor, QPalette, QColor




from PyQt6.QtWidgets import QComboBox

class ItemWidget(QWidget):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.text = text
        # Create a combo box
        self.comboBox = QComboBox(self)
        self.comboBox.addItem("1 oz")
        self.comboBox.addItem("2 oz")
        self.comboBox.addItem("3 oz")
        self.layout.addWidget(self.comboBox)
        self.comboBox.setFixedSize(50, 25)

        self.checkBox = QCheckBox(text, self)

        self.checkBox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.checkBox, stretch=1)

        # self.checkBox.setFixedSize(100, 15)
        # self.layout.addWidget(self.checkBox)
        self.setStyleSheet("QCheckBox::indicator:unchecked { background: black; }")

    def set_highlight(self, highlight):
        if highlight:
            palette = self.palette()
            palette.setColor(QPalette.ColorRole.Window, QColor('red'))
            self.setAutoFillBackground(True)
            self.setPalette(palette)
        else:
            self.setAutoFillBackground(False)
            self.setPalette(self.style().standardPalette())

