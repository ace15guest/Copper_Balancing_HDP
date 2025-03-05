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

        # Create a new check box and add it to the layout before the combo box
        self.inverted_layer = QCheckBox(self)
        self.layout.addWidget(self.inverted_layer)
        self.inverted_layer.setToolTip("Check this layer if the layer is inverted")

        # Create a combo box
        self.comboBox = QComboBox(self)
        self.comboBox.addItem("1 oz")
        self.comboBox.addItem("2 oz")
        self.comboBox.addItem("3 oz")
        self.layout.addWidget(self.comboBox)
        self.comboBox.setFixedSize(55, 25)

        self.selected_layer = QCheckBox(text, self)

        self.selected_layer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.selected_layer)
        self.selected_layer.setToolTip("Check this layer if you want to use this layer in the calculations")

        # self.selected_layer.setFixedSize(100, 15)
        # self.layout.addWidget(self.selected_layer)
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

