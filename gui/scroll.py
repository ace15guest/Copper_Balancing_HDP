from PyQt6.QtWidgets import QApplication, QMainWindow, QScrollArea, QWidget, QHBoxLayout, QCheckBox
from PyQt6.QtGui import QDrag, QPixmap
from PyQt6.QtCore import Qt, pyqtSignal, QMimeData
class DragItem(QCheckBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.original = self.text()
        self.current = self.original
        self.setToolTip(f"Original: {self.original}\n Current: {self.current}")
        self.setContentsMargins(25, 5, 25, 5)
        self.setStyleSheet("border: 1px solid black;")
        self.setAcceptDrops(True)

    def set_data(self, number):
        self.current = number
        self.setToolTip(f"Original: {self.original}\n Current: {self.current}")

    def dragEnterEvent(self, e):
        if e.mimeData().hasImage():
            e.accept()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if e.mimeData().hasImage():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        source_pos = e.source().current
        current_pos = self.current
        self.window().swap.emit(*sorted([source_pos, current_pos]))

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.MouseButton.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            pixmap = QPixmap(self.size())
            mime.setImageData(pixmap)
            drag.setMimeData(mime)
            self.render(pixmap)
            drag.setPixmap(pixmap)
            drag.exec(Qt.DropAction.MoveAction)

