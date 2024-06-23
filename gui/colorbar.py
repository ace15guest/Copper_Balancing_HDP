import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt6.QtGui import QPainter, QColor, QLinearGradient, QFontMetrics
from PyQt6.QtCore import Qt, QRect

class BlendedColorBar(QWidget):
    def __init__(self, colors, labels, parent=None):
        super().__init__(parent)
        self.colors = colors
        self.labels = labels

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()

        gradient = QLinearGradient(0, 0, 0, rect.height())

        num_colors = len(self.colors)
        for i, color in enumerate(self.colors):
            gradient.setColorAt(i / (num_colors - 1), QColor(color))

        painter.fillRect(QRect(0, 0, rect.width() - 30, rect.height()), gradient)

        # Draw labels
        painter.setPen(Qt.GlobalColor.black)
        font_metrics = QFontMetrics(painter.font())
        label_height = rect.height() / (num_colors - 1)

        for i, label in enumerate(self.labels):
            text_rect = font_metrics.boundingRect(label)
            text_x = rect.width() - text_rect.width() - 5
            text_y = int(i * label_height + label_height / 2 + text_rect.height() / 4)
            painter.drawText(text_x, text_y, label)

        painter.end()