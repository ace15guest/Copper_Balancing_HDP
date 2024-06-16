from PyQt6 import QtWidgets, QtGui, QtCore
import sys
import time
class LoadingScreen(QtWidgets.QSplashScreen):
    cancel_requested = QtCore.pyqtSignal()
    def __init__(self):
        super().__init__()

        # Set the size of the loading screen
        self.setFixedSize(QtCore.QSize(500, 200))

        # Set the window flags to frameless
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)

        # Set the window opacity
        self.setWindowOpacity(0.9)

        # Create a QProgressBar and a QLabel
        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setMaximum(100)
        self.progressBar.setStyleSheet("""
            QProgressBar {
                border: none;
                text-align: center;
                color: black;
            }
            QProgressBar::chunk {
                background-color: #05B8CC;
            }
        """)
        self.label = QtWidgets.QLabel()
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        # Create a vertical layout and add the QProgressBar and QLabel
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.progressBar)
        layout.addWidget(self.cancel_button)

        # Set the layout
        self.setLayout(layout)

    def set_progress(self, value, message):
        # Update the progress bar value and the label text
        self.progressBar.setValue(value)
        self.label.setText(f"<h2>{message}</h2>")