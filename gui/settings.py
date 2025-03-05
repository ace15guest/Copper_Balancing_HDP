# Assisted by watsonx Code Assistant
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QCheckBox, QMainWindow, QSpinBox, QMessageBox
from loading import app_settings_ini

class SettingsPage(QWidget):
    def __init__(self, config):
        super().__init__()
        self.setWindowTitle('Settings')
        self.config = config
        self.initUI()
    def savesettings(self):
        try:
            self.config["Algorithm"]["blurring"] = self.blurring_combobox.currentText()
            self.config["Algorithm"]["kernel size"] = str(self.kernel_pixels_size_spinbox.value())
            self.config["Algorithm"]["gauss sigma"] = str(self.gauss_sigma_combobox.value()/10)
            self.config["Algorithm"]["dpi"] = str(self.dpi_combobox.value())
            app_settings_ini.write_to_config(self.config)
            msg_box = QMessageBox()
            msg_box.setText("The Settings have been successfully saved!")
            msg_box.setWindowTitle("Saved")
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        except:
            msg_box = QMessageBox()
            msg_box.setText("The settings could not be saved properly. Please ensure all values are formatted properly")
            msg_box.setWindowTitle("Saved Settings Error")
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            result = msg_box.exec()
        result = msg_box.exec()
        pass

    def initUI(self):
        vbox = QVBoxLayout()
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()
        hbox4 = QHBoxLayout()
        hbox5 = QHBoxLayout()

        label1 = QLabel("Blurring Algorithm:")
        label2 = QLabel("Kernel Pixels From Center:")
        label3 = QLabel("Sigma (Gauss) / 10:")
        label4 = QLabel("DPI: ")


        self.blurring_combobox = QComboBox()
        self.blurring_combobox.addItems(["gauss", "box", "median", "bilateral", "MetAve"])
        self.blurring_combobox.setCurrentText(self.config["Algorithm"]["blurring"])

        self.kernel_pixels_size_spinbox = QSpinBox()
        self.kernel_pixels_size_spinbox.setRange(1,400)

        self.kernel_pixels_size_spinbox.setValue(int(self.config["Algorithm"]["kernel size"]))


        self.gauss_sigma_combobox = QSpinBox() # Only accepts integers
        self.gauss_sigma_combobox.setRange(1, 100)
        self.gauss_sigma_combobox.setSingleStep(1)
        self.gauss_sigma_combobox.setValue(int(float(self.config["Algorithm"]["gauss sigma"])*10))


        self.dpi_combobox = QSpinBox()
        self.dpi_combobox.setRange(100, 2000)
        self.dpi_combobox.setSingleStep(25)
        self.dpi_combobox.setValue(int(float(self.config["Algorithm"]["dpi"])))
        # self.line_spacing_spinbox = QLineEdit()
        # self.line_spacing_spinbox.setText("1.5")
        #
        # self.tab_size_spinbox = QLineEdit()
        # self.tab_size_spinbox.setText("4")

        self.enable_auto_save_button = QPushButton("Save", self)
        self.enable_auto_save_button.clicked.connect(self.savesettings)


        hbox1.addWidget(label1)
        hbox1.addWidget(self.blurring_combobox)

        hbox2.addWidget(label2)
        hbox2.addWidget(self.kernel_pixels_size_spinbox)

        hbox3.addWidget(label3)
        hbox3.addWidget(self.gauss_sigma_combobox)

        hbox4.addWidget(label4)
        hbox4.addWidget(self.dpi_combobox)

        hbox5.addWidget(self.enable_auto_save_button)

        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox4)
        vbox.addLayout(hbox5)

        self.setLayout(vbox)

        self.setWindowTitle("Settings")
        self.setGeometry(500, 500, 300, 200)
        self.show()

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     ex = SettingsPage()
#     sys.exit(app.exec())
