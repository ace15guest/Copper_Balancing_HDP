import time

from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import QRect, pyqtSignal, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QScrollArea, QWidget, QVBoxLayout, QPushButton, QGroupBox
from gui.loading_bar import LoadingScreen
from gui import scroll
from loading.gerber_load import check_gerber, verify_gerber
import os
import json
from PyQt6 import QtGui
from PyQt6.QtWidgets import QMessageBox
from gui.colorbar import BlendedColorBar
from gui.plotting_canvas import MplCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from loading.gerber_conversions import gerber_to_svg_gerbv, svg_to_tiff_inkscape, svg_to_tiff
from loading.img2array import bitmap_to_array
from calculations.layer_calcs import blur_tiff_manual, blur_tiff_gauss
from calculations.multi_layer import multiple_layers
from file_handling import clear_folder

class MainWindow(QMainWindow):
    swap = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()

        self.sigma = 1
        self.blur_x = 2
        self.blur_y = 2
        self.run_verification = True  # Run the verification on input files
        self.blur = 'gauss'  # The type of blur to apply to the tiff files

        self.temp_folder = "Assets/temp"
        self.temp_svg_folder = "Assets/temp_svg"
        self.temp_error_folder = "Assets/temp_error"
        self.temp_tiff_folder = "Assets/temp_tiff"
        clear_folder(self.temp_folder)
        clear_folder(self.temp_svg_folder)
        clear_folder(self.temp_error_folder)
        clear_folder(self.temp_tiff_folder)


        self.items = []

        self.setupUi()

        self.show()

    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1605, 900)
        self.centralwidget = QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.central_layout = QVBoxLayout(self.centralwidget)
        self.central_layout.setGeometry(QRect(100, 100, 85, 94))
        # Place the gerber folder input
        self.place_gerber_folder()
        self.place_group_box()
        self.place_scroll_widget()
        self.place_color_buttons()
        self.place_plotting_canvas()
        self.place_buttons_below_scroll()

    #####################
    ### Place Widgets ###
    #####################

    def place_group_box(self):
        self.right_groupbox = QGroupBox(self.centralwidget)
        self.right_layout = QVBoxLayout(self.right_groupbox)

        self.right_groupbox.setGeometry(500, 50, 1100, 850)
        self.right_layout.addWidget(self.right_groupbox)

    def place_scroll_widget(self):
        self.file_scrollArea = QScrollArea(self.centralwidget)
        self.file_scrollArea.setGeometry(QRect(250, 80, 200, 601))
        self.file_scrollArea.setWidgetResizable(True)
        self.file_scrollArea.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOn)  # Set horizontal scroll bar
        self.file_scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.file_scrollAreaWidgetContents = QWidget()
        self.file_scrollArea.setWidget(self.file_scrollAreaWidgetContents)
        self.scroll_areaLayout = QVBoxLayout(self.file_scrollAreaWidgetContents)
        self.file_scrollAreaWidgetContents.setLayout(self.scroll_areaLayout)
        self.file_scrollArea.horizontalScrollBar().setMaximum(5000)

    def place_buttons_below_scroll(self):
        x = 250
        self.moveUpButton = QPushButton('Move Up', self)
        self.moveUpButton.clicked.connect(self.moveSelectedItemUp)
        self.moveUpButton.setGeometry(QRect(x, 680, 200, 50))

        self.moveDownButton = QPushButton('Move Down', self)
        self.moveDownButton.clicked.connect(self.moveSelectedItemDown)
        self.moveDownButton.setGeometry(QRect(x, 730, 200, 50))

        self.moveDownButton = QPushButton('Save Order', self)
        self.moveDownButton.clicked.connect(self.save_folder_order)
        self.moveDownButton.setGeometry(QRect(x, 780, 200, 50))

        self.submit_button = QPushButton('Submit', self)
        self.submit_button.clicked.connect(lambda x: self.start_loading(self.submit_button_clicked))
        self.submit_button.setGeometry(QRect(x, 830, 200, 50))

    def place_color_buttons(self):
        self.color_info = {}
        color_labels = [QtWidgets.QLabel(self), QtWidgets.QLineEdit(self), QtWidgets.QLineEdit(self),
                        QtWidgets.QLineEdit(self), QtWidgets.QLineEdit(self),
                        QtWidgets.QLineEdit(self), QtWidgets.QLineEdit(self),
                        QtWidgets.QLineEdit(self), QtWidgets.QLineEdit(self),
                        QtWidgets.QLabel(self)]
        qt_checkboxes = [QtWidgets.QCheckBox(self) for _ in range(10)]
        colors = ['#FF0000', '#FF4040', '#FF8080', '#FFBFBF', '#FFFFFF', '#CCCCFF', '#9999FF', '#6666FF', '#3333FF',
                  '#0000FF']
        y_loc = 0
        for i, _ in enumerate(color_labels):
            y_start = 100
            self.color_info[i] = {}
            if i == 0:
                color_labels[i].setText("100%")
            elif i == len(color_labels) - 1:
                color_labels[i].setText("0%")

            qt_checkboxes[i].setGeometry(QtCore.QRect(130, y_start + 6 + i * 30, 71, 21))
            color_labels[i].setGeometry(QtCore.QRect(20, y_start + 4 + i * 30, 51, 21))
            self.color_info[i]['color_button'] = QtWidgets.QPushButton(parent=self)
            self.color_info[i]['color_button'].setGeometry(QtCore.QRect(90, y_start + i * 30, 30, 30))
            self.color_info[i]['color_button'].clicked.connect(lambda x, i=i: self.on_color_box_clicked(i))
            self.color_info[i]['color_button'].setStyleSheet(f"background-color: {colors[i]}")
            y_loc = y_start + 6 + i * 30

        y_loc += 30

        self.color_pallete_options_combobox = QtWidgets.QComboBox(self)
        self.color_pallete_options_combobox.setGeometry(QtCore.QRect(10, y_loc, 191, 31))

        self.recolor_button = QtWidgets.QPushButton(parent=self, text="Recolor")
        self.recolor_button.setGeometry(QtCore.QRect(10, y_loc + 30, 191, 31))
        self.recolor_button.clicked.connect(self.recolor_button_clicked)

        self.newname_entry = QtWidgets.QLineEdit(self)
        self.newname_entry.setGeometry(QtCore.QRect(10, y_loc + 60, 191, 31))
        self.newname_entry.setPlaceholderText("New Color Pallette Name")

        self.save_color_palette_button = QtWidgets.QPushButton(parent=self, text="Save Color Pallette")
        self.save_color_palette_button.setGeometry(QtCore.QRect(10, y_loc + 90, 191, 31))
        self.save_color_palette_button.clicked.connect(self.save_color_palette_button_clicked)

        self.delete_color_palette_button = QtWidgets.QPushButton(parent=self, text="Delete Color Pallette")
        self.delete_color_palette_button.setGeometry(QtCore.QRect(10, y_loc + 120, 191, 31))
        self.delete_color_palette_button.clicked.connect(self.delete_color_palette_button_clicked)

        self.load_color_palette()

    def place_gerber_folder(self):
        self.gerber_folder_line_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.gerber_folder_line_edit.setGeometry(QtCore.QRect(10, 10, 531, 21))
        self.gerber_folder_line_edit.setObjectName("gerber_folder_line_edit")
        self.gerber_folder_line_edit.setPlaceholderText("Gerber Folder Path")

        # Place the gerber folder button
        self.gerber_folder_button = QtWidgets.QPushButton(parent=self.centralwidget, text="Gerber Folder")  # Create the button
        self.gerber_folder_button.clicked.connect(lambda x: self.start_loading(self.gerber_folder_button_clicked))  # Connect the button to the function
        self.gerber_folder_button.setGeometry(QtCore.QRect(550, 10, 111, 21))  # Set the geometry of the button

    def place_plotting_canvas(self):
        self.canvas = MplCanvas(self.right_groupbox, width=5, height=4, dpi=100)
        self.right_layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.right_layout.addWidget(self.toolbar)

    ######################
    ### Button Actions ###
    ######################

    def submit_button_clicked(self):
        self.files_chosen = []
        for item in self.items:
            if item.checkBox.isChecked():
                self.files_chosen.append(item.checkBox.text())
        self.loading_screen.progressBar.setMaximum(len(self.files_chosen))

        self.loading_data = {}
        self.data_paths_to_convert = {}
        for idx, file in enumerate(self.files_chosen):
            self.loading_screen.set_progress(idx, f"Converting to vectorized format... {file}")
            gerber_to_pdf
            # svg_name, error_log = gerber_to_svg_gerbv(os.path.join(self.folder_name, file), file)
            # self.data_paths_to_convert[svg_name] = {}
            #
            # self.data_paths_to_convert[svg_name]['error_log'] = error_log
            # self.data_paths_to_convert[svg_name]['file name'] = file

        # files_made = False
        # while not files_made:
        #     if len(os.listdir(self.temp_svg_folder)) >= len(self.files_chosen):
        #         files_made = True
        #     else:
        #         time.sleep(3) # Wait for the files to be made 3 seconds because errors occur if we do not give ample time
        #
        #
        # for idx, file in enumerate(self.data_paths_to_convert.keys()):
        #     error_log = self.data_paths_to_convert[file]['error_log']
        #
        #     file_name = self.data_paths_to_convert[file]['file name']
        #     tiff_path = f"Assets/temp_tiff/{file_name}.png"
        #
        #     # svg_to_tiff_inkscape(file, tiff_path, height=100, width=100, error_log_path=error_log)
        #     svg_to_tiff(file, tiff_path, error_log_path=error_log)
        #     if self.blur == 'manual':
        #         self.loading_data[file] = blur_tiff_manual(bitmap_to_array(tiff_path), blur_x=self.blur_x, blur_y=self.blur_y)
        #     elif self.blur == 'gauss':
        #         self.loading_data[file] = blur_tiff_gauss(bitmap_to_array(tiff_path), sigma=self.sigma)
        #     pass
        #
        # self.out_data = multiple_layers(self.loading_data)
        self.loading_screen.close()

        pass

    # noinspection PyUnboundLocalVariable
    def gerber_folder_button_clicked(self):
        # options |= QtWidgets.QFileDialog.DontUseNativeDialog

        if self.folder_name:
            self.loading_screen.progressBar.setMaximum(len(os.listdir(self.folder_name) * 2))

            self.gerber_folder_line_edit.setText(self.folder_name)

            self.clear_layout(self.scroll_areaLayout)
            self.items = []
            self.selected_item = None
            max_width = 0
            check_names = []
            for idx, i in enumerate(os.listdir(self.folder_name)):
                if self.run_verification:  # Always run verification
                    check_names.append((check_gerber(os.path.join(self.folder_name, i)), i))
                self.loading_screen.set_progress(idx, f"Please Wait Loading Files... {i}%")

            waiting = True
            while waiting:
                for idx, file in enumerate(check_names):
                    try:
                        if verify_gerber(file[0]):
                            item = scroll.ItemWidget(f"{file[1]}")
                            self.scroll_areaLayout.addWidget(item)
                            item.checkBox.stateChanged.connect(lambda state, it=item: self.selectItem(it))
                            max_width = max(max_width, item.sizeHint().width())
                            self.items.append(item)
                        else:
                            print(f"Failed to verify {file[0]}")
                        check_names.remove(file)
                        self.loading_screen.set_progress(self.loading_screen.progressBar.value() + idx,
                                                         f"Verifying Files... {file[1]}%")

                    except:
                        pass

                if len(check_names) == 0:
                    waiting = False

            # Set the maximum value of the horizontal scroll bar
            self.file_scrollArea.horizontalScrollBar().setMaximum(max_width + 100)
        self.loading_screen.close()
        self.loading_screen.destroy()

    def selectItem(self, item):
        if self.selected_item is not None:
            self.selected_item.set_highlight(False)
        self.selected_item = item
        self.selected_item.set_highlight(True)

    def start_loading(self, func):
        if func == self.gerber_folder_button_clicked:
            self.folder_name = QtWidgets.QFileDialog.getExistingDirectory(self.centralwidget, "Select Folder",
                                                                          directory=r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\gerbers\Example1")
        elif func == self.submit_button_clicked:
            pass

        self.loading_screen = LoadingScreen()

        # Show the loading screen
        self.loading_screen.show()
        self.loading_screen.raise_()

        # Create a QThread
        self.thread = QtCore.QThread(self)

        # Move the loading screen to the QThread
        self.loading_screen.moveToThread(self.thread)

        # Start the QThread
        self.thread.started.connect(func)
        self.thread.start()

    def stop_loading(self):
        # Stop the QThread
        print('Hello')
        self.thread.terminate()

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def moveSelectedItemUp(self):
        if self.selected_item:
            index = self.items.index(self.selected_item)
            if index > 0:
                self.items[index], self.items[index - 1] = self.items[index - 1], self.items[index]
                self.updateLayout()

    def moveSelectedItemDown(self):
        if self.selected_item:
            index = self.items.index(self.selected_item)
            if index < len(self.items) - 1:
                self.items[index], self.items[index + 1] = self.items[index + 1], self.items[index]
                self.updateLayout()

    def updateLayout(self):
        for i in reversed(range(self.scroll_areaLayout.count())):
            self.scroll_areaLayout.itemAt(i).widget().setParent(None)
        for item in self.items:
            self.scroll_areaLayout.addWidget(item)

    def save_folder_order(self):
        for idx, item in enumerate(self.items):
            print(f"{idx + 1}: {item.checkBox.text(), item.comboBox.currentText()}")

    def on_color_box_clicked(self, i):
        button = self.color_info[i]['color_button']
        # button = self.mainwidget.sender()
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            button.setStyleSheet(f"background-color: {color.name()}")

    def recolor_button_clicked(self):
        pass

    def save_color_palette_button_clicked(self):
        if self.newname_entry.text() == "":
            show_error_message("Please enter a name for the color palette")
            return
        if os.path.exists('Assets/colors'):
            with open('Assets/colors', 'r') as file:
                color_palettes = json.load(file)
        else:
            color_palettes = {}

        color_palettes[self.newname_entry.text()] = {}
        for i, info in self.color_info.items():
            color = info['color_button'].palette().window().color().name()
            color_palettes[self.newname_entry.text()][i] = color
        if len(color_palettes) > 10:
            show_error_message("Cannot have more than 10 colors in a color palette")
            color_palettes = {list(color_palettes.keys())[i]: color_palettes[list(color_palettes.keys())[i]] for i in
                              range(10)}
        with open('Assets/colors', 'w') as file:
            json.dump(color_palettes, file)  # Write the color palettes to the file
        self.load_color_palette()

    def delete_color_palette_button_clicked(self):

        if os.path.exists('Assets/colors'):
            with open('Assets/colors', 'r') as file:
                color_palettes = json.load(file)
                if color_palettes.get(self.color_pallete_options_combobox.currentText()) is None:
                    show_error_message(
                        "No saved color palettes exist. Please save a color palette first. Up to 10 color palettes can be saved.")
                    return
            del color_palettes[self.color_pallete_options_combobox.currentText()]
            with open('Assets/colors', 'w') as file:
                json.dump(color_palettes, file)  # Write the color palettes to the file
            self.load_color_palette()
        else:
            return

    def load_color_palette(self):
        with open('Assets/colors', 'r') as file:
            color_palettes = json.load(file)
        self.color_pallete_options_combobox.clear()
        for name in color_palettes.keys():
            self.color_pallete_options_combobox.addItem(name)


def show_error_message(message):
    error_dialog = QMessageBox()
    error_dialog.setIcon(QMessageBox.Icon.Critical)
    error_dialog.setText(message)
    error_dialog.setWindowTitle("Error")
    error_dialog.exec()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
