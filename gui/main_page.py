import time

import numpy as np
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import QRect, pyqtSignal, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QMainWindow, QScrollArea, QWidget, QVBoxLayout, QPushButton, QGroupBox
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PyQt6.QtWidgets import QSizePolicy
from gui.loading_bar import LoadingScreen
from gui import scroll
from loading.gerber_load import check_gerber, verify_gerber
import os
import json
import matplotlib.colors as c

from PyQt6 import QtGui
from PyQt6.QtWidgets import QMessageBox
from gui.colorbar import BlendedColorBar
from gui.plotting_canvas import MplCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from loading.gerber_conversions import gerber_to_svg_gerbv, svg_to_tiff_inkscape, svg_to_tiff, gerber_to_pdf_gerbv, pdf_page_to_array, gerber_to_png_gerbv, check_tiff_dimensions
from loading.img2array import bitmap_to_array
import matplotlib.pyplot as plt
from calculations.layer_calcs import blur_tiff_manual, blur_tiff_gauss
from calculations.multi_layer import multiple_layers
from file_handling import clear_folder


class MainWindow(QMainWindow):
    swap = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()

        self.load_file = False
        self.sigma = 2
        self.blur_x = 5
        self.blur_y = 5
        self.dpi_val = 100
        self.run_verification = True  # Run the verification on input files
        self.blur = 'gauss'  # The type of blur to apply to the tiff files

        self.temp_folder = "Assets/temp"
        self.temp_svg_folder = "Assets/temp_svg"
        self.temp_error_folder = "Assets/temp_error"
        self.temp_tiff_folder = "Assets/temp_tiff"
        self.temp_pdf_folder = "Assets/temp_pdf"

        clear_folder(self.temp_folder)
        clear_folder(self.temp_svg_folder)
        clear_folder(self.temp_error_folder)
        clear_folder(self.temp_tiff_folder)
        clear_folder(self.temp_pdf_folder)

        self.items = []

        self.setupUi()

        self.show()

    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1605, 900)
        self.centralwidget = QWidget(self)

        # Create a scroll area and set its widget to the central widget
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)  # Make the scroll area resizable
        self.scrollArea.setWidget(self.centralwidget)  # Set the central widget as the scroll area's widget

        # Set the scroll area as the central widget of the main window
        self.setCentralWidget(self.scrollArea)

        # Set scroll policies
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)


        # self.setCentralWidget(self.centralwidget)
        self.central_layout = QVBoxLayout(self.centralwidget)
        self.central_layout.setGeometry(QRect(100, 100, 85, 94))
        # self.add_menu_bar()
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
    def add_menu_bar(self):
        self.menuBar().setNativeMenuBar(True)
        self.file_menu = self.menuBar().addMenu("File")
        self.edit_menu = self.menuBar().addMenu("Edit")
        self.view_menu = self.menuBar().addMenu("View")
        self.help_menu = self.menuBar().addMenu("Help")

        # fileMenu = self.menuBar.addMenu('&File')
        # # Add actions to "File" menu
        # openAction = QAction('&Open', self)
        # openAction.triggered.connect(self.openFile)  # Assuming openFile is a method for opening files
        # fileMenu.addAction(openAction)

    def place_group_box(self):
        self.right_groupbox = QGroupBox(self.centralwidget)
        self.right_layout = QVBoxLayout(self.right_groupbox)

        self.right_groupbox.setGeometry(500, 50, 1100, 850)
        self.right_layout.addWidget(self.right_groupbox)

    def place_scroll_widget(self):
        self.file_scrollArea = QScrollArea(self.centralwidget)
        self.file_scrollArea.setGeometry(QRect(205, 50, 290, 600))
        self.file_scrollArea.setWidgetResizable(True)
        self.file_scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # Set horizontal scroll bar
        self.file_scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.file_scrollAreaWidgetContents = QWidget()
        self.file_scrollArea.setWidget(self.file_scrollAreaWidgetContents)
        self.scroll_areaLayout = QVBoxLayout(self.file_scrollAreaWidgetContents)

        # Adding labels with long text
        # for i in range(5):  # Assuming you want to add 5 labels
        #     long_text = "This is a very long label text " * 10  # Repeat the text to make it long
        #     label =QtWidgets.QLabel(long_text, self.file_scrollAreaWidgetContents)
        #     self.scroll_areaLayout.addWidget(label)

    # def place_scroll_widget(self):
    #     self.file_scrollArea = QScrollArea(self.centralwidget)
    #     self.file_scrollArea.setGeometry(QRect(250, 80, 200, 601))
    #     self.file_scrollArea.setWidgetResizable(True)
    #     self.file_scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # Set horizontal scroll bar
    #     self.file_scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    #     self.file_scrollAreaWidgetContents = QWidget()
    #     self.file_scrollArea.setWidget(self.file_scrollAreaWidgetContents)
    #     self.scroll_areaLayout = QVBoxLayout(self.file_scrollAreaWidgetContents)
    #     # self.file_scrollAreaWidgetContents.setLayout(self.scroll_areaLayout)
    #     # self.file_scrollArea.horizontalScrollBar().setMaximum(5000)

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

        # Step 1 & 2: Add the button and connect it
        self.checkAllButton = QPushButton('Check All', self)
        self.checkAllButton.setGeometry(QRect(x, 875, 200, 25))  # Adjust the position as needed
        self.checkAllButton.clicked.connect(self.checkAllFiles)

        # Step 3 & 4: Implement the method to check all files

    def checkAllFiles(self):
        for item in self.items:
            item.selected_layer.setChecked(True)

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
                qt_checkboxes[i].setChecked(True)
                qt_checkboxes[i].setEnabled(False)
            elif i == len(color_labels) - 1:
                color_labels[i].setText("0%")
                qt_checkboxes[i].setChecked(True)
                qt_checkboxes[i].setEnabled(False)
            qt_checkboxes[i].setGeometry(QtCore.QRect(130, y_start + 6 + i * 30, 71, 21))
            qt_checkboxes[i].setChecked(True)
            if i != 0 and i != 9:
                color_labels[i].setText(f"{100 - i * 10}")
            color_labels[i].setGeometry(QtCore.QRect(20, y_start + 4 + i * 30, 51, 21))
            self.color_info[i]['color_button'] = QtWidgets.QPushButton(parent=self)
            self.color_info[i]['color_button'].setGeometry(QtCore.QRect(90, y_start + i * 30, 30, 30))
            self.color_info[i]['color_button'].clicked.connect(lambda x, i=i: self.on_color_box_clicked(i))
            self.color_info[i]['color_button'].setStyleSheet(f"background-color: {colors[i]}")

            self.color_info[i]['check_box'] = qt_checkboxes[i]
            self.color_info[i]['entry_box'] = color_labels[i]
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
        self.canvas = MplCanvas(self.right_groupbox, width=5, height=4, dpi=50)
        self.right_layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.toolbar.setMovable(True)  # Make the toolbar movable
        # self.toolbar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.right_layout.addWidget(self.toolbar)

    ######################
    ### Button Actions ###
    ######################

    def submit_button_clicked(self, outline_file=None):
        self.files_chosen = {}
        self.files_inversed_tracking = {}

        for item in self.items:
            if item.selected_layer.isChecked():
                self.files_chosen[item.selected_layer.text()] = {}
                self.files_chosen[item.selected_layer.text()]['Cu Weight'] = item.comboBox.currentText()
                self.files_chosen[item.selected_layer.text()]['Inverted'] = item.inverted_layer.isChecked()
        self.loading_screen.progressBar.setMaximum(len(self.files_chosen))

        self.loading_data = {}
        self.data_paths_to_convert = {}
        self.arrays = {}


        file_ct = 0

        for idx, file in enumerate(self.files_chosen):
            self.loading_screen.set_progress(idx, f"Converting to vectorized format... {file}")
            tiff_name = os.path.join(self.temp_tiff_folder, file)

            gerber_to_png_gerbv(os.path.join(self.folder_name, file), self.temp_tiff_folder, tiff_name, dpi=self.dpi_val, scale=1, error_log_path=os.path.join(self.temp_error_folder, file), outline_file=outline_file)
            file_ct += 1

        while len(os.listdir(self.temp_tiff_folder)) < file_ct:
            time.sleep(5)

        all_same_size = check_tiff_dimensions(self.temp_tiff_folder)
        if not all_same_size:
            self.loading_screen.close()
            self.loading_screen.destroy()
            show_error_message("The Files must be the same size. Please indicate the outline file")
            select_file = QtWidgets.QFileDialog.getOpenFileName(self.centralwidget, "Select Outline File", directory=fr'{os.getcwd()}\Assets\gerbers\GaryExample')
            if select_file[0] == "":
                return
            outline_file = select_file[0]
            for file in os.listdir(self.temp_tiff_folder):
                os.remove(os.path.join(self.temp_tiff_folder, file))
            self.submit_button_clicked(outline_file)
            return

        for idx, file in enumerate(os.listdir(self.temp_tiff_folder)):
            inverted = self.files_chosen[file.replace('.tif', '')]['Inverted']
            self.arrays[file] = bitmap_to_array(os.path.join(self.temp_tiff_folder, file), inverted)
            self.loading_screen.set_progress(idx, f"Converting to array... {file}")



        data = multiple_layers(self.arrays)


        data = blur_tiff_gauss(data, self.sigma)

        data = (data * 255.0 / np.max(data))

        self.loading_screen.close()
        self.loading_screen.destroy()
        self.current_data = data
        self.plot_data()
        # svg_name, error_log = gerber_to_svg_gerbv(os.path.join(self.folder_name, file), file)

        pass

    def plot_data(self):
        # data = np.random.rand(10, 10)
        #
        # data = (data - min(data.flatten())) / (max(data.flatten()) - min(data.flatten()))
        custom_colormap_colors, norm = self.create_custom_colormap_with_values(values=self.current_data)
        self.canvas.axes.imshow(self.current_data, cmap=custom_colormap_colors, norm=norm)
        # self.canvas.axes.imshow(data, cmap='Oranges')

        self.canvas.draw()

    def create_custom_colormap_with_values(self, values):
        """
        Creates a custom colormap from a list of colors and their corresponding value ranges.

        :param colors: A list of color hex codes.
        :param values: A list of values corresponding to each color.
        :return: A LinearSegmentedColormap object and a Normalize object for mapping values.
        """
        cvals = []
        colors_chose = []
        max_value = max(values.flatten())
        for color_ in self.color_info:
            name = self.color_info[color_]['color_button'].palette().window().color().name()  # name
            checked = self.color_info[color_]['check_box'].isChecked()
            if checked:
                value = self.color_info[color_]['entry_box'].text().strip('%')
                try:
                    cvals.append(float(value) / 100 * max_value)
                    colors_chose.append(name)
                except Exception as err:
                    show_error_message(err)
                    return
        normalized_data = (values - np.min(values)) / (np.max(values) - np.min(values))
        cvals.sort()
        norm = plt.Normalize(min(cvals), max(cvals))
        colors_chose.reverse()
        cmap = LinearSegmentedColormap.from_list('my_colors', list(zip(norm(cvals), colors_chose)))

        # Ensure the values list is one item longer than the colors list
        # assert len(values) == len(colors) + 1, "Values list must be one item longer than colors list."

        # new_colors = []
        # for i, color in enumerate(colors_custom):
        #     new_colors.append((thresholds[i], color[1]))
        #
        # colors_custom = new_colors
        # Generate a list of 100 colors from the colormap

        return cmap, norm

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

            if os.path.exists(os.path.join(self.folder_name, "items_data.json")):
                self.load_file = True
            failed_to_verify = []
            while waiting:
                if self.load_file:
                    try:
                        files_info = self.read_and_sort_json_by_index(os.path.join(self.folder_name, "items_data.json"))
                        for item in files_info:
                            item = scroll.ItemWidget(item['file_name'], self.file_scrollAreaWidgetContents)
                            self.scroll_areaLayout.addWidget(item)
                            item.selected_layer.stateChanged.connect(lambda state, it=item: self.selectItem(it))
                            max_width = max(max_width, item.sizeHint().width())
                            self.items.append(item)
                        pass
                        waiting = False
                    except:
                        show_error_message("The file items_data.json is corrupted. The file will be deleted. The files will be reloaded in the order they are in the folder.")
                        os.remove(os.path.join(self.folder_name, "items_data.json"))
                        self.load_file = False
                elif not self.load_file:

                    for idx, file in enumerate(check_names):
                        try:
                            if verify_gerber(file[0]):
                                item = scroll.ItemWidget(f"{file[1]}", self.file_scrollAreaWidgetContents)
                                self.scroll_areaLayout.addWidget(item)
                                item.selected_layer.stateChanged.connect(lambda state, it=item: self.selectItem(it))
                                max_width = max(max_width, item.sizeHint().width())
                                self.items.append(item)
                            else:
                                failed_to_verify.append(file[0])
                            check_names.remove(file)

                            self.loading_screen.set_progress(self.loading_screen.progressBar.value() + idx,
                                                             f"Verifying Files... {file[1]}%")

                        except:
                            pass

                    if len(check_names) == 0:
                        waiting = False
            failed_to_verify = list(set(failed_to_verify))
            if failed_to_verify:
                fail_message = '\n'.join(failed_to_verify)
                fail_message = 'The following files could not be read: \n' + fail_message
                show_error_message(fail_message, win_title='Warning', icon=QMessageBox.Icon.Warning)
            # for i in range(1):  # Assuming you want to add 5 labels
            #     long_text = "This is a very long label text " * 10  # Repeat the text to make it long
            #     label =QtWidgets.QLabel(long_text, self.file_scrollAreaWidgetContents)
            #     self.scroll_areaLayout.addWidget(label)
            # Set the maximum value of the horizontal scroll bar
            # self.file_scrollArea.horizontalScrollBar().setMaximum(max_width + 100)
        self.loading_screen.close()
        self.loading_screen.destroy()

    def selectItem(self, item):
        if self.selected_item is not None:
            self.selected_item.set_highlight(False)
        self.selected_item = item
        self.selected_item.set_highlight(True)

    def start_loading(self, func):
        if func == self.gerber_folder_button_clicked:
            self.folder_name = QtWidgets.QFileDialog.getExistingDirectory(self.centralwidget, "Select Folder", directory=fr'{os.getcwd()}\Assets\gerbers\GaryExample')
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
        items_data = []
        for idx, item in enumerate(self.items):

            file_path = os.path.join(self.folder_name, item.selected_layer.text())
            if os.path.isfile(file_path):
                # Assuming the Cu weight is part of the file name, e.g., "file_name_CuWeight.txt"
                # You might need to adjust this logic depending on how the Cu weight is stored
                cu_weight = item.comboBox.currentText()  # Replace this with actual extraction logic
                items_data.append({"index": idx, "file_path": file_path, "file_name": item.selected_layer.text(), "cu_weight": cu_weight})
        output_json_path = os.path.join(self.folder_name, "items_data.json")
        with open(output_json_path, 'w') as json_file:
            json.dump(items_data, json_file, indent=4)
            # print(f"{idx + 1}: {item.selected_layer.text(), item.comboBox.currentText()}")

    def read_and_sort_json_by_index(self, json_file_path):
        # Open and read the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Sort the list of dictionaries by the 'index' key
        sorted_data = sorted(data, key=lambda x: x['index'])

        # Return the sorted list
        return sorted_data

    def on_color_box_clicked(self, i):
        button = self.color_info[i]['color_button']
        # button = self.mainwidget.sender()
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            button.setStyleSheet(f"background-color: {color.name()}")

    def recolor_button_clicked(self):
        if os.path.exists('Assets/colors'):
            with open('Assets/colors', 'r') as file:
                color_palettes = json.load(file)

        new_colors = color_palettes[self.color_pallete_options_combobox.currentText()]
        for color in new_colors:
            self.color_info[int(color)]['color_button'].setStyleSheet(f"background-color: {new_colors[color]}")
        try:
            self.plot_data()

        except AttributeError:
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
        try:
            with open('Assets/colors', 'r') as file:
                color_palettes = json.load(file)
            self.color_pallete_options_combobox.clear()
            for name in color_palettes.keys():
                self.color_pallete_options_combobox.addItem(name)
        except:
            pass


def show_error_message(message, win_title="Error", icon=QMessageBox.Icon.Critical):
    error_dialog = QMessageBox()
    error_dialog.setIcon(icon)
    error_dialog.setText(message)
    error_dialog.setWindowTitle(win_title)
    error_dialog.exec()

