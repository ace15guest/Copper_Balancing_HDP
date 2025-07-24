import time

import numpy as np
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import QRect, pyqtSignal, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QMainWindow, QScrollArea, QWidget, QVBoxLayout, QPushButton, QGroupBox, QComboBox
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PyQt6.QtWidgets import QSizePolicy
from gui.loading_bar import LoadingScreen
from gui import scroll
from loading.gerber_load import check_gerber, verify_gerber
import os
import json
import matplotlib.colors as c
from loading import app_settings_ini
from PyQt6 import QtGui
from PyQt6.QtWidgets import QMessageBox
from gui.colorbar import BlendedColorBar
from gui.plotting_canvas import MplCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from loading.gerber_conversions import gerber_to_svg_gerbv, svg_to_tiff_inkscape, svg_to_tiff, gerber_to_pdf_gerbv, pdf_page_to_array, gerber_to_png_gerbv, check_tiff_dimensions
from loading.img2array import bitmap_to_array
import matplotlib.pyplot as plt
from calculations.layer_calcs import blur_tiff_manual, blur_tiff_gauss, box_blur, median_blur, met_ave, scan_gerber_extrema
from calculations.multi_layer import multiple_layers
from file_handling import clear_folder
from gui.settings import SettingsPage
from file_handling import create_outline_gerber_from_file
import plotly.graph_objects as go


class MainWindow(QMainWindow):
    swap = pyqtSignal(str, str) # I forget what this does and if it has any functionality
    def __init__(self):
        super().__init__()

        self.selected_item = None # The item that is selected in the scroll window changed by selectedItem function
        self.load_file = False
        self.sigma = 2 # Gaussian Blur parameter
        self.blur_x = 5 # Blur in the y direction
        self.blur_y = 5 # Blur in the X direction
        self.dpi_val = 400 #dots per inch of tiff output Thi
        self.run_verification = True  # Run the verification on input files
        self.raw_tiff_selected = False # If the tiff folder is selected outright
        self.blur_type = 'gauss'  # The type of blur to apply to the tiff files
        self.gerber_folder_name = ''  # The folder where the gerber data is held
        self.tiff_folder_name = ''  # The folder where tiff files are held

        # Default folder locations
        self.temp_folder = f"{os.environ['Temp']}/CuBalancing/temp"
        self.temp_svg_folder = f"{os.environ['Temp']}/CuBalancing/temp_svg"
        self.temp_error_folder = f"{os.environ['Temp']}/CuBalancing/temp_error"
        self.temp_tiff_folder = f"{os.environ['Temp']}/CuBalancing/temp_tiff"
        self.temp_pdf_folder = f"{os.environ['Temp']}/CuBalancing/temp_pdf"

        # Give our object us access to the configuration file
        self.config = app_settings_ini.create_config_parser()

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
        self.resize(1605, 940)
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
        self.add_menu_bar()
        self.place_gerber_folder()
        self.place_tiff_folder()
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
        self.setup_menu = self.menuBar().addMenu("Setup")
        self.edit_menu = self.menuBar().addMenu("Edit")
        self.view_menu = self.menuBar().addMenu("View")
        self.help_menu = self.menuBar().addMenu("Help")

        self.add_drop_down_options()
        # Create the QAction for showing the dropdown

        # fileMenu = self.menuBar.addMenu('&File')
        # # Add actions to "File" menu
        # openAction = QAction('&Open', self)
        # openAction.triggered.connect(self.openFile)  # Assuming openFile is a method for opening files
        # fileMenu.addAction(openAction)

    def add_drop_down_options(self):
        # Check if the dropdown already exists to avoid creating multiple instances
        self.show_dropdown_action = QAction("Settings", self)
        self.show_dropdown_action.setStatusTip('Alter application settings')
        self.setup_menu.addAction(self.show_dropdown_action)

        # Connect the action to the method for handling the dropdown
        self.show_dropdown_action.triggered.connect(self.open_settings)

    def open_settings(self):
        self.settings_window = SettingsPage(self.config)
        self.settings_window.show()
        return

    def place_group_box(self):
        self.right_groupbox = QGroupBox(self.centralwidget)
        self.right_layout = QVBoxLayout(self.right_groupbox)

        self.right_groupbox.setGeometry(500, 50, 1100, 850)
        self.right_layout.addWidget(self.right_groupbox)

    def place_scroll_widget(self):
        self.file_scrollArea = QScrollArea(self.centralwidget)
        self.file_scrollArea.setGeometry(QRect(205, 50, 290, 565))
        self.file_scrollArea.setWidgetResizable(True)
        self.file_scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # Set horizontal scroll bar
        self.file_scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.file_scrollAreaWidgetContents = QWidget()
        self.file_scrollArea.setWidget(self.file_scrollAreaWidgetContents)
        self.scroll_areaLayout = QVBoxLayout(self.file_scrollAreaWidgetContents)

    def place_buttons_below_scroll(self):
        x = 205
        x_size = 290
        y_size = 25

        y_orig = 650

        self.moveUpButton = QPushButton('Move Up', self)
        self.moveUpButton.clicked.connect(self.moveSelectedItemUp)
        self.moveUpButton.setGeometry(QRect(x, y_orig, x_size, y_size))

        self.moveDownButton = QPushButton('Move Down', self)
        self.moveDownButton.clicked.connect(self.moveSelectedItemDown)
        self.moveDownButton.setGeometry(QRect(x, y_orig + y_size, x_size, y_size))

        self.moveDownButton = QPushButton('Save Order', self)
        self.moveDownButton.clicked.connect(self.save_folder_order)
        self.moveDownButton.setGeometry(QRect(x, y_orig + y_size * 2, x_size, y_size))

        self.moveDownButton = QPushButton('Delete Order', self)
        self.moveDownButton.clicked.connect(self.delete_folder_order)
        self.moveDownButton.setGeometry(QRect(x, y_orig + y_size * 3, x_size, y_size))

        self.submit_button = QPushButton('Submit', self)
        self.submit_button.clicked.connect(lambda dummy: self.start_loading(self.submit_button_clicked))
        self.submit_button.setGeometry(QRect(x, y_orig + y_size * 4, x_size, y_size))

        # Step 1 & 2: Add the button and connect it
        self.checkAllButton = QPushButton('Check All Files', self)
        self.checkAllButton.setGeometry(QRect(x, y_orig + y_size * 5, x_size, y_size))  # Adjust the position as needed
        self.checkAllButton.clicked.connect(self.checkAllFiles)

        self.checkAllButton = QPushButton('Deselect All Files', self)
        self.checkAllButton.setGeometry(QRect(x, y_orig + y_size * 6, x_size, y_size))  # Adjust the position as needed
        self.checkAllButton.clicked.connect(self.deselectAllFiles)

        self.checkAllButton = QPushButton('Check All Inverse', self)
        self.checkAllButton.setGeometry(QRect(x, y_orig + y_size * 7, x_size, y_size))  # Adjust the position as needed
        self.checkAllButton.clicked.connect(self.checkAllInverse)

        self.checkAllButton = QPushButton('Deselect All Inverse', self)
        self.checkAllButton.setGeometry(QRect(x, y_orig + y_size * 8, x_size, y_size))  # Adjust the position as needed
        self.checkAllButton.clicked.connect(self.deselectAllInverse)

        # Step 3 & 4: Implement the method to check all files

    def delete_folder_order(self):
        if self.gerber_folder_name == '':
            show_error_message('Please select a gerber file')
            return
        elif os.path.exists(os.path.join(self.gerber_folder_name, "items_data.json")):
            os.remove(os.path.join(self.gerber_folder_name, "items_data.json"))
            return
        else:
            show_error_message("No order file exists. Please select 'Save Order' button to create one.")
            return

    def checkAllFiles(self):
        for item in self.items:
            item.selected_layer.setChecked(True)

    def deselectAllFiles(self):
        for item in self.items:
            item.selected_layer.setChecked(False)

    def checkAllInverse(self):
        for item in self.items:
            item.inverted_layer.setChecked(True)

    def deselectAllInverse(self):
        for item in self.items:
            item.inverted_layer.setChecked(False)

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
            self.color_info[i]['color_button'].clicked.connect(lambda x, k=i: self.on_color_box_clicked(k))
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

    def place_tiff_folder(self):

        self.tiff_folder_line_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.tiff_folder_line_edit.setGeometry(QtCore.QRect(700, 10, 531, 21))
        self.tiff_folder_line_edit.setObjectName("tiff_folder_line_edit")
        self.tiff_folder_line_edit.setPlaceholderText("Tiff Folder Path")

        # Place Tiff folder button
        self.tiff_folder_button = QtWidgets.QPushButton(parent=self.centralwidget, text="Tiff Folder")
        self.tiff_folder_button.clicked.connect(lambda x: self.start_loading(self.tiff_folder_button_clicked))
        self.tiff_folder_button.setGeometry(QtCore.QRect(1240, 10, 111, 21))  # Set the geometry of the button

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
        """
        @param outline_file:
        @return:
        """
        self.files_chosen = {}  # Tracking the selected files
        if not self.items:
            return
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
        extrema = scan_gerber_extrema(self.gerber_folder_name)
        create_outline_gerber_from_file(extrema['file_path'], extrema, os.path.join(self.gerber_folder_name, "outline_cubalance_HDP.gbr"))
        outline_file = os.path.join(self.gerber_folder_name, "outline_cubalance_HDP.gbr")
        # If we did not select raw tiff files (we selected gerber)
        if not self.raw_tiff_selected:
            for idx, file in enumerate(self.files_chosen):
                self.loading_screen.set_progress(idx, f"Converting to vectorized format... {file}")
                tiff_name = os.path.join(self.temp_tiff_folder, file)
                gerber_to_png_gerbv(os.path.join(self.gerber_folder_name, file), self.temp_tiff_folder, tiff_name, dpi=self.config["Algorithm"]["dpi"], scale=1, error_log_path=os.path.join(self.temp_error_folder, file), outline_file=outline_file)
                file_ct += 1

            while len(os.listdir(self.temp_tiff_folder)) < file_ct:
                time.sleep(5)

            all_same_size = check_tiff_dimensions(self.temp_tiff_folder) # Make sure all gerber are the same dimensions
        else:
            all_same_size = True # We can set this to true since we already checked the size of the tiff files

        if not self.raw_tiff_selected:
            tiff_folder = self.temp_tiff_folder
        else:
            tiff_folder = self.tiff_folder_name

        for idx, file in enumerate(os.listdir(tiff_folder)):
            if not self.raw_tiff_selected:
                inverted = self.files_chosen[file.replace('.tif', '')]['Inverted']
            else:
                if file not in list(self.files_chosen.keys()): # Because we were choosing a tiff folder that has all tiff and not only items we selected we have to exclude names not chosen
                    continue
                inverted = self.files_chosen[file]['Inverted']
            self.arrays[file] = bitmap_to_array(os.path.join(tiff_folder, file), inverted)
            self.loading_screen.set_progress(idx, f"Converting to array... {file}")

        def crop_dict_to_smallest_shape(arr_dict):
            # Find the smallest shape across all arrays
            min_rows = min(arr.shape[0] for arr in arr_dict.values())
            min_cols = min(arr.shape[1] for arr in arr_dict.values())
            target_shape = (min_rows, min_cols)

            # Crop all arrays to that shape
            cropped_dict = {
                key: arr[:min_rows, :min_cols] for key, arr in arr_dict.items()
            }

            return cropped_dict, target_shape

        cropped_arrays = crop_dict_to_smallest_shape(self.arrays)
        data = multiple_layers(cropped_arrays[0])


        # cropped_arrays, shape_used = crop_dict_to_smallest_shape(data)
        if self.config['Algorithm']['blurring'] == 'gauss':
            data = blur_tiff_gauss(data, float(self.config['Algorithm']['gauss sigma']))
        elif self.config['Algorithm']['blurring'] == 'box':
            data = box_blur(data, int(self.config['Algorithm']['kernel size']))
        elif self.config['Algorithm']['blurring'] == 'median':
            data = median_blur(data, int(self.config['Algorithm']['kernel size']))
        elif self.config['Algorithm']['blurring'] == 'bilateral':
            pass
        elif self.config['Algorithm']['blurring'] == 'MetAve':
            data = met_ave(data, int(self.config['Algorithm']['kernel size']))

        import numpy as np



        # Normalize the data between 0 and 255
        data = (data * 255.0 / np.max(data))


        self.loading_screen.close()
        self.loading_screen.destroy()
        self.current_data = data
        self.plot_data()



        pass


    def plot_data(self):
        custom_colormap_colors, norm = self.create_custom_colormap_with_values(values=self.current_data)
        im = self.canvas.axes.imshow(self.current_data, cmap=custom_colormap_colors, norm=norm)
        if hasattr(self, 'current_color_bar'):
            self.current_color_bar.remove()
        self.current_color_bar = self.canvas.figure.colorbar(im, ax=self.canvas.axes, orientation='horizontal', fraction=0.046, pad=0.04)

        # self.canvas.axes.imshow(data, cmap='Oranges')
        self.canvas.draw()
        # self.plot_data_plotly()


    def plot_data_plotly(self):
        custom_colormap_colors, norm = self.create_custom_colormap_with_values(values=self.current_data)

        # Create the heatmap using Plotly
        fig = go.Figure(data=go.Heatmap(z=self.current_data, colorscale='Viridis',  # Replace with your custom colormap if needed
            colorbar=dict(
                orientation='h',
                x=0.5,
                xanchor='center',
                y=-0.2
            )
        ))

        # Update layout for better appearance
        fig.update_layout(
            title='Heatmap with Horizontal Colorbar',
            margin=dict(l=40, r=40, t=40, b=40)
        )

        # Show the plot
        fig.show()

    def create_custom_colormap_with_values(self, values):
        # self.removeColorBar()
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

        # self.color_bar.labels = cvals

        # self.color_bar = BlendedColorBar(colors_chose, cvals, self)
        # self.color_bar.setGeometry(10, 10, 191, 31)
        # self.central_layout.addWidget(self.color_bar)
        # self.color_bar.setFixedHeight(30)  # Set fixed height for the color bar
        # Add the color bar to the layout

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
    def tiff_folder_button_clicked(self):
        self.selected_item =None
        self.raw_tiff_selected = True

        # options |= QtWidgets.QFileDialog.DontUseNativeDialog
        max_width = 0

        if self.tiff_folder_name:
            self.loading_screen.progressBar.setMaximum(len(os.listdir(self.tiff_folder_name)*2))  # Set the maximum length of the progress bar

            self.tiff_folder_line_edit.setText(self.tiff_folder_name)  # Display the Tiff Folder on the screen within the entry box
            self.config["FileLocations"]["tiff path"] = self.tiff_folder_name
            app_settings_ini.write_to_config(self.config)
            self.clear_layout(self.scroll_areaLayout)  # Clear any items in the scroll area

            self.items = []  # Hold the items that will be in the scroll area layout
            check_names = []

            if self.run_verification: # This should always be True
                self.loading_screen.set_progress(15, f"Verifying all tiff files are the same size")

                all_same_size, file_names = check_tiff_dimensions(folder_path=self.tiff_folder_name, raw_tiff=True)

                if not all_same_size:
                    show_error_message("All Tiff files must be the same size")

        waiting = True # Waits for all files to be loaded in
        if os.path.exists(os.path.join(self.tiff_folder_name, "items_data.json")):  # If there is a saved setting file, we will load it in
            self.load_file = True
        else:
            self.load_file = False
        failed_to_read = []
        # create counter
        count = 0
        while waiting:
            count +=1# Keep this while loop
            if self.load_file:
                # TODO: Read in Json file for tiff
                pass
            else:
                for idx,file in enumerate(file_names):
                    item = scroll.ItemWidget(f"{file}", self.file_scrollAreaWidgetContents)
                    self.scroll_areaLayout.addWidget(item)
                    item.selected_layer.stateChanged.connect(lambda state, it=item: self.selectItem(it))
                    max_width = max(max_width, item.sizeHint().width())  # Finds the max width but current
                    self.items.append(item)
                    file_names.remove(file)
            if len(file_names) == 0:
                waiting=False
            elif count>100:
                waiting=False
        self.loading_screen.close()
        self.loading_screen.destroy()
        return


    def gerber_folder_button_clicked(self):
        failed_to_verify = []
        self.raw_tiff_selected = False

        if self.gerber_folder_name:
            self.loading_screen.progressBar.setMaximum(len(os.listdir(self.gerber_folder_name) * 2))

            self.gerber_folder_line_edit.setText(self.gerber_folder_name)
            # Update the config file
            self.config["FileLocations"]["Gerber Path"] = self.gerber_folder_name
            app_settings_ini.write_to_config(self.config)

            self.clear_layout(self.scroll_areaLayout)
            self.items = []  # Items to be placed in the scroll area
            self.selected_item = None # probably best to keep this within the function but add to __init__ too
            max_width = 0
            check_names = []
            processes = []
            for idx, i in enumerate(os.listdir(self.gerber_folder_name)):
                if self.run_verification:  # Always run verification
                    log_file_name, process = check_gerber(os.path.join(self.gerber_folder_name, i))
                    processes.append(process)
                    check_names.append((log_file_name, i)) # Keep check names as a tuple. It will make implementation in this function easier
                self.loading_screen.set_progress(idx, f"Please Wait Loading Files... {i}%")

            waiting = True

            if os.path.exists(os.path.join(self.gerber_folder_name, "items_data.json")): # If there is a preloaded order load it in
                self.load_file = True # This allows us to load in the files in the while loop
            else:
                self.load_file = False # This will check all files within the folder and only take the one
            failed_to_verify = [] # Holds the files that the program could not read
            fail_count = 0 # How many cycles we will try before we fail a file
            while waiting: # Wrap it in a while loop because it sometimes fails with issues reading
                if self.load_file:
                    try:
                        files_info = self.read_and_sort_json_by_index(os.path.join(self.gerber_folder_name, "items_data.json"))
                        for file_saved in files_info:
                            item = scroll.ItemWidget(file_saved['file_name'], self.file_scrollAreaWidgetContents)
                            self.scroll_areaLayout.addWidget(item)

                            item.comboBox.setCurrentText(file_saved['cu_weight'])

                            if file_saved['inverted']:
                                item.inverted_layer.setChecked(True)
                            if file_saved['selected']:
                                item.selected_layer.setChecked(True)
                            item.selected_layer.stateChanged.connect(lambda state, it=item: self.selectItem(it))
                            max_width = max(max_width, item.sizeHint().width())
                            self.items.append(item)
                        pass
                        waiting = False
                    except Exception as error:
                        show_error_message("The file items_data.json is corrupted. The file will be deleted. The files will be reloaded in the order they are in the folder.")
                        os.remove(os.path.join(self.gerber_folder_name, "items_data.json"))
                        self.load_file = False
                elif not self.load_file:
                    for idx, file in enumerate(check_names):
                        try:
                            if verify_gerber(file[0]): # If the file is valid add it to the scroll area
                                item = scroll.ItemWidget(f"{file[1]}", self.file_scrollAreaWidgetContents)
                                self.scroll_areaLayout.addWidget(item)
                                item.selected_layer.stateChanged.connect(lambda state, it=item: self.selectItem(it))
                                max_width = max(max_width, item.sizeHint().width()) # Finds the max width but current
                                self.items.append(item)
                            else:
                                failed_to_verify.append(file[0]) # Files that the program could not verify
                            check_names.remove(file) # Once the file is checked remove it from the list

                            self.loading_screen.set_progress(self.loading_screen.progressBar.value() + idx,
                                                             f"Verifying Files... {file[1]}%")
                            fail_count = 0
                        except Exception as error:
                            print(error)
                            # fail_count+=1
                            # if fail_count > 750:
                            #     failed_to_verify.append(file[0])
                            #     check_names.remove(file)  # Once the file is checked remove it from the list

                    if len(check_names) == 0:
                        waiting = False
            failed_to_verify = list(set(failed_to_verify))


        self.loading_screen.close()
        self.loading_screen.destroy()
        # Show what items could not be ver
        if failed_to_verify:
            fail_message = '\n'.join(failed_to_verify)
            fail_message = 'The following files could not be read: \n' + fail_message
            show_error_message(fail_message, win_title='Warning', icon=QMessageBox.Icon.Warning)

    def selectItem(self, item):
        if self.selected_item is not None:
            self.selected_item.set_highlight(False)
        self.selected_item = item
        self.selected_item.set_highlight(True)

    def start_loading(self, func):

        if func == self.gerber_folder_button_clicked:
            tmp_name = QtWidgets.QFileDialog.getExistingDirectory(self.centralwidget, "Select Folder", directory=fr'{self.config["FileLocations"]["gerber path"]}')
            if tmp_name == "":
                return
            self.gerber_folder_name = tmp_name
        elif func == self.submit_button_clicked:
            pass
        elif func == self.tiff_folder_button_clicked:
            tmp_name = QtWidgets.QFileDialog.getExistingDirectory(self.centralwidget, "Select Folder", directory=fr'{self.config["FileLocations"]["tiff path"]}')
            if tmp_name == "":
                return
            self.tiff_folder_name = tmp_name


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
        try:

            if self.selected_item:
                index = self.items.index(self.selected_item)
                if index > 0:
                    self.items[index], self.items[index - 1] = self.items[index - 1], self.items[index]
                    self.updateLayout()
        except:
            pass

    def moveSelectedItemDown(self):
        try:
            if self.selected_item:
                index = self.items.index(self.selected_item)
                if index < len(self.items) - 1:
                    self.items[index], self.items[index + 1] = self.items[index + 1], self.items[index]
                    self.updateLayout()
        except:
            pass

    def updateLayout(self):
        for i in reversed(range(self.scroll_areaLayout.count())):
            self.scroll_areaLayout.itemAt(i).widget().setParent(None)
        for item in self.items:
            self.scroll_areaLayout.addWidget(item)

    def save_folder_order(self):
        items_data = []
        if not self.items:
            return
        for idx, item in enumerate(self.items):

            file_path = os.path.join(self.gerber_folder_name, item.selected_layer.text())
            if os.path.isfile(file_path):
                # Assuming the Cu weight is part of the file name, e.g., "file_name_CuWeight.txt"
                # You might need to adjust this logic depending on how the Cu weight is stored
                cu_weight = item.comboBox.currentText()  # Replace this with actual extraction logic
                inverted = item.inverted_layer.isChecked()
                selected = item.selected_layer.isChecked()
                items_data.append({"index": idx, "file_path": file_path, "file_name": item.selected_layer.text(), "cu_weight": cu_weight, "inverted": inverted, "selected": selected})
        output_json_path = os.path.join(self.gerber_folder_name, "items_data.json")
        with open(output_json_path, 'w') as json_file:
            json.dump(items_data, json_file, indent=4)

        show_error_message("Order saved successfully", win_title="Success", icon=QMessageBox.Icon.Information)
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




