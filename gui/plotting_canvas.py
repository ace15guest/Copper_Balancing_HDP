from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=2, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)