import matplotlib.pyplot as plt
import numpy as np

a = np.random.random((16, 16))

def plot_heat_map(array):
    plt.imshow(array, cmap='hot', interpolation='nearest')
    plt.show()
