# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.colors as c
# from matplotlib.colors import LinearSegmentedColormap
# def plot_heat_map(array, colors_custom=None):
#     max_num = np.max(array)
#     min_num = np.min(array)
#     array = max_num + min_num - array
#     # Create a colormap that goes from blue to white to red
#     cmap = plt.get_cmap('bwr')
#     colors_custom = [(0, 'blue'), (.3, 'lightblue'), (.5, 'white'), (.6, 'lightyellow'),  (.9, 'red'),(1, 'red')]
#     cmap = LinearSegmentedColormap.from_list('custom_cmap', colors_custom)
#     # Generate a list of 100 colors from the colormap
#     color = [cmap(i) for i in range(cmap.N)]
#
#     # Convert the colors to hexadecimal
#     hex_colors = [c.to_hex(i) for i in color]
#
#
#     # Define the colors and values in hex
#     colors = hex_colors
#     values = np.linspace(0, np.max(array.flatten()), 100)  # Modify the range to start at 0 and end at 255
#
#     # Create the custom color map
#     cmap = plt.cm.colors.ListedColormap(colors)
#
#     # Define the boundaries for each color
#
#
#     # Normalize the color map
#     norm = plt.cm.colors.BoundaryNorm(values, cmap.N)
#
#     # Rest of your code
#     plt.imshow(array, cmap=cmap, norm=norm, interpolation='nearest')
#     plt.colorbar()
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as c
from matplotlib.colors import LinearSegmentedColormap

def plot_heat_map(array, ax=None, colors_custom=None):
    if ax is None:
        ax = plt.gca()

    max_num = np.max(array)
    min_num = np.min(array)
    array = max_num + min_num - array

    # Create a colormap that goes from blue to white to red

    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors_custom)

    # Generate a list of 100 colors from the colormap
    colors = [cmap(i) for i in range(cmap.N)]

    # Convert the colors to hexadecimal
    hex_colors = [c.to_hex(color) for color in colors]

    # Define the colors and values in hex
    values = np.linspace(0, np.max(array.flatten()), 100)  # Modify the range to start at 0 and end at the max value

    # Create the custom color map
    cmap = plt.cm.colors.ListedColormap(hex_colors)

    # Normalize the color map
    norm = plt.cm.colors.BoundaryNorm(values, cmap.N)

    # Plot the heatmap
    cax = ax.imshow(array, cmap=cmap, norm=norm, interpolation='nearest')
    plt.colorbar(cax, ax=ax)
    plt.show()


# # Example usage:
# if __name__ == '__main__':
#     data = np.random.rand(10, 10)
#     fig, ax = plt.subplots()
#     plot_heat_map(data, ax=ax)
