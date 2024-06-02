import numpy as np
def multiple_layers(layer_dict: dict) -> np.array:
    
    layer_array = sum(layer_dict.values())

    return layer_array

layers = {'Layer 1':np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), 'Layer 2': np.array([[1, 2, 3], [0,0,0], [7, 8, 9]]), 'Layer 3': np.array([[1, 2, 3], [20,3,5], [7, 8, 9]])}
multiple_layers(layers)