import numpy as np
def multiple_layers(layer_dict: dict) -> np.array:
    
    layer_array =  np.sum(list(layer_dict.values()), axis=0)

    return layer_array

