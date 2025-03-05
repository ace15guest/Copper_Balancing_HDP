import numpy as np


def multiple_layers(layer_dict: dict) -> np.array:
    """
    Add all layers together
    @param layer_dict:
    @return:
    """
    try:
        layer_array = np.sum(list(layer_dict.values()), axis=0)
    except Exception as e:
        print(e)
        layer_array = None
    return layer_array
