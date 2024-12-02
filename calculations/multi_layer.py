import numpy as np


def multiple_layers(layer_dict: dict) -> np.array:
    try:
        print(layer_dict)
        layer_array = np.sum(list(layer_dict.values()), axis=0)
    except Exception as e:
        print(e)
        layer_array = None
    print('Done')
    return layer_array
