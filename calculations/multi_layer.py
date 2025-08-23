import numpy as np

def multiple_layers(layer_dict: dict) -> np.ndarray:
    """
    Crop all layers to the smallest common shape and sum them together.

    @param layer_dict: Dictionary of 2D numpy arrays
    @return: Summed 2D numpy array
    """
    try:
        # Step 1: Find the smallest shape
        min_rows = min(arr.shape[0] for arr in layer_dict.values())
        min_cols = min(arr.shape[1] for arr in layer_dict.values())

        # Step 2: Crop each array to the smallest shape
        cropped_layers = [
            arr[:min_rows, :min_cols] for arr in layer_dict.values()
        ]

        # Step 3: Sum the cropped arrays
        layer_array = np.sum(cropped_layers, axis=0)

    except Exception as e:
        print(f"Error while summing layers: {e}")
        layer_array = None

    return layer_array

