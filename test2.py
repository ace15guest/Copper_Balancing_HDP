import subprocess
import os
import re


def get_bounding_box(file_path):
    result = subprocess.run(['Assets\gerbv\gerbv', '-C', '-D', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = result.stdout
    # Extract bounding box from the output
    match = re.search(r'\(([-+]?\d*\.\d+|\d+), ([-+]?\d*\.\d+|\d+)\) \(([-+]?\d*\.\d+|\d+), ([-+]?\d*\.\d+|\d+)\)', output)
    if match:
        x_min = float(match.group(1))
        y_min = float(match.group(2))
        x_max = float(match.group(3))
        y_max = float(match.group(4))
        return x_min, y_min, x_max, y_max
    return None


def find_outlining_gerber(folder_path):
    largest_bbox = None
    outlining_file = None

    for file_name in os.listdir(folder_path):
        if file_name:
            file_path = os.path.join(folder_path, file_name)
            bbox = get_bounding_box(file_path)
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                if not largest_bbox:
                    largest_bbox = bbox
                    outlining_file = file_name
                else:
                    if x_min < largest_bbox[0] and y_min < largest_bbox[1] and x_max > largest_bbox[2] and y_max > largest_bbox[3]:
                        largest_bbox = bbox
                        outlining_file = file_name

    return outlining_file, largest_bbox


# Specify the folder containing the Gerber files
folder_path = r'C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\gerbers\Coupon 1'

outlining_file, bbox = find_outlining_gerber(folder_path)
if outlining_file:
    print(f"The Gerber file that outlines all others is: {outlining_file}")
    print(f"Bounding Box: {bbox}")
else:
    print("No Gerber files found or no single file outlines all others.")