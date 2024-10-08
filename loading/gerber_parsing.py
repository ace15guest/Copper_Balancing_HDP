import os
import re
from gerber import load_layer


def find_extrema_points(gerber_files: list):
    # Initialize extrema values for the current file
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')

    min_x_str = ''
    min_y_str = ''
    out_file = ''
    max_x_str = ''
    max_y_str = ''
    for gerber_file in gerber_files:
        with open(gerber_file, 'r', encoding='utf-8') as file:
            try:
                for line in file:
                    # print(line)
                    if line.startswith('%') or ',' in line:
                        continue

                    coord_pattern = re.compile(r'[XY]([-+]?[0-9]*\.?[0-9]+)')
                    matches = coord_pattern.findall(line)
                    if matches:
                        old_min_x = min_x
                        old_max_x = max_x

                        old_min_y = min_y
                        old_max_y = max_y
                        if len(matches) == 2:
                            x = float(matches[0])
                            y = float(matches[1])

                            min_x = min(min_x, x)
                            max_x = max(max_x, x)
                            min_y = min(min_y, y)
                            max_y = max(max_y, y)

                            if old_min_x != min_x:
                                min_x_str = matches[0]
                            if old_max_x != max_x:
                                max_x_str = matches[0]
                            if old_min_y != min_y:
                                min_y_str = matches[1]
                            if old_max_y != max_y:
                                max_y_str = matches[1]

                            out_file = gerber_file

                        if len(matches) == 1:

                            if line.startswith('X'):
                                x = float(matches[0])
                                min_x = min(min_x, x)
                                max_x = max(max_x, x)

                                if old_min_x != min_x:
                                    min_x_str = matches[0]
                                if old_max_x != max_x:
                                    max_x_str = matches[0]

                            elif line.startswith('Y'):
                                y = float(matches[0])

                                min_y = min(min_y, y)
                                max_y = max(max_y, y)

                                if old_min_y != min_y:
                                    min_y_str = matches[0]
                                if old_max_y != max_y:
                                    max_y_str = matches[0]
                            out_file = gerber_file
            except:
                pass

    return min_x_str, max_x_str, min_y_str, max_y_str, out_file


def write_bounding_file(other_file, out_dir, outer_coords, top_expand=0, bottom_expand=0, left_expand=0, right_expand=0):
    outer_coords = list(outer_coords)
    outer_coords[0] = increment_string_by_n(outer_coords[0], left_expand)
    outer_coords[1] = increment_string_by_n(outer_coords[1], right_expand)
    outer_coords[2] = increment_string_by_n(outer_coords[2], bottom_expand)
    outer_coords[3] = increment_string_by_n(outer_coords[3], top_expand)


    ll = f'X{outer_coords[0]}Y{outer_coords[2]}D02*\n'
    lr = f'X{outer_coords[1]}Y{outer_coords[2]}D01*\n'
    ur = f'X{outer_coords[1]}Y{outer_coords[3]}D01*\n'
    ul = f'X{outer_coords[0]}Y{outer_coords[3]}D01*\n'
    ll_f = f'X{outer_coords[0]}Y{outer_coords[2]}D01*\n'


    with open(f'{out_dir}\\bounding_box.gbr', 'w') as file:
        with open(other_file, 'r') as other:
            for line in other:
                if line.startswith('%'):
                    file.write(line)
        file.write(ll)
        file.write(lr)
        file.write(ur)
        file.write(ul)
        file.write(ll_f)
        file.write("M02*")

    return


def increment_string_by_n(original_string, increment_value):
    # Convert the string to an integer and add the increment value
    incremented_value = int(original_string) + int(increment_value*int(original_string)/100)

    # Determine the original length of the string
    original_length = len(original_string)

    # Convert back to string, zero-padded to the original length
    new_string = f"{incremented_value:0{original_length}}"

    return new_string

file1 = r"C:\Users\6J2739897\Documents\projects\Projects4Others\HDP\Copper_Balancing_HDP\Assets\gerbers\CAF Coupon No Info\p5"
file2 = r"C:\Users\6J2739897\Documents\projects\Projects4Others\HDP\Copper_Balancing_HDP\Assets\gerbers\CAF Coupon No Info\p7"
file3 = r"C:\Users\6J2739897\Documents\projects\Projects4Others\HDP\Copper_Balancing_HDP\Assets\gerbers\CAF Coupon No Info\s04"
out = find_extrema_points([file1, file2, file3])
write_bounding_file(out[4], r'C:\Users\6J2739897\Documents\projects\Projects4Others\HDP\Copper_Balancing_HDP\Assets\gerbers\CAF Coupon No Info', out[:4], top_expand=40, bottom_expand=10, left_expand=10,right_expand=3)