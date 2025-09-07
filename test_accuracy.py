import numpy as np
import time
from pathlib import Path
# custom imports
from file_handling import get_global_files, list_gerbers_with_weights, wait_for_folder_complete, clear_folder
from calculations.transformation import *
from loading.gerber_conversions import gerber_to_png_gerbv
from loading.img2array import bitmap_to_array
from calculations.multi_layer import multiple_layers_weighted
from calculations.layer_calcs import met_ave
from plotting.comparing import plot_pointclouds_and_heatmaps
from calculations.padding import fill_border
from calculations.comparison import align_and_compare
top_data_loc = r"C:\Users\Asa Guest\Downloads\files\CuBalanceDatFiles\TopDatFiles"  # The folder that holds the top Akro data
bot_data_loc = r"C:\Users\Asa Guest\Downloads\files\CuBalanceDatFiles\BottomDatFiles"  # The folder that holds the bottom Akro data

temp_tiff_folder = r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\temp_tiff"
temp_tiff_folder_path = Path(temp_tiff_folder)

Q1_folder = r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\gerbers\Cu_Balancing_Gerber\Q1"
LR_folder = r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\gerbers\Cu_Balancing_Gerber\Q3"
UL_folder = r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\gerbers\Cu_Balancing_Gerber\Q2"
UR_folder = r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\gerbers\Cu_Balancing_Gerber\Q4"
dpi_results = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
fills = ['nearest', 'idw', 'biharmonic', 'local_mean']
radii = [50, 100, 200, 300, 400, 500, 600, 700, 800]

Quartile_loc = ""
if __name__ == '__main__':
    # Read in all global DAT forms
    top_data_files = get_global_files(top_data_loc)
    bot_data_files = get_global_files(bot_data_loc)
    # Gerber Files
    Q1_Gerber_files = list_gerbers_with_weights(Q1_folder)
    LR_Gerber_files = list_gerbers_with_weights(LR_folder)
    UL_Gerber_files = list_gerbers_with_weights(UL_folder)
    UR_Gerber_files = list_gerbers_with_weights(UR_folder)
    # Cycle through the dpi
    for edge_fill in fills:
        for dpi in dpi_results:
            for radius in radii:
                arrays = {}

                # Cycle through the Top Global Data Files
                for top_global_path in top_data_files:
                    # Material and Supplier
                    mat_sup_id = '-'.join(top_global_path.split('\\')[-1].split('-')[0:3])
                    mat_sup_folder = '-'.join(top_global_path.split('\\')[-1].split('-')[0:2])
                    # Clear the temporary tiff folder
                    print("Clearing Temporary Tiff Folder")
                    clear_folder(temp_tiff_folder)
                    time.sleep(5)
                    # Create list of layer names to be blended
                    layer_names_for_blend = []
                    layer_weights_for_blend = {}
                    recalculate_array = False  # If we have the array already stored in memory, no need to recalculate anything

                    # Read the Akro Arrays and interpolate the nan values so we dont have 9999 or np.nan
                    dat_file_orig = np.loadtxt(top_global_path, delimiter="\t")
                    dat_file_filled = np.where(dat_file_orig == 9999.0, np.nan, dat_file_orig)
                    dat_file_9999_filled = fill_nans_nd(dat_file_filled, 'iterative')
                    #Wait for calculations
                    wait_for_calcs = False
                    if "Q1" in top_global_path:
                        Quartile_loc = "Q1"
                        for gerber_path in Q1_Gerber_files:
                            name = gerber_path[0].split("\\")[-1].split(".gbr")[0]
                            layer_names_for_blend.append(name)
                            layer_weights_for_blend[name] = gerber_path[1]
                            if name in arrays:
                                continue
                            gerber_to_png_gerbv(gerb_file=gerber_path[0], save_folder=temp_tiff_folder, save_path=rf"{temp_tiff_folder}\{name}", dpi=dpi, scale=1)  # Convert the gerbers to arrays
                            wait_for_calcs = True
                        if wait_for_calcs:
                            wait_for_folder_complete(temp_tiff_folder, expected_count=20)  # Wait for the 20 files to all show up
                        recalculate_array = True

                    elif "Q2" in top_global_path:
                        pass
                    elif "Q3" in top_global_path:
                        pass
                    elif "Q4" in top_global_path:
                        pass

                    if recalculate_array:
                        # Now convert the files in the temp_tiff_folder to bitmaps
                        for file in temp_tiff_folder_path.iterdir():
                            name_key = str(file).split("\\")[-1].split(".tif")[0]
                            arrays[name_key] = bitmap_to_array(file)
                        calculated_layers_preblend = multiple_layers_weighted(arrays)
                        calculated_layers_preblend_edge_mask = fill_border(calculated_layers_preblend, method=edge_fill)
                        calculated_layers_blended = met_ave(calculated_layers_preblend_edge_mask, radius=radius)
                        calculated_layers_blended_shrink = shrink_array(calculated_layers_blended, dat_file_9999_filled.shape)
                        calculated_layers_blended_shrink_rescale, dat_file_9999_filled_rescale, scale = rescale_to_shared_minmax(calculated_layers_blended_shrink, dat_file_9999_filled)
                        plot_save_folder = fr"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\Output\{mat_sup_folder}\{Quartile_loc}"
                        plot_save_name = f'{Quartile_loc}_{mat_sup_id}'

                        stats = align_and_compare(calculated_layers_blended_shrink_rescale, dat_file_9999_filled_rescale, ignore_zeros=False, detrend=True, with_scaling=False)
                        plot_pointclouds_and_heatmaps(calculated_layers_blended_shrink_rescale, dat_file_9999_filled_rescale, plot_save_folder, plot_save_name, stats_text=stats["text"])
                        # fig = plot_point_clouds_side_by_side_same_cmap(calculated_layers_blended_shrink_rescale, dat_file_9999_filled_rescale)
                        # fig.show()

                        pass
                    pass


print()