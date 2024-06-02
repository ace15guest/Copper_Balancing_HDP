from calculations.layer_calcs import *
from loading.img2array import *
from plotting import *

array = bitmap_to_array(r"C:\Users\Asa Guest\Documents\Projects\Copper "
                        r"Balancing\Assets\Design_Tif_Rip50\Design_Tif_Rip50\l7_signal_1oz_0.tif")
array = blur_tiff_gauss(array, sigma=0)
plot_heat_map(array)