from calculations.multi_layer import *
from calculations.layer_calcs import *
from loading.img2array import *
from plotting import *

# array = bitmap_to_array(r"C:\Users\Asa Guest\Documents\Projects\Copper "
#                         r"Balancing\Assets\Design_Tif_Rip50\Design_Tif_Rip50\l7_signal_1oz_0.tif")
# array = blur_tiff_gauss(array, sigma=0)
bitmap_dict = open_multiple_bitmaps(r"C:\Users\Asa Guest\Documents\Projects\Copper Balancing\Assets\Design_Tif_Rip50\Design_Tif_Rip50")
pic = multiple_layers(bitmap_dict)
blur = 1
pic = blur_tiff_manual(pic, blur_x=blur, blur_y=blur)
# pic = blur_tiff_gauss(pic, sigma=1)
plot_heat_map(pic)