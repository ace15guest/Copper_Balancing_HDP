import configparser as cp
import os
from idlelib.configdialog import is_int

from file_handling import ini_global_path, create_folders


def check_structure(config):
    section1_check = "FileLocations"
    section2_check = "Algorithm"
    # check to make sure all file locations exist
    check1, check2, check3 = False, False, False
    check4, check5, check6, check7 = False, False, False, False
    if section1_check in config.sections():
        check1 = os.path.exists(config[section1_check]["Tiff Path".lower()])
        check2 = os.path.exists(config[section1_check]["Gerber Path".lower()])
        check3 = os.path.exists(config[section1_check]["Save Path".lower()])
    # Check to make sure all blurring parameters exist
    if section2_check in config.sections():
        try:

            check4 = config[section2_check]['blurring'] in ['gauss', 'box', 'median', 'bilateral', "MetAve"]
            check5 = isinstance(float(config[section2_check]['gauss sigma']), float)
            check6 = isinstance(int(config[section2_check]['kernel size']), int)
            check7 = isinstance(int(config[section2_check]['dpi']), int)
        except Exception as error:
            return False
    if check1 and check2 and check3 and check4 and check5 and check6 and check7:
            return True
    return False

def create_config_parser(db_path=None):
    config = cp.ConfigParser()

    if os.path.exists(ini_global_path):
        config.read(ini_global_path)
        if check_structure(config):
            return config

    config["FileLocations"] = {"Tiff Path".lower(): "C:/",
                               "Gerber Path".lower(): "C:/",
                               "Save Path".lower(): "C:/"
                               }
    config["Algorithm"] = {"blurring": 'gauss',
                           "gauss sigma": 5,
                           "kernel size": 5,
                           "dpi": 500,
                           }
    create_folders(ini_global_path)
    # Write the config to a file
    write_to_config(config)
    return config

def write_to_config(config):
    with open(ini_global_path, 'w') as configfile:
        config.write(configfile)
    return