from classes.Mrz import Mrz
from helpers.CVHelper import OpenCVHelper
from helpers.Utils import UtilsHelper

if __name__ == '__main__':
    # UtilsHelper.use_test_images = True
    if not UtilsHelper.use_test_images:
        OpenCVHelper.start_camera()
    Mrz.detect()
    UtilsHelper.remove_all_images()
