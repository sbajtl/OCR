import os

import cv2


class UtilsHelper:
    path_to_temp_images = os.getcwd() + "/temp_images/"
    path_to_test_images = os.getcwd() + "/test_images/"
    use_test_images = False

    @staticmethod
    def load_images():
        if UtilsHelper.use_test_images:
            images = UtilsHelper.load_from_folder(UtilsHelper.path_to_test_images)
        else:
            images = UtilsHelper.load_from_folder(UtilsHelper.path_to_temp_images)

        return images;

    @staticmethod
    def remove_all_images():
        if not UtilsHelper.use_test_images:
            files = [f for f in os.listdir(UtilsHelper.path_to_temp_images) if f.endswith(".jpg")]
            for f in files:
                os.remove(os.path.join(UtilsHelper.path_to_temp_images, f))

    @staticmethod
    def load_from_folder(path_to_images):
        images = []
        for filename in os.listdir(path_to_images):
            img = cv2.imread(os.path.join(path_to_images, filename))
            if img is not None:
                images.append(img)
        return images
