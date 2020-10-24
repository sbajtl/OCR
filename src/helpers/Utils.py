import os

import cv2


class UtilsHelper:
    path_to_images = os.getcwd() + "/images/"

    @staticmethod
    def load_images():
        images = []
        for filename in os.listdir(UtilsHelper.path_to_images):
            img = cv2.imread(os.path.join(UtilsHelper.path_to_images, filename))
            if img is not None:
                images.append(img)
        return images

    @staticmethod
    def remove_all_images():
        files = [f for f in os.listdir(UtilsHelper.path_to_images) if f.endswith(".jpg")]
        for f in files:
            os.remove(os.path.join(UtilsHelper.path_to_images, f))
