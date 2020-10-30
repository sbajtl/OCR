import cv2
import imutils
import numpy as np
import pytesseract as pytesseract
from PIL import Image

from helpers.Utils import UtilsHelper


class Mrz:

    @staticmethod
    def detect():
        # initialize a rectangular and square structuring kernel
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

        image = None
        roi = None

        images = UtilsHelper.load_images()

        # loop over the input image paths
        for image in images:
            # load the image, resize it, and convert it to grayscale
            image = imutils.resize(image, height=600)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # smooth the image using a 3x3 Gaussian, then apply the blackhat
            # morphological operator to find dark regions on a light background
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

            # compute the Scharr gradient of the blackhat image and scale the
            # result into the range [0, 255]
            grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
            grad_x = np.absolute(grad_x)
            (minVal, maxVal) = (np.min(grad_x), np.max(grad_x))
            grad_x = (255 * ((grad_x - minVal) / (maxVal - minVal))).astype("uint8")

            # apply a closing operation using the rectangular kernel to close
            # gaps in between letters -- then apply Otsu's thresholding method
            grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
            thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            # perform another closing operation, this time using the square
            # kernel to close gaps between lines of the MRZ, then perform a
            # series of erosions to break apart connected components
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sq_kernel)
            thresh = cv2.erode(thresh, None, iterations=4)

            # during thresholding, it's possible that border pixels were
            # included in the thresholding, so let's set 5% of the left and
            # right borders to zero
            p = int(image.shape[1] * 0.05)
            thresh[:, 0:p] = 0
            thresh[:, image.shape[1] - p:] = 0

            # find contours in the thresholded image and sort them by their
            # size
            contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # loop over the contours
            for c in contours:
                # compute the bounding box of the contour and use the contour to
                # compute the aspect ratio and coverage ratio of the bounding box
                # width to the width of the image
                (x, y, w, h) = cv2.boundingRect(c)
                ar = w / float(h)
                cr_width = w / float(gray.shape[1])

                # check to see if the aspect ratio and coverage width are within
                # acceptable criteria
                if ar > 5 and cr_width > 0.75:
                    # pad the bounding box since we applied erosions and now need
                    # to re-grow it
                    p_x = int((x + w) * 0.03)
                    p_y = int((y + h) * 0.03)
                    (x, y) = (x - p_x, y - p_y)
                    (w, h) = (w + (p_x * 2), h + (p_y * 2))

                    # extract the ROI from the image and draw a bounding box
                    # surrounding the MRZ
                    roi = image[y:y + h, x:x + w].copy()
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    break

        # show the output temp_images
        cv2.imshow("Image", image)
        # cv2.imshow("ROI", roi)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # check to see if we should apply thresholding to preprocess the
        # image
        # if args["preprocess"] == "thresh":
        # gray = cv2.threshold(gray, 0, 255,
        #                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # make a check to see if median blurring should be done to remove
        # noise
        # elif args["preprocess"] == "blur":
        gray = cv2.medianBlur(gray, 3)

        # write the grayscale image to disk as a temporary file so we can
        # apply OCR to it
        filename = "temp_images/temp_gray.jpg"
        cv2.imwrite(filename, gray)

        # load the image as a PIL/Pillow image, apply OCR, and then delete
        # the temporary file
        text = pytesseract.image_to_string(Image.open(filename), lang="ocrb", config="--psm 4 --oem 3 -c tessedit_char_whitelist=-01234567890ABCDEFGHIJKLMNOPRSTUVWXYZ<:")
        # os.remove(filename)
        print(text)

        # show the output temp_images
        # cv2.imshow("Image", image)
        cv2.imshow("Output", gray)

        cv2.waitKey(0)
