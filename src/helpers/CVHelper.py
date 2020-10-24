import cv2


class OpenCVHelper:
    show_pic = None

    @staticmethod
    def start_camera():
        # Open video file
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        frame = None

        while camera.isOpened():
            # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
            # i.e. a single-column array, where each item in the column has the pixel RGB value
            ret, frame = camera.read()

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break

        # Image saving
        cv2.imwrite("images/temp_document.jpg", frame)

        # Clean up
        camera.release()
        cv2.destroyAllWindows()

