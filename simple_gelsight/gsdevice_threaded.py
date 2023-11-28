import cv2
import numpy as np
import warnings
import threading

def resize_crop_mini(img, imgw, imgh):
    # remove 1/7th of border from each size
    border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(np.floor(img.shape[1] * (1 / 7)))
    # keep the ratio the same as the original image size
    img = img[border_size_x+2:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
    # final resize for 3d
    return cv2.resize(img, (imgw, imgh))


class Camera:
    def __init__(self, dev_num, flush_frames=50) -> None:
        # variable to store data
        self.image = None
        self.dev_id = dev_num
        self.imgw = 320
        self.imgh = 240
        self.millimeters_per_pixel = 0.0634 # mini gel 18x24mm at 240x320
        self.cam = None
        self.record = False
        self.image_num = 0 # number of images collected
        self.returned_image_num = 0 # num of the last image returned
        self.image_lock = threading.Lock() # lock for when self.image is being altered
        self.new_image_event = threading.Event() # event to signal when a new image is available
        self.safe_to_release_cam = threading.Event() # event to signal when to kill the camera
        self.connect(flush_frames)

    def connect(self, flush_frames=50) -> bool:

        # The camera in Mini is a USB camera and uses open cv to get the video data from the streamed video
        self.safe_to_release_cam.clear() # reset event to signal that the camera is not safe to release
        self.cam = cv2.VideoCapture(self.dev_id)
        if self.cam is None or not self.cam.isOpened():
            warnings.warn('Warning: unable to open video source: ' + str(self.dev_id))
            return False

        # flush out first few frames
        for i in range(flush_frames):
            print('flushing frame', i)
            self.cam.read()
            self.image_num += 1

        # Start recording images:
        threading.Thread(target=self.run_camera, daemon=True).start()

        # Wait for the first image to be collected:
        self.new_image_event.wait()

        return True

    def get_image(self) -> np.ndarray:
        with self.image_lock:
            self.returned_image_num = self.image_num
            self.new_image_event.clear()
            return self.image.copy()
    
    def get_next_image(self) -> np.ndarray:
        # Returns the next image. Waits for the camera to get a new image if necessary.
        if self.returned_image_num == self.image_num:
            self.new_image_event.wait()
        return self.get_image()

    def run_camera(self) -> None:
        self.record = True
        while self.record:
            ret, frame = self.cam.read()
            # print('got frame!')
            if ret:
                new_image = resize_crop_mini(frame, self.imgw, self.imgh)
            else:
                warnings.warn('ERROR! reading image from camera!')

            with self.image_lock:
                self.image = new_image
                self.new_image_event.set()
                self.image_num += 1
        self.safe_to_release_cam.set()

    def save_image(self, fname) -> None:
         cv2.imwrite(fname, self.image)

    def close(self) -> None:
        self.record = False
        self.safe_to_release_cam.wait()
        self.cam.release()



