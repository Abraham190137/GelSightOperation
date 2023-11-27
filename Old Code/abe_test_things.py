import time
from typing import Any
import cv2
import threading

# cap = cv2.VideoCapture(0)


class ThreadedCap:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.image = None
        self.thread = threading.Thread(target=self.run, args=(), daemon=True)
        self.keep_running = True
        self.thread.start()

    def run(self):
        while self.keep_running:
            _, self.image = self.cap.read()

    def read(self):
        return self.image
    
    def release(self):
        self.keep_running = False
        self.cap.release()
# Open the camera

cap = ThreadedCap()
while True:
    # Read an image
    last_time = time.time()
    frame = cap.read()
    print("get frame time", time.time()-last_time)
    # time.sleep(1)


# Release the camera
cap.release()
# Close all windows
cv2.destroyAllWindows()
