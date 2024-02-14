import time
import cv2
from simple_gelsight import Gelsight, get_camera_id
import numpy as np
from simple_gelsight import StrainInterpolation
import pickle
# from gsdevice_threaded import get_camera_ide


device_number = get_camera_id('GelSight')
MyGelsight = Gelsight(device_number)


for i in range(10000):
    print(i)
    # last_time = time.time()
    frame, marker_data, depth_image, x_inter, y_inter = MyGelsight.get_frame()
    # print(frame_data)
    cv2.imshow('frame', frame)
    cv2.imshow('depth', depth_image/depth_image.max())
    print('max depth', depth_image.max())
    print('min depth', depth_image.min())
    cv2.imshow('x_inter', x_inter/20 + 0.5)
    cv2.imshow('y_inter', y_inter/20 + 0.5)

    max_depth = 10
    max_strain = 30
    # Show all three using LAB color space
    normalized_depth = np.clip(100*np.maximum(depth_image, 0)/max_depth, 0, 100)
    # normalized_depth = np.clip(100*(depth_image/depth_image.max()), 0, 100)
    normalized_x_inter = np.clip(128*(x_inter/max_strain), -128, 127)
    normalized_y_inter = np.clip(128*(y_inter/max_strain), -128, 127)
    combo_image = np.stack((normalized_depth, normalized_x_inter, normalized_y_inter), axis=-1)
    
    print(combo_image.shape)
    combo_image = cv2.cvtColor(combo_image.astype(np.float32), cv2.COLOR_LAB2BGR)
    cv2.imshow('combo', combo_image)
    cv2.waitKey(1)



print('start cleaning up')
MyGelsight.clean_up()


    