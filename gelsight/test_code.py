import time
import cv2
from GelsightDataCollector import GelsightDataCollector
import numpy as np
from abe_create_strain_images import StrainInterpolation
import pickle
# from gsdevice_threaded import get_camera_ide


device_number = 0
Gelsight = GelsightDataCollector(device_number)
Interpolator = StrainInterpolation(240, 320, 9, 7)


for i in range(10000):
    print(i)
    # last_time = time.time()
    frame, depth_image, frame_data = Gelsight.collect_frame_data() 
    # print(frame_data)
    cv2.imshow('frame', frame)
    cv2.imshow('depth', depth_image/depth_image.max())
    print('max depth', depth_image.max())
    print('min depth', depth_image.min())
    x_inter, y_inter = Interpolator(frame_data) #create_strain_images(frame_data)
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
Gelsight.clean_up()


    