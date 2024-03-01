import cv2
from . import find_marker, setting, marker_detection

import numpy as np
import os
from .gs3drecon import Reconstruction3D
# import rospy
import os
from .strain_interpolation import StrainInterpolation
import warnings
import time
import matplotlib.pyplot as plt

PATH = os.path.dirname(os.path.abspath(__file__))

def resize_crop_mini(img, imgw, imgh):
    # remove 1/7th of border from each size
    border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(np.floor(img.shape[1] * (1 / 7)))
    # keep the ratio the same as the original image size
    img = img[border_size_x+2:img.shape[0] - border_size_x, border_size_y:img.shape[1] - border_size_y]
    # final resize for 3d
    return cv2.resize(img, (imgw, imgh))

class MultiProcessedCamera:
    def __init__(self, device_number, flush_frames=0):
        self.device_number = device_number
        self.last_frame_num = 0

        # open the camera to get the frame size
        self.shape = np.array([240, 320, 3])

        self.image_array = multiprocessing.Array(ctypes.c_uint8, int(self.shape[0]*self.shape[1]*self.shape[2]), lock=True)
        self.frame_num_queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(target=self._run, args=(self.image_array, device_number, self.frame_num_queue, flush_frames))
        self.process.start()
        self.get_next_frame()
        print('MultiProcessedCamera: started!')

    @staticmethod
    def _run(image_array, device_number, frame_num_queue, flush_frames):
        image_array_np = np.ctypeslib.as_array(image_array.get_obj())
        cam = cv2.VideoCapture(device_number)
        if cam is None or not cam.isOpened():
            warnings.warn('Warning: unable to open video source: ' + str(device_number))
            return False
        for i in range(flush_frames):
            print('flushing frame', i)
            cam.read()
        frame_num = 0
        while True:
            ret, frame = cam.read()            
            if ret:
                frame = resize_crop_mini(frame, 320, 240)
                with image_array.get_lock():
                    image_array_np[:] = frame.ravel()
                frame_num_queue.put(frame_num)
                frame_num += 1
    
    def get_frame(self):
        with self.image_array.get_lock():
            image_array_np = np.ctypeslib.as_array(self.image_array.get_obj())
            frame = image_array_np.copy().reshape(self.shape)
        
        return frame
    
    def get_next_frame(self):
        frame_num = self.frame_num_queue.get() # wait for the next frame to be ready

        # clear the queue so that next time we can wait for the next frame
        while True:
            try:
                self.frame_num_queue.get_nowait()
            except:
                break

        return self.get_frame()
    
    def close(self):
        self.process.terminate()
        self.process.join()

    def __del__(self):
        self.close()

class Camera:
    def __init__(self, device_number, flush_frames=0):
        self.device_number = device_number
        self.cam = cv2.VideoCapture(device_number)
        if self.cam is None or not self.cam.isOpened():
            warnings.warn('Warning: unable to open video source: ' + str(device_number))
            return False
        self.shape = np.array([240, 320, 3])
        for i in range(flush_frames):
            self.get_next_frame()
        print('Camera: started!')

    def get_frame(self):
        ret, frame = self.cam.read()
        if ret:
            frame = resize_crop_mini(frame, 320, 240)
        return frame
    
    def get_next_frame(self):
        return self.get_frame()
    
    def close(self):
        self.cam.release()

    def __del__(self):
        self.close()


class Gelsight:
    # hardcoded values for the mini gelsight
    height = 240
    width = 320
    millimeters_per_pixel = 0.0634 # mini gel 18x24mm at 240x320

    def __init__(self, device_number, use_gpu=True, flush_frames=50, multiprocessed_cam=False):
        # Initialize camera
        if multiprocessed_cam:
            self.cam = MultiProcessedCamera(device_number, flush_frames)
        else:
            self.cam = Camera(device_number, flush_frames)


        self.interpolation = StrainInterpolation(self.height, self.width, 9, 7)

        # Initialize marker detection settings
        setting.init()

        # Get a frame from the camera
        frame = self.get_image()


        ### find marker masks
        mask = marker_detection.find_marker(frame)
        ### find marker centers
        self.mc = marker_detection.marker_center(mask, frame)

        #Depth Map initializatioin
        self.MASK_MARKERS_FLAG = True

        # Set up neural network for depth map
        net_path = os.path.join(PATH, './nnmini.pt')

        if use_gpu:
            gpuorcpu = 'cuda'
        else:
            gpuorcpu = 'cpu'

        # Initialize neural network
        self.nn = Reconstruction3D(self.height, self.width)
        self.nn.load_nn(net_path, gpuorcpu)

        # Initialize depth map by running the neural network on the first 50 frames
        for i in range(50):
            print('initializing depth map', i)
            frame = self.get_image()
            self.nn.get_depthmap(frame, self.MASK_MARKERS_FLAG)

        mc = self.mc
        mc_sorted1 = mc[mc[:,0].argsort()]
        mc1 = mc_sorted1[:setting.N_]
        mc1 = mc1[mc1[:,1].argsort()]

        mc_sorted2 = mc[mc[:,1].argsort()]
        mc2 = mc_sorted2[:setting.M_]
        mc2 = mc2[mc2[:,0].argsort()]


        """
        N_, M_: the row and column of the marker array
        x0_, y0_: the coordinate of upper-left marker
        dx_, dy_: the horizontal and vertical interval between adjacent markers
        """
        N_= setting.N_
        M_= setting.M_
        fps_ = setting.fps_
        x0_ = np.round(mc1[0][0])
        y0_ = np.round(mc1[0][1])
        dx_ = mc2[1, 0] - mc2[0, 0]
        dy_ = mc1[1, 1] - mc1[0, 1]

        self.marker_finder = find_marker.Matching(N_,M_,fps_,x0_,y0_,dx_,dy_)

    def get_image(self):
        frame = self.cam.get_next_frame()
        # frame = resize_crop_mini(frame, self.width, self.height)
        return frame

    def get_frame(self):
        
        # get_image_time = time.time()
        frame = self.get_image()
        # print('get_image_time', time.time() - get_image_time)
        # last_time = time.time()
        depthmap = self.nn.get_depthmap(frame, self.MASK_MARKERS_FLAG)
        # print('depthmap time', time.time() - last_time)
        

        ''' EXTRINSIC calibration ... 
        ... the order of points [x_i,y_i] | i=[1,2,3,4], are same 
        as they appear in plt.imshow() image window. Put them in 
        clockwise order starting from the topleft corner'''

        ### find marker masks
        mask = marker_detection.find_marker(frame)

        ### find marker centers
        mc = marker_detection.marker_center(mask, frame)

        # tm = time.time()
        ### matching init
        self.marker_finder.init(mc)

        ### matching
        self.marker_finder.run()
        # print(time.time() - tm)

        ### matching result
        """
        output: (Ox, Oy, Cx, Cy, Occupied) = flow
            Ox, Oy: N*M matrix, the x and y coordinate of each marker at frame 0
            Cx, Cy: N*M matrix, the x and y coordinate of each marker at current frame
            Occupied: N*M matrix, the index of the marker at each position, -1 means inferred. 
                e.g. Occupied[i][j] = k, meaning the marker mc[k] lies in row i, column j.
        """
        Ox, Oy, Cx, Cy, Occupied = self.marker_finder.get_flow()

        frame_data = np.array([])
        for i in range(len(Ox)):
            for j in range(len(Ox[i])):
                append_data = np.array([i, j, Ox[i][j], Oy[i][j], Cx[i][j], Cy[i][j]])
                frame_data = np.concatenate([frame_data, append_data], axis = 0)
        
        frame_data = frame_data.reshape((len(frame_data)//6, 6))
        # last_time = time.time()
        strain_x, strain_y = self.interpolation(frame_data)
        # print('interpolation time', time.time() - last_time)

        return frame, frame_data, depthmap, strain_x, strain_y
    
    def get_next_frame(self):
        return self.get_frame()
    
    def close(self):
        self.cam.close()

    def __del__(self):
        self.close()

import multiprocessing
import ctypes

class GelSightMultiprocessed:
    def __init__(self, device_number, use_gpu=True, flush_frames=50):
        self.device_number = device_number
        self.use_gpu = use_gpu
        self.flush_frames = flush_frames

        H = Gelsight.height
        W = Gelsight.width
        self.height = H
        self.width = W

        self.size = (H, W)
        self.last_frame_num = 0

        self.image_array = multiprocessing.Array(ctypes.c_uint8, H*W*3, lock=True)
        self.data_array = multiprocessing.Array(ctypes.c_float, H*W*3 + 7*9*6, lock=True)

        self.info_queue = multiprocessing.Queue()

        self.process = multiprocessing.Process(target=self._run, 
                                              args=(self.image_array, 
                                                    self.data_array, 
                                                    device_number, 
                                                    use_gpu, 
                                                    flush_frames, 
                                                    self.info_queue,
                                                    H, W))
        self.process.start()
        self.get_next_frame() # wait for the first frame to be ready
        print('GelSightMultiprocessed: started!')

    @staticmethod
    def _run(image_array, 
             data_array, 
             device_number, 
             use_gpu, 
             flush_frames, 
             frame_num_queue,
             H, W):
        
        image_np = np.frombuffer(image_array.get_obj(), dtype=np.uint8)
        data_np = np.frombuffer(data_array.get_obj(), dtype=np.float32)
        
        gelsight = Gelsight(device_number, use_gpu, flush_frames, multiprocessed_cam=True)

        frame_num = 0
        while True:
            frame, frame_data, depthmap, strain_x, strain_y = gelsight.get_frame()
            with image_array.get_lock():
                image_np[:] = frame.flatten()
            with data_array.get_lock():
                data_np[:H*W] = depthmap.flatten()
                data_np[H*W:H*W*2] = strain_x.flatten()
                data_np[H*W*2:H*W*3] = strain_y.flatten()
                data_np[H*W*3:] = frame_data.flatten()
            
            frame_num_queue.put(frame_num)
            frame_num += 1

    def get_frame(self, wait=False):
        # unpack the arrays
        H, W = self.size
        with self.image_array.get_lock():
            frame = np.frombuffer(self.image_array.get_obj(), dtype=np.uint8).reshape((H, W, 3)).copy()
        with self.data_array.get_lock():
            data_np = np.frombuffer(self.data_array.get_obj(), dtype=np.float32)
            depthmap = data_np[:H*W].reshape((H, W)).copy()
            strain_x = data_np[H*W:H*W*2].reshape((H, W)).copy()
            strain_y = data_np[H*W*2:H*W*3].reshape((H, W)).copy()
            frame_data = data_np[H*W*3:].reshape((7*9, 6)).copy()
        
        return frame, frame_data, depthmap, strain_x, strain_y
    
    def get_next_frame(self):
        frame_num = self.info_queue.get() # wait for the next frame to be ready
        # clear the queue
        while True:
            try:
                self.info_queue.get_nowait()
            except:
                break
        if frame_num != self.last_frame_num + 1:
            print('GelSightMultiprocessed: missed frames!', frame_num, self.last_frame_num)
        self.last_frame_num = frame_num
        return self.get_frame()
    
    def close(self):
        self.process.terminate()
        self.process.join()

    def __del__(self):
        self.close()


        

        


