import cv2
import find_marker
from tracking import *
import setting
import numpy as np
import os
from gsdevice_threaded import Camera
import gs3drecon
# import rospy

class GelsightDataCollector:
    def __init__(self, camera_device_number, use_gpu=True):
        # Initialize threaded camera
        self.dev = Camera(camera_device_number)

        # Initialize marker detection settings
        setting.init()

        # Get a frame from the camera
        frame = self.dev.get_image()
        ### find marker masks
        mask = marker_detection.find_marker(frame)
        ### find marker centers
        self.mc = marker_detection.marker_center(mask, frame)

        #Depth Map initializatioin
        self.MASK_MARKERS_FLAG = True

        # Set up neural network for depth map
        net_file_path = './nnmini.pt'
        model_file_path = "."
        net_path = os.path.join(model_file_path, net_file_path)

        if use_gpu:
            gpuorcpu = 'cuda'
        else:
            gpuorcpu = 'cpu'

        # Initialize neural network
        self.nn = gs3drecon.Reconstruction3D(self.dev)
        self.nn.load_nn(net_path, gpuorcpu)

        # Initialize depth map by running the neural network on the first 50 frames
        for i in range(50):
            print('initializing depth map', i)
            frame = self.dev.get_next_image()
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


    def collect_frame_data(self):

        frame = self.dev.get_next_image()

        depthmap = self.nn.get_depthmap(frame, self.MASK_MARKERS_FLAG)
        

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

        return frame, depthmap, frame_data
        
     
    def clean_up(self):
        self.dev.close()



