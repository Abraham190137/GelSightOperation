from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import numpy as np
from typing import List, Tuple, Dict

def create_strain_images(frame_data) -> Tuple[np.ndarray, np.ndarray]:
        
        # print(data_firststep)
        # print(len(frame_data))
        # print(frame_data)
        dx = frame_data [:,4] - frame_data [:,2]
        dy = frame_data [:,5] - frame_data [:,3]

        points = frame_data[:, 2:4]

        # Generate the 320x240 grid of points
        xi = np.linspace(0, 320, 320)
        yi = np.linspace(0, 240, 240)
        xi, yi = np.meshgrid(xi, yi)

        # Perform the interpolation
        strain_x = griddata(points, dx.ravel(), (xi, yi), method='nearest')
        # strain_x = gaussian_filter(strain_x, 10)
        
        strain_y = griddata(points, dy.ravel(), (xi, yi), method='nearest')
        # strain_y = gaussian_filter(strain_y, 10)
                
        return strain_x, strain_y