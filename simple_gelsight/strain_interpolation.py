import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class StrainInterpolation:
    def __init__(self, image_hight=240, image_width=320, num_marker_rows=7, num_marker_columns=9):
        """
        Class to perform the bilinar inpolation of the data on a 2d grid of points. Used in GelSight for the strain data.
        :param image_hight: hight of the image in pixel
        :param image_width: width of the image in pixel
        :param num_marker_columns: number of marker columns in the image
        :param num_marker_rows: number of marker rows in the image
        """
        self.image_hight = image_hight
        self.image_width = image_width
        self.num_marker_columns = num_marker_columns
        self.num_marker_rows = num_marker_rows

        # To perform a bilinear interpolation, we will be using a weighted average. To speed up the compute process at run time, 
        # we will precompute the weights and store them in a numpy array.
        # Note: The weights are based on the initial point configuration. If the points move too far, the weights will be wrong, but its good enough.

        # Weight raduis = size of the weight array/2, or the distance from the center to the edge
        self.weight_radius = np.array([int(image_hight/(num_marker_rows+1)), int(image_width/(num_marker_columns+1))])
        self.weights = np.zeros(2*(self.weight_radius) + 1)

        # Calculate the weights using the formual found here: https://en.wikipedia.org/wiki/Bilinear_interpolation
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                self.weights[i,j] = (self.weight_radius[0] - abs(i-self.weight_radius[0]))*(self.weight_radius[1] - abs(j-self.weight_radius[1]))
        
        # The edges and corners need to be larger, incase the marker is far from the edge. 
        # The same formula is used, but the "next marker" is assumed to be twice as far away in the direction of the edge of the image,
        # resutling in an array of size 3*(weight_radius) + 1, centered at weight_radius
        # This is for the right edge 
        self.right_edge_weights = np.zeros([2*(self.weight_radius[0]) + 1, 3*(self.weight_radius[1]) + 1])
        self.right_edge_weights[:, :self.weight_radius[1]] = self.weights[:, :self.weight_radius[1]] # The left half is the same
        for i in range(self.right_edge_weights.shape[0]):
            for j in range(self.weight_radius[1], self.right_edge_weights.shape[1]):
                self.right_edge_weights[i,j] = (self.weight_radius[0] - abs(i-self.weight_radius[0]))*(0.5*(3*self.weight_radius[1] - j))

        # The left edge is the same as the right edge, but flipped
        self.left_edge_weights = np.flip(self.right_edge_weights, axis=1)

        # The bottom edge is calculated similarly
        self.bottom_edge_weights = np.zeros([3*(self.weight_radius[0]) + 1, 2*(self.weight_radius[1]) + 1])
        self.bottom_edge_weights[:self.weight_radius[0], :] = self.weights[:self.weight_radius[0], :] # The top half is the same
        for i in range(self.weight_radius[0], self.bottom_edge_weights.shape[0]):
            for j in range(self.bottom_edge_weights.shape[1]):
                self.bottom_edge_weights[i,j] = (0.5*(3*self.weight_radius[0] - i))*(self.weight_radius[1] - abs(j-self.weight_radius[1]))

        # The top edge is the same as the bottom edge, but flipped
        self.top_edge_weights = np.flip(self.bottom_edge_weights, axis=0)
        
        # Similarly, the corners need to be larger
        # This is for the bottom right corner 
        self.corner_weights = np.zeros([3*(self.weight_radius[0]) + 1, 3*(self.weight_radius[1]) + 1])

        # The top is the same as the right edge weights
        self.corner_weights[:self.weight_radius[0], :] = self.right_edge_weights[:self.weight_radius[0], :]

        # The left is the same as the bottom edge weights
        self.corner_weights[:, :self.weight_radius[1]] = self.bottom_edge_weights[:, :self.weight_radius[1]]

        for i in range(self.weight_radius[0], self.corner_weights.shape[0]):
            for j in range(self.weight_radius[1], self.corner_weights.shape[1]):
                self.corner_weights[i,j] = (0.5*(3*self.weight_radius[0] - i))*(0.5*(3*self.weight_radius[1] - j))

        # The other corners are the same as the bottom right corner, but flipped
        self.bottom_right_corner_weights = self.corner_weights
        self.bottom_left_corner_weights = np.flip(self.corner_weights, axis=1)
        self.top_left_corner_weights = np.flip(self.bottom_left_corner_weights, axis=0)
        self.top_right_corner_weights = np.flip(self.bottom_right_corner_weights, axis=0)

        # # Plot the edge weights
        # plt.figure()
        # plt.imshow(self.right_edge_weights, interpolation='none')
        # plt.title('right edge weights')
        # plt.colorbar()

        # plt.figure()
        # plt.imshow(self.left_edge_weights, interpolation='none')
        # plt.title('left edge weights')
        # plt.colorbar()

        # plt.figure()
        # plt.imshow(self.bottom_edge_weights, interpolation='none')
        # plt.title('bottom edge weights')
        # plt.colorbar()

        # plt.figure()
        # plt.imshow(self.top_edge_weights, interpolation='none')
        # plt.title('top edge weights')
        # plt.colorbar()

        # # Plot the corner weights
        # plt.figure()
        # plt.imshow(self.bottom_right_corner_weights, interpolation='none')
        # plt.title('bottom right corner weights')
        # plt.colorbar()

        # plt.figure()
        # plt.imshow(self.bottom_left_corner_weights, interpolation='none')
        # plt.title('bottom left corner weights')
        # plt.colorbar()

        # plt.figure()
        # plt.imshow(self.top_left_corner_weights, interpolation='none')
        # plt.title('top left corner weights')
        # plt.colorbar()

        # plt.figure()
        # plt.imshow(self.top_right_corner_weights, interpolation='none')
        # plt.title('top right corner weights')
        # plt.colorbar()

        # plt.show()


    def __call__(self, frame_data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate the strain data
        :param frame_data: data of the markers in the image in the form of [i, j, y, x, dy, dx]
        i is the row of the marker, j is the column of the marker, y is the original y location of the marker in the image, 
        x is the original x location of the marker in the image, dy is new y location of the marker in the image, 
        dx is the new x location of the marker in the image
        :return: strain_x, strain_y
        """


        strain_x = np.zeros((self.image_hight, self.image_width))
        strain_y = np.zeros((self.image_hight, self.image_width))
        total_weight = np.zeros((self.image_hight, self.image_width)) + 1e-5

        # loop over all markers. Point is its location in the image, dx and dy are the strain values

        for idx in range(self.num_marker_columns*self.num_marker_rows):
            # get the marker data
            dy = frame_data[idx,4] - frame_data[idx,2]
            dx = frame_data[idx,5] - frame_data[idx,3]
            x = frame_data[idx, 2].astype(int)
            y = frame_data[idx, 3].astype(int)
            i = frame_data[idx, 0].astype(int)
            j = frame_data[idx, 1].astype(int)

            # if the marker is near the edge of image, we will need to crop both the weights and the strain image.
            # Below calculates the indicies of the strain image and the weights that will be used
            strain_low_i = max(0, y-self.weight_radius[0])
            weight_low_i = max(0, self.weight_radius[0]-y)

            strain_high_i = min(self.image_hight, y+self.weight_radius[0]+1)
            weight_high_i = self.weights.shape[0] + min(0, self.image_hight - (y+self.weight_radius[0]+1))

            strain_low_j = max(0, x-self.weight_radius[1])
            weight_low_j = max(0, self.weight_radius[1]-x)

            strain_high_j = min(self.image_width, x+self.weight_radius[1]+1)
            weight_high_j = self.weights.shape[1] + min(0, self.image_width - (x+self.weight_radius[1]+1))

            use_weights = self.weights # default to the normal weights

        
            # If the maker is one of the edges or corners, we need to use the edge or corner weights instead
            if i == 0: # top edge
                use_weights = self.top_edge_weights
                strain_low_i = max(0, y-2*self.weight_radius[0])
                weight_low_i = self.top_edge_weights.shape[0] - (strain_high_i - strain_low_i)
                weight_high_i = self.top_edge_weights.shape[0]

            elif i == self.num_marker_rows-1: # bottom edge
                use_weights = self.bottom_edge_weights
                strain_high_i = min(self.image_hight, y+2*self.weight_radius[0]+1)
                weight_high_i = strain_high_i - strain_low_i
                weight_low_i = 0

            if j == 0: # left edge
                strain_low_j = max(0, x-2*self.weight_radius[1])
                weight_low_j = self.left_edge_weights.shape[1] - (strain_high_j - strain_low_j)
                weight_high_j = self.left_edge_weights.shape[1]
                if i == 0: # top left corner
                    use_weights = self.top_left_corner_weights
                elif i == self.num_marker_rows-1: # bottom left corner
                    use_weights = self.bottom_left_corner_weights
                else: 
                    use_weights = self.left_edge_weights

            elif j == self.num_marker_columns-1: # right edge
                strain_high_j = min(self.image_width, x+2*self.weight_radius[1]+1)
                weight_high_j = strain_high_j - strain_low_j
                weight_low_j = 0
                if i == 0: # top right corner
                    use_weights = self.top_right_corner_weights
                elif i == self.num_marker_rows-1: # bottom right corner
                    use_weights = self.bottom_right_corner_weights
                else:
                    use_weights = self.right_edge_weights
                
            
            # add the strain to the total strain
            strain_x[strain_low_i:strain_high_i, strain_low_j:strain_high_j] += dx*use_weights[weight_low_i:weight_high_i, weight_low_j:weight_high_j]
            strain_y[strain_low_i:strain_high_i, strain_low_j:strain_high_j] += dy*use_weights[weight_low_i:weight_high_i, weight_low_j:weight_high_j]

            # add the weights to the total weight
            total_weight[strain_low_i:strain_high_i, strain_low_j:strain_high_j] += use_weights[weight_low_i:weight_high_i, weight_low_j:weight_high_j]

        # divide by the total weight to get the mean
        strain_x = strain_x/total_weight
        strain_y = strain_y/total_weight
            
        return strain_x, strain_y