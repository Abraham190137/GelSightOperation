import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class StrainInterpolation:
    def __init__(self, image_hight, image_width, num_marker_columns, num_marker_rows):
        """
        Class to perform the inpolation of the strain data
        :param image_hight: hight of the image in pixel
        :param image_width: width of the image in pixel
        :param num_marker_columns: number of marker columns in the image
        :param num_marker_rows: number of marker rows in the image
        """
        self.image_hight = image_hight
        self.image_width = image_width
        self.num_marker_columns = num_marker_columns
        self.num_marker_rows = num_marker_rows

        # create the weights for the interpolation. The weights are 1/(distance to the marker + 0.25)
        self.weight_radius = np.array([int(image_hight/(num_marker_rows+1)), int(image_width/(num_marker_columns+1))])
        self.weights = np.zeros(2*(self.weight_radius) + 1)
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                # self.weights[i,j] =  min([i/self.weight_radius[0], 
                #                           j/self.weight_radius[1], 
                #                           (self.weights.shape[0] - i)/self.weight_radius[0],
                #                           (self.weights.shape[1] - j)/self.weight_radius[1]])
                
                # self.weights[i,j] = max(0, max(self.weight_radius) - np.linalg.norm(np.array([i,j])-self.weight_radius))
                
                # self.weights[i,j] = 1/(np.linalg.norm(np.array([i,j])-self.weight_radius)+1)

                self.weights[i,j] = (self.weight_radius[0] - abs(i-self.weight_radius[0]))*(self.weight_radius[1] - abs(j-self.weight_radius[1]))
        
        # The edges and corners need to be larger, incase the marker is far from the edge
        # This is for the right edge (math is simpler)
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
        # This is for the bottom right corner (math is simpler)
        self.corner_weights = np.zeros([3*(self.weight_radius[0]) + 1, 3*(self.weight_radius[1]) + 1])

        # The top is the same as the right edge weights
        self.corner_weights[:self.weight_radius[0], :] = self.right_edge_weights[:self.weight_radius[0], :]

        # The left is the same as the bottom edge weights
        self.corner_weights[:, :self.weight_radius[1]] = self.bottom_edge_weights[:, :self.weight_radius[1]]

        for i in range(self.weight_radius[0], self.corner_weights.shape[0]):
            for j in range(self.weight_radius[1], self.corner_weights.shape[1]):
                self.corner_weights[i,j] = (0.5*(3*self.weight_radius[0] - i))*(0.5*(3*self.weight_radius[1] - j))

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
        :param frame_data: data of the markers in the image
        :return: interpolated strain data
        """


        strain_x = np.zeros((self.image_hight, self.image_width))
        strain_y = np.zeros((self.image_hight, self.image_width))
        total_weight = np.zeros((self.image_hight, self.image_width)) + 1e-5

        # loop over all markers. Point is its location in the image, dx and dy are the strain values

        for idx in range(self.num_marker_columns*self.num_marker_rows):
            dy = frame_data[idx,4] - frame_data[idx,2]
            dx = frame_data[idx,5] - frame_data[idx,3]
            x = frame_data[idx, 2].astype(int)
            y = frame_data[idx, 3].astype(int)
            i = frame_data[idx, 0].astype(int)
            j = frame_data[idx, 1].astype(int)
            # print("i: ", i, "j: ", j, "x: ", x, "y: ", y)

            # if the marker is near the image, crop the weights
            strain_low_i = max(0, y-self.weight_radius[0])
            weight_low_i = max(0, self.weight_radius[0]-y)

            strain_high_i = min(self.image_hight, y+self.weight_radius[0]+1)
            weight_high_i = self.weights.shape[0] + min(0, self.image_hight - (y+self.weight_radius[0]+1))

            strain_low_j = max(0, x-self.weight_radius[1])
            weight_low_j = max(0, self.weight_radius[1]-x)

            strain_high_j = min(self.image_width, x+self.weight_radius[1]+1)
            weight_high_j = self.weights.shape[1] + min(0, self.image_width - (x+self.weight_radius[1]+1))

            use_weights = self.weights

        
            
            if i == 0:
                use_weights = self.top_edge_weights
                strain_low_i = max(0, y-2*self.weight_radius[0])
                weight_low_i = self.top_edge_weights.shape[0] - (strain_high_i - strain_low_i)
                weight_high_i = self.top_edge_weights.shape[0]

            elif i == self.num_marker_rows-1:
                use_weights = self.bottom_edge_weights
                strain_high_i = min(self.image_hight, y+2*self.weight_radius[0]+1)
                weight_high_i = strain_high_i - strain_low_i
                weight_low_i = 0

            if j == 0:
                strain_low_j = max(0, x-2*self.weight_radius[1])
                weight_low_j = self.left_edge_weights.shape[1] - (strain_high_j - strain_low_j)
                weight_high_j = self.left_edge_weights.shape[1]
                if i == 0:
                    use_weights = self.top_left_corner_weights
                elif i == self.num_marker_rows-1:
                    use_weights = self.bottom_left_corner_weights
                else:
                    use_weights = self.left_edge_weights

            elif j == self.num_marker_columns-1:
                strain_high_j = min(self.image_width, x+2*self.weight_radius[1]+1)
                weight_high_j = strain_high_j - strain_low_j
                weight_low_j = 0
                if i == 0:
                    use_weights = self.top_right_corner_weights
                elif i == self.num_marker_rows-1:
                    use_weights = self.bottom_right_corner_weights
                else:
                    use_weights = self.right_edge_weights
                
            
            # add the strain to the total strain
            # print("strain_low_i: ", strain_low_i, "strain_high_i: ", strain_high_i, "strain_low_j: ", strain_low_j, "strain_high_j: ", strain_high_j)
            # print("weight_low_i: ", weight_low_i, "weight_high_i: ", weight_high_i, "weight_low_j: ", weight_low_j, "weight_high_j: ", weight_high_j)
            strain_x[strain_low_i:strain_high_i, strain_low_j:strain_high_j] += dx*use_weights[weight_low_i:weight_high_i, weight_low_j:weight_high_j]
            strain_y[strain_low_i:strain_high_i, strain_low_j:strain_high_j] += dy*use_weights[weight_low_i:weight_high_i, weight_low_j:weight_high_j]

            # add the weights to the total weight
            total_weight[strain_low_i:strain_high_i, strain_low_j:strain_high_j] += use_weights[weight_low_i:weight_high_i, weight_low_j:weight_high_j]

            # plt.figure()
            # plt.imshow(strain_x/total_weight, interpolation='none')
            # plt.colorbar()
            # plt.scatter(points[:,0], points[:,1], c='k')
            # plt.show()

        # divide by the total weight to get the mean
        # plt.show()
        strain_x = strain_x/total_weight
        strain_y = strain_y/total_weight
            
        return strain_x, strain_y
                


def create_strain_images(frame_data) -> Tuple[np.ndarray, np.ndarray]:
        from scipy
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

if __name__ == '__main__':
    import pickle
    import matplotlib.pyplot as plt
    import time

    with open('frame_data.pickle', 'rb') as f:
        data = pickle.load(f)
    
    interpolate = StrainInterpolation(240, 320, 9, 7)
    # interpolate = StrainInterpolation(270, 380, 9, 7)
    last_time = time.time()
    x_inter, y_inter = interpolate(data)
    print(time.time() - last_time)
    points = data[:, 2:4]

    plt.figure()
    plt.imshow(interpolate.weights, interpolation='none')
    plt.colorbar()

    # make sure the images are pixelated
    plt.figure()
    plt.imshow(x_inter, interpolation='none')
    plt.colorbar()
    plt.scatter(points[:,0], points[:,1], c='k')

    plt.figure()
    plt.imshow(y_inter, interpolation='none')
    plt.colorbar()
    plt.scatter(points[:,0], points[:,1], c='k')
    plt.show()