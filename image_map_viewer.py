#This code creates heat maps of the image maps found using the sparse encoders. 


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def display_convolutions_1D(in_tensor=None, in_file='', padding=4, out_file='', show_it=False):
    
    if in_file == '':
        data = in_tensor
    if in_file != '':
        data = np.load(in_file)

    # N is the total number of convolutions
    N = data.shape[2] #80
    
    #print(data.shape)

    # Ensure the resulting image is square
    filters_per_row = int(np.ceil(np.sqrt(N))) #9
    
    filter_size = [data.shape[0], 1]
    
    # Size of the result image including padding
    result_size = filters_per_row * (filter_size[0] + padding) - padding, \
                    filters_per_row * (filter_size[1] + padding) - padding
    
    # Initialize result image to all zeros
    result = np.zeros((result_size[0], result_size[1]))

    # Tile the filters into the result image
    filter_x = 0
    filter_y = 0
    for n in range(data.shape[2]):
        for c in range(data.shape[1]):
            
            if filter_x == filters_per_row:
                filter_y += 1
                filter_x = 0
            
            for i in range(filter_size[0]):
                    result[filter_y * (filter_size[0] + padding) + i, filter_x * padding] = \
                        data[i, c, n]
            filter_x += 1
   
    # Normalize image to 0-1
    min = result.min()
    max = result.max()
    result = (result - min) / (max - min)

    # Plot figure
    plt.figure(figsize=(20, 30))
    plt.axis('off')
    plt.imshow(result.T, cmap='hot', interpolation='nearest')

    # Save plot if filename is set
    if out_file != '':
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
        print('figure for %s outputted' % out_file)
    if show_it == True:
        plt.show()


def display_convolutions_2D(in_tensor=None, in_file='', padding=4, out_file='', show_it=False):
       
    if in_file == '':
        data = in_tensor
    if in_file != '':
        data = np.load(in_file)

    # N is the total number of convolutions
    N = data.shape[2] * data.shape[3]
    
    #print(data.shape)

    # Ensure the resulting image is square
    filters_per_row = int(np.ceil(np.sqrt(N)))
    # Assume the filters are square
    filter_size = data.shape[0], data.shape[1]
    # Size of the result image including padding
    result_size = filters_per_row * (filter_size[0] + padding) - padding, \
                    filters_per_row * (filter_size[1] + padding) - padding
    # Initialize result image to all zeros
    result = np.zeros((result_size[0], result_size[1]))

    # Tile the filters into the result image
    filter_x = 0
    filter_y = 0
    for n in range(data.shape[3]):
        for c in range(data.shape[2]):
            if filter_x == filters_per_row:
                filter_y += 1
                filter_x = 0
            for i in range(filter_size[0]):
                for j in range(filter_size[1]):
                    result[filter_y * (filter_size[0] + padding) + i, filter_x * (filter_size[1] + padding) + j] = \
                        data[i, j, c, n]
            filter_x += 1

    # Normalize image to 0-1
    min = result.min()
    max = result.max()
    result = (result - min) / (max - min)

    # Plot figure
    plt.figure(figsize=(20, 30))
    plt.axis('off')
    plt.imshow(result.T, cmap='hot', interpolation='nearest')

    # Save plot if filename is set
    if out_file != '':
        plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
        print('figure for %s outputted' % out_file)

    if show_it == True:
        plt.show()
 

def main():
    
    map_dir_1D = '/home/eden/Results/sparse_encoder-single_layer-ECG-1D/ImageMaps/'
    map_dir_2D = '/home/eden/Results/2D-CNN-ECG-sparse-encoder/ImageMaps/'
    
    if not os.path.exists(map_dir_1D + 'Images/'):
        os.makedirs(map_dir_1D + 'Images/')
    if not os.path.exists(map_dir_2D + 'Images/'):
        os.makedirs(map_dir_2D +'Images/')
            
    for ker in [20, 25, 30]:
        for sc in [float('%.5f' % x) for x in list(np.arange(0, 1e-4, 1.1e-5))]:
            
            in_file = map_dir_1D + 'ker-' + str(ker) + '/sc-' + str(sc) + '.npy'
            
            display_convolutions_1D(in_file = in_file,
                out_file = map_dir_1D + 'Images/1D_ECG_ker-' + str(ker) + '_sc-' + str(sc) + '.png')


    for ker in [10]:
        for sc in [float('%.5f' % x) for x in list(np.arange(0, 1e-4, 1.1e-5))]:
            
            in_file = map_dir_2D + 'ker-' + str(ker) + '/sc-' + str(sc) + '.npy'
            
            display_convolutions_2D(in_file = in_file,
                out_file = map_dir_2D + 'Images/2D_ECG_ker-' + str(ker) + '_sc-' + str(sc) + '.png')
    

if __name__ == "__main__":
   main()
