import numpy as np
from skimage.color import rgb2gray
from imageio import imread
import matplotlib.pyplot as plt
# import cv2

GREYSCALE = 1

MAX_PIXEL_SIZE = 255

rgb_to_yiq_matrix = [[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]]


def read_image(filename, representation):
    im = imread(filename)
    im = im.astype(np.float64)
    im /= float(MAX_PIXEL_SIZE)
    if representation == GREYSCALE:
        return rgb2gray(im)
    return im


def imdisplay(filename, representation):
    im = read_image(filename, representation)
    if representation == GREYSCALE:
        plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    else:
        plt.imshow(im)
    plt.show()


def rgb2yiq(imRGB):
    return np.dot(imRGB,np.transpose(rgb_to_yiq_matrix)).clip(min=-1,max=1)

def yiq2rgb(imYIQ):
    return np.dot(imYIQ, np.transpose(np.linalg.inv(rgb_to_yiq_matrix))).clip(min=0, max=1)

def greyscale_histogram_equalize(im):
    hist, bins = np.histogram(im,MAX_PIXEL_SIZE+1,(0,1))
    cumulative_hist = np.cumsum(hist)
    cumulative_hist_normalized = (cumulative_hist/(im.shape[0]*im.shape[1]))*MAX_PIXEL_SIZE
    

def histogram_equalize(im_orig):
    pass



if __name__ == "__main__":
    im = rgb2yiq(read_image('sample.jpeg',2))
    print(5)
    plt.imshow(im[:,:,0], cmap='gray')
    plt.show()
    plt.imshow(yiq2rgb(im))
    plt.show()
    imdisplay('sample.jpeg',2)
    imdisplay('sample.jpeg',1)
