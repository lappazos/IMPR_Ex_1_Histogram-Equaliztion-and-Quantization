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
    im = normalize_0_to_1(im)
    if representation == GREYSCALE:
        return rgb2gray(im)
    return im


def normalize_0_to_1(im):
    im = im.astype(np.float64)
    im /= float(MAX_PIXEL_SIZE)
    return im

def expand_to_255(im):
    return np.around(im * MAX_PIXEL_SIZE).astype(np.uint8)


def imdisplay(filename, representation):
    im = read_image(filename, representation)
    if representation == GREYSCALE:
        plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    else:
        plt.imshow(im)
    plt.show()


def rgb2yiq(imRGB):
    return np.dot(imRGB, np.transpose(rgb_to_yiq_matrix)).clip(min=-1, max=1)


def yiq2rgb(imYIQ):
    return np.dot(imYIQ, np.transpose(np.linalg.inv(rgb_to_yiq_matrix))).clip(min=0, max=1)


def greyscale_histogram_equalize(im):
    hist, _ = np.histogram(im, [i / MAX_PIXEL_SIZE for i in range(MAX_PIXEL_SIZE+2)])
    cumulative_hist = np.cumsum(hist)
    c_m = cumulative_hist[np.argmax(cumulative_hist > 0)]
    cumulative_hist_normalized = np.around(((cumulative_hist - c_m) / (cumulative_hist[255] - c_m)) * MAX_PIXEL_SIZE)
    im_255 = expand_to_255(im)
    lookup_table = lambda i: cumulative_hist_normalized[i]
    im_eq = normalize_0_to_1(lookup_table(im_255))
    return [im_eq.clip(min=0,max=1), hist, np.histogram(im_eq, [i / MAX_PIXEL_SIZE for i in range(MAX_PIXEL_SIZE)])]


def histogram_equalize(im_orig):
    if len(im_orig.shape) > 2:
        im_yiq = rgb2yiq(im_orig)
        equalized_y = greyscale_histogram_equalize(im_yiq[:, :, 0])
        im_yiq[:, :, 0] = equalized_y[0]
        im_eq = yiq2rgb(im_yiq)
        return [im_eq, equalized_y[1], equalized_y[2]]
    return greyscale_histogram_equalize(im_orig)

def quantize_grey_scale(im_orig, n_quant, n_iter):
    im_255 = expand_to_255(im_orig)
    hist, _ = np.histogram(im_255, [i for i in range(MAX_PIXEL_SIZE + 2)])
    cumulative_hist = np.cumsum(hist)
    probability_hist = hist/cumulative_hist[255]
    z_pz=probability_hist*np.arange(MAX_PIXEL_SIZE+1)
    zs = [0]
    for z in range(1,n_quant):
        zs.append(np.argmax(cumulative_hist > (cumulative_hist[255]/n_quant)*z))
    zs.append(255)
    q=[]
    for iter in n_iter:
        
        for i in n_quant:
            q.append(np.around(np.sum(z_pz[zs[i]:zs[i+1]+1])/np.sum(probability_hist[zs[i]:zs[i+1]+1])))






def quantize (im_orig, n_quant, n_iter):




if __name__ == "__main__":
    # im = rgb2yiq(read_image('sample.jpeg', 2))
    # print(5)
    # plt.imshow(im[:, :, 0], cmap='gray')
    # plt.show()
    # plt.imshow(yiq2rgb(im))
    # plt.show()
    # imdisplay('sample.jpeg', 2)
    # imdisplay('sample.jpeg', 1)
    imdisplay('jerusalem.jpg',2)
    a=histogram_equalize(read_image('jerusalem.jpg',2))[0]
    plt.imshow(a)
    plt.show()
    imdisplay('monkey.jpg', 2)
    a = histogram_equalize(read_image('monkey.jpg', 2))[0]
    plt.imshow(a)
    plt.show()
    imdisplay('low_contrast.jpg', 2)
    a = histogram_equalize(read_image('low_contrast.jpg', 2))[0]
    plt.imshow(a)
    plt.show()
