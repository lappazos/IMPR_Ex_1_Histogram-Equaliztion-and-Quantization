import numpy as np
from skimage.color import rgb2gray
from imageio import imread
import matplotlib.pyplot as plt

GREYSCALE = 1

MAX_PIXEL_SIZE = 255

rgb_to_yiq_matrix = [[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]]

intensities = np.array([i for i in range(256)])


def read_image(filename, representation):
    """
    
    :param filename:
    :param representation:
    :return:
    """
    im = None
    try:
        im = imread(filename)
    except Exception:  # internet didnt have specific documentation regarding the exceptions this func throws
        print("File Problem")
        exit()
    representation_check(representation)
    im = normalize_0_to_1(im)
    if representation == GREYSCALE:
        return rgb2gray(im)
    return im.astype(np.float64)


def representation_check(representation):
    if representation not in [1, 2]:
        print("Representation code not exist. please use 1 or 2")
        exit()


def normalize_0_to_1(im):
    im = im.astype(np.float64)
    im /= MAX_PIXEL_SIZE
    return im


def expand_to_255(im):
    return (im * MAX_PIXEL_SIZE).round().astype(np.uint8)


def imdisplay(filename, representation):
    im = read_image(filename, representation)
    if representation == GREYSCALE:
        plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    else:
        plt.imshow(im)
    plt.show()


def rgb2yiq(imRGB):
    return np.dot(imRGB, np.transpose(rgb_to_yiq_matrix)).astype(np.float64)


def yiq2rgb(imYIQ):
    return np.dot(imYIQ, np.transpose(np.linalg.inv(rgb_to_yiq_matrix))).astype(np.float64)


def perform_func_grey_or_rgb(im_orig, func, *args):
    if np.max(im_orig) > 1 or np.min(im_orig) < 0:
        print("Wrong Format 1")
        exit()
    if len(im_orig.shape) > 2:
        im_yiq = rgb2yiq(im_orig)
        new_y = func(im_yiq[:, :, 0], *args)
        im_yiq[:, :, 0] = new_y[0]
        im_eq = yiq2rgb(im_yiq)
        return [im_eq.astype(np.float64)] + new_y[1:]
    return func(im_orig, *args)


def histogram_equalize(im_orig):
    def greyscale_histogram_equalize(im):
        hist, _ = np.histogram(im, MAX_PIXEL_SIZE + 1)
        cumulative_hist = np.cumsum(hist)
        c_m = cumulative_hist[np.argmax(cumulative_hist > 0)]
        cumulative_hist_normalized = np.rint(
            ((cumulative_hist - c_m) / (cumulative_hist[255] - c_m)) * MAX_PIXEL_SIZE)
        im_255 = expand_to_255(im)
        lookup_table = lambda i: cumulative_hist_normalized[i]
        im_eq = normalize_0_to_1(lookup_table(im_255))
        return [im_eq.clip(min=0, max=1).astype(np.float64), hist,
                np.histogram(im_eq, MAX_PIXEL_SIZE + 1)[0]]

    return perform_func_grey_or_rgb(im_orig, greyscale_histogram_equalize)


def quantize(im_orig, n_quant, n_iter):
    if n_iter < 1:
        print("need at least 1 iter")
        exit()

    def quantization_process(num_iter, num_quant, probability_hist, z):
        error = []
        q = []
        for iteration in range(num_iter + 1):
            q = []
            for i in range(num_quant):
                q.append(np.sum(probability_hist[z[i]:z[i + 1]] * intensities[z[i]:z[i + 1]]) / np.sum(
                    probability_hist[z[i]:z[i + 1]]))
            temp_z = np.empty(num_quant + 1)
            temp_z[0] = 0
            for zi in range(1, num_quant):
                temp_z[zi] = np.ceil((q[zi - 1] + q[zi]) / 2.0)
            temp_z[num_quant] = 255
            if np.array_equal(z, temp_z):
                return z, q, error
            z = temp_z.astype(np.uint64)

            error_i = 0
            for i in range(num_quant):
                error_i += (
                        np.square(np.rint(q[i]) - intensities[z[i]: z[i + 1]]) * probability_hist[z[i]:z[i + 1]]).sum()
            error.append(error_i)

        return z, np.rint(q), error

    def quantize_grey_scale(image_orig, num_quant, num_iter):
        im_255 = expand_to_255(image_orig)
        hist, _ = np.histogram(im_255, MAX_PIXEL_SIZE + 1)
        cumulative_hist = np.cumsum(hist)
        probability_hist = (hist / cumulative_hist[255])
        z = np.empty(n_quant + 1)
        z[0] = 0
        for zi in range(1, num_quant):
            z[zi] = np.argmax(cumulative_hist > (cumulative_hist[255] / num_quant) * zi)
        z[n_quant] = 255
        z = z.astype(np.uint64)
        z, q, error = quantization_process(num_iter, num_quant, probability_hist, z)
        lookup_table = np.empty(MAX_PIXEL_SIZE + 1)
        for i in range(num_quant):
            lookup_table[z[i]:z[i + 1]] = q[i]
        im_quant = lookup_table[im_255]
        return [normalize_0_to_1(im_quant).astype(np.float64), error]

    return perform_func_grey_or_rgb(im_orig, quantize_grey_scale, n_quant, n_iter)
