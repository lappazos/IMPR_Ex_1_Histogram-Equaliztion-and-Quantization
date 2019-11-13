import numpy as np
from skimage.color import rgb2gray
from imageio import imread
import matplotlib.pyplot as plt


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


def perform_func_grey_or_rgb(im_orig, func, *args):
    if len(im_orig.shape) > 2:
        im_yiq = rgb2yiq(im_orig)
        new_y = func(im_yiq[:, :, 0], *args)
        im_yiq[:, :, 0] = new_y[0]
        im_eq = yiq2rgb(im_yiq)
        return [im_eq] + new_y[1:]
    return func(im_orig, *args)


def histogram_equalize(im_orig):
    def greyscale_histogram_equalize(im):
        hist, _ = np.histogram(im, [i / MAX_PIXEL_SIZE for i in range(MAX_PIXEL_SIZE + 2)])
        cumulative_hist = np.cumsum(hist)
        c_m = cumulative_hist[np.argmax(cumulative_hist > 0)]
        cumulative_hist_normalized = np.around(
            ((cumulative_hist - c_m) / (cumulative_hist[255] - c_m)) * MAX_PIXEL_SIZE)
        im_255 = expand_to_255(im)
        lookup_table = lambda i: cumulative_hist_normalized[i]
        im_eq = normalize_0_to_1(lookup_table(im_255))
        return [im_eq.clip(min=0, max=1), hist,
                np.histogram(im_eq, [i / MAX_PIXEL_SIZE for i in range(MAX_PIXEL_SIZE)])]

    return perform_func_grey_or_rgb(im_orig, greyscale_histogram_equalize)


def quantize(im_orig, n_quant, n_iter):
    def quantization_process(error, num_iter, num_quant, probability_hist, q, z, z_pz):
        for iteration in range(num_iter):
            if iteration != 0:
                temp_z = [0]
                for zi in range(1, num_quant):
                    temp_z.append(np.around((q[zi - 1] + q[zi]) / 2))
                temp_z.append(255)
                if z == temp_z:
                    break
                z = temp_z
            error_i = 0
            for i in range(num_quant):
                q[i] = np.around(np.sum(z_pz[z[i]:z[i + 1] + 1]) / np.sum(probability_hist[z[i]:z[i + 1] + 1]))
                error_i += np.sum(np.square(np.arange(z[i], z[i + 1] + 1) - q[i]) * probability_hist[z[i]:z[i + 1] + 1])
            error[iteration] = error_i
        return z

    def quantize_grey_scale(image_orig, num_quant, num_iter):
        im_255 = expand_to_255(image_orig)
        hist, _ = np.histogram(im_255, [i for i in range(MAX_PIXEL_SIZE + 2)])
        cumulative_hist = np.cumsum(hist)
        probability_hist = hist / cumulative_hist[255]
        z_pz = probability_hist * np.arange(MAX_PIXEL_SIZE + 1)
        z = [0]
        for zi in range(1, num_quant):
            z.append(np.argmax(cumulative_hist > (cumulative_hist[255] / num_quant) * zi))
        z.append(255)
        q = []
        error = np.empty((num_iter,))
        z = quantization_process(error, num_iter, num_quant, probability_hist, q, z, z_pz)
        im_quant = None
        for i in range(num_quant):
            im_quant = np.where(z[i] <= im_255 <= z[i + 1], q[i], im_255)
        return [normalize_0_to_1(im_quant), error]

    return perform_func_grey_or_rgb(im_orig, quantize_grey_scale, n_quant, n_iter)


if __name__ == "__main__":
    # im = rgb2yiq(read_image('sample.jpeg', 2))
    # print(5)
    # plt.imshow(im[:, :, 0], cmap='gray')
    # plt.show()
    # plt.imshow(yiq2rgb(im))
    # plt.show()
    # imdisplay('sample.jpeg', 2)
    # imdisplay('sample.jpeg', 1)

    imdisplay('jerusalem.jpg', 2)
    a = histogram_equalize(read_image('jerusalem.jpg', 2))[0]
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


