import numpy as np
from skimage.color import rgb2gray
from imageio import imread
import matplotlib.pyplot as plt

MIN_INTENSITY = 0

AT_LEAST_ITER = "need at least 1 iter"

HISTOGRAM = 0

RGB_DIM = 3

MIN_IMG_VAL = 0

MAX_IMG_VAL = 1

IMAGE_NOT_IN_RANGE_ = "Image not in range 0-1"

REPRESENTATION_ERROR = "Representation code not exist. please use 1 or 2"

RGB = 2

GREY_SCALE = 1

FILE_PROBLEM = "File Problem"

GREYSCALE = 1

MAX_INTENSITY = 255

rgb_to_yiq_matrix = [[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]]

intensities = np.array([i for i in range(256)])


def read_image(filename, representation):
    """
    This function returns an image, make sure the output image is represented by a matrix of type
    np.float64 with intensities (either grayscale or RGB channel intensities) normalized to the range [0, 1].
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image (1) or an RGB image (2).
    :return: an image
    """
    im = None
    try:
        im = imread(filename)
    except Exception:  # internet didnt have specific documentation regarding the exceptions this func throws
        print(FILE_PROBLEM)
        exit()
    representation_check(representation)
    im = normalize_0_to_1(im)
    if representation == GREYSCALE:
        return rgb2gray(im)
    return im.astype(np.float64)


def representation_check(representation):
    """
    check if representation code is valid
    :param representation: representation code
    """
    if representation not in [GREY_SCALE, RGB]:
        print(REPRESENTATION_ERROR)
        exit()


def normalize_0_to_1(im):
    """
    normalize picture
    :param im: image in range 0-255
    :return: image in range [0,1]
    """
    if im.dtype != np.float64:
        im = im.astype(np.float64)
        im /= MAX_INTENSITY
    return im


def expand_to_255(im):
    """
    expand picture
    :param im: image in range [0,1]
    :return: image in range 0-255
    """
    if im.dtype != np.uint8:
        return (im * MAX_INTENSITY).round().astype(np.uint8)
    return im


def imdisplay(filename, representation):
    """
    o display an image in a given representation
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
    image (1) or an RGB image (2).
    """
    im = read_image(filename, representation)
    if representation == GREYSCALE:
        plt.imshow(im, cmap='gray', vmin=MIN_IMG_VAL, vmax=MAX_IMG_VAL)
    else:
        plt.imshow(im)
    plt.show()


def rgb2yiq(imRGB):
    """
    transform an RGB image into the YIQ color space
    :param imRGB: height×width×3 np.float64 matrices, imRGB is
    in the [0, 1] range
    :return: height×width×3 np.float64 matrices, e the Y channel is in the [0,1] range,
    the I and Q channels are in the [−1, 1] range
    """
    return np.dot(imRGB, np.transpose(rgb_to_yiq_matrix)).astype(np.float64)  # transpose so the dimensions will match


def yiq2rgb(imYIQ):
    """
    transform an YIQ image into the RGB color space
    :param imYIQ: height×width×3 np.float64 matrices, e the Y channel is in the [0,1] range,
    the I and Q channels are in the [−1, 1] range
    :return: height×width×3 np.float64 matrices, imRGB is
    in the [0, 1] range
    """
    return np.dot(imYIQ, np.transpose(np.linalg.inv(rgb_to_yiq_matrix))).astype(
        np.float64)  # transpose so the dimensions will match


def perform_func_grey_or_rgb(im_orig, func, *args):
    """
    perform func grey or rgb
    :param im_orig: image
    :param func: func to perform
    :param args: args for func
    :return: func return
    """
    if np.max(im_orig) > MAX_IMG_VAL or np.min(im_orig) < MIN_IMG_VAL:
        print(IMAGE_NOT_IN_RANGE_)
        exit()
    if len(im_orig.shape) == RGB_DIM:
        im_yiq = rgb2yiq(im_orig)
        new_y = func(im_yiq[:, :, 0], *args)
        im_yiq[:, :, 0] = new_y[0]
        im_eq = yiq2rgb(im_yiq)
        return [im_eq.astype(np.float64)] + new_y[1:]
    return func(im_orig, *args)


def histogram_equalize(im_orig):
    """
    performs histogram equalization of a given grayscale or RGB image
    :param im_orig: is the input grayscale or RGB float64 image with values in [0, 1].
    :return: The function returns a list [im_eq, hist_orig, hist_eq] where
    im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
    hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
    hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) )
    """

    def greyscale_histogram_equalize(im):
        """
        performs histogram equalization of a given grayscal image
        :param im_orig: is the input grayscale float64 image with values in [0, 1].
        :return: The function returns a list [im_eq, hist_orig, hist_eq] where
        im_eq - is the equalized image. grayscale float64 image with values in [0, 1].
        hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
        hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) )
        """
        hist, _ = np.histogram(im, MAX_INTENSITY + 1)
        cumulative_hist = np.cumsum(hist)
        c_m = cumulative_hist[np.argmax(cumulative_hist > 0)]
        cumulative_hist_normalized = np.rint(
            ((cumulative_hist - c_m) / (cumulative_hist[MAX_INTENSITY] - c_m)) * MAX_INTENSITY)
        im_255 = expand_to_255(im)
        lookup_table = lambda i: cumulative_hist_normalized[i]
        im_eq = normalize_0_to_1(lookup_table(im_255))
        return [im_eq.clip(min=MIN_IMG_VAL, max=MAX_IMG_VAL).astype(np.float64), hist,
                np.histogram(im_eq, MAX_INTENSITY + 1)[HISTOGRAM]]

    return perform_func_grey_or_rgb(im_orig, greyscale_histogram_equalize)


def quantize(im_orig, n_quant, n_iter):
    """
    performs optimal quantization of a given grayscale or RGB image
    :param im_orig: is the input grayscale or RGB image to be quantized (float64 image with values in [0, 1])
    :param n_quant: is the number of intensities your output im_quant image should have
    :param n_iter: is the maximum number of iterations of the optimization procedure (may converge earlier.)
    :return: the output is a list [im_quant, error] where
    im_quant - is the quantized output image.
    error - is an array with shape (n_iter,) (or less) of the total intensities error for each iteration of the
    quantization procedure.
    """
    if n_iter < 1:
        print(AT_LEAST_ITER)
        exit()

    def quantization_process(num_iter, num_quant, probability_hist, z):
        """
        perform quantization iterations
        :param num_iter: is the maximum number of iterations of the optimization procedure (may converge earlier.)
        :param num_quant: is the number of intensities your output im_quant image should have
        :param probability_hist: histogram with probability of each intensity in the image
        :param z:  initial segment division of [0..255] to segments, z
        :return: final Z segmentation, quants values, and error array
        """
        error = []
        q = None
        for iteration in range(num_iter):
            q = []
            # quants calculation
            for i in range(num_quant):
                q.append(np.sum(probability_hist[z[i]:z[i + 1]] * intensities[z[i]:z[i + 1]]) / np.sum(
                    probability_hist[z[i]:z[i + 1]]))
            # z calculation
            temp_z = np.empty(num_quant + 1)
            temp_z[0] = MIN_INTENSITY
            for zi in range(1, num_quant):
                temp_z[zi] = np.ceil((q[zi - 1] + q[zi]) / 2)
            temp_z[num_quant] = MAX_INTENSITY
            # check convergence
            if np.array_equal(z, temp_z):
                return z, q, error
            z = temp_z.astype(np.uint64)
            # error calculation
            error_i = 0
            for i in range(num_quant):
                error_i += np.sum(
                    np.square(np.rint(q[i]) - intensities[z[i]: z[i + 1]]) * probability_hist[z[i]:z[i + 1]])
            error.append(error_i)

        return z, np.rint(q), error

    def quantize_grey_scale(image_orig, num_quant, num_iter):
        """
        performs optimal quantization of a given grayscale image
        :param image_orig: is the input grayscale image to be quantized (float64 image with values in [0, 1])
        :param num_quant: is the number of intensities your output im_quant image should have
        :param num_iter: is the maximum number of iterations of the optimization procedure (may converge earlier.)
        :return: the output is a list [im_quant, error] where
        im_quant - is the quantized output image.
        error - is an array with shape (n_iter,) (or less) of the total intensities error for each iteration of the
        quantization procedure.
        """
        im_255 = expand_to_255(image_orig)
        hist, _ = np.histogram(im_255, MAX_INTENSITY + 1)
        cumulative_hist = np.cumsum(hist)
        probability_hist = (hist / cumulative_hist[MAX_INTENSITY])
        # initial z
        z = np.empty(n_quant + 1)
        z[0] = MIN_INTENSITY
        for zi in range(1, num_quant):
            z[zi] = np.argmax(cumulative_hist > (cumulative_hist[MAX_INTENSITY] / num_quant) * zi)
        z[n_quant] = MAX_INTENSITY
        z = z.astype(np.uint64)
        z, q, error = quantization_process(num_iter, num_quant, probability_hist, z)
        # change image accordingly
        lookup_table = np.empty(MAX_INTENSITY + 1)
        for i in range(num_quant):
            lookup_table[z[i]:z[i + 1]] = q[i]
        im_quant = lookup_table[im_255]
        return [normalize_0_to_1(im_quant).astype(np.float64), error]

    return perform_func_grey_or_rgb(im_orig, quantize_grey_scale, n_quant, n_iter)
