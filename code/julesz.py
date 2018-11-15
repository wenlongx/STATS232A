from numpy.ctypeslib import ndpointer
import ctypes
import numpy as np


def load_lib():
    return ctypes.cdll.LoadLibrary('./libjulesz.so')


def get_histogram(lib, image, filters, width, height, num_bins=15, max_intensity=255):

    num_filters = len(filters)
    [im_width, im_height] = image.shape

    response = np.zeros((1, num_filters * num_bins))

    c_array = lambda a: (a.__array_interface__['data'][0] + np.arange(a.shape[0]) * a.strides[0]).astype(np.uintp)
    c_int32 = lambda x: x.astype(np.int32)
    ndpointerpointer = lambda: ndpointer(dtype=np.uintp, ndim=1, flags='C')

    getHistogram = lib.getHistogram
    getHistogram.restype = None
    # (const double **filter, const int *width, const int *height, const int num_filters, const int *image, const int im_width, const int im_height, double *response)
    getHistogram.argtypes = [ndpointerpointer(), ndpointer(ctypes.c_int), ndpointer(ctypes.c_int), ctypes.c_int, ndpointer(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_double), ctypes.c_int]

    getHistogram(c_array(filters), c_int32(width), c_int32(height), num_filters, c_int32(image), im_width, im_height, num_bins, response, max_intensity)

    return response


def julesz(lib, responsematrix, filtermatrix, width, height, im_width, im_height, max_intensity=255):

    num_bins = responsematrix.shape[1]
    max_size = filtermatrix.shape[1]
    num_filters = filtermatrix.shape[0]

    filters = np.zeros((num_filters, max_size))
    orig_response = np.zeros((num_filters, num_bins))

    for i in range(0, num_filters):
        for j in range(num_bins):
            orig_response[i, j] = (responsematrix[i, j] * im_width * im_height + 0.499999).astype(np.int)
        filters[i] = filtermatrix[i]

    synthesized = (max_intensity * np.random.rand(im_width * im_height)).astype(np.int32)
    syn_response = np.zeros(num_filters * num_bins)

    c_array = lambda a: (a.__array_interface__['data'][0] + np.arange(a.shape[0]) * a.strides[0]).astype(np.uintp)
    c_int32 = lambda x: x.astype(np.int32)
    ndpointerpointer = lambda: ndpointer(dtype=np.uintp, ndim=1, flags='C')

    Julesz = lib.Julesz
    Julesz.restype = None
    # (const double **filters, const int *width, const int *height, const int num_filters, const int num_bins, const double **orig_response, int *synthesized, double *final_response)
    Julesz.argtypes = [ndpointerpointer(), ndpointer(ctypes.c_int), ndpointer(ctypes.c_int), ctypes.c_int, ctypes.c_int, ndpointerpointer(), ndpointer(ctypes.c_int), ndpointer(ctypes.c_double), ctypes.c_int]

    Julesz(c_array(filters), c_int32(width), c_int32(height), num_filters, num_bins, c_array(orig_response), synthesized, syn_response, im_width, im_height, max_intensity)

    return synthesized, syn_response

def main():
    from skimage import transform, io
    from filters import get_filters

    max_intensity = 7

    lib = load_lib()

    [F, filters, width, height] = get_filters()

    im_w = im_h = 256

    image = io.imread('images/fur_obs.jpg', as_grey=True)
    image = transform.resize(image, (im_w, im_h), mode='symmetric', preserve_range=True)
    image = (image * max_intensity).astype(np.int32)

    n = 7

    h2 = get_histogram(lib, image, filters[n:n+1], width[n:n+1], height[n:n+1])
    syn2, synresp = julesz(lib, h2.reshape(1, 15), filters[n:n+1], width[n:n+1], height[n:n+1], im_w, im_h, max_intensity=max_intensity)

    print(h2)
    print(synresp)


if __name__ == '__main__':
    main()
