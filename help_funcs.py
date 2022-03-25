import numpy as np


def k_max(arr, k):
    """
    Find k max values in numpy array
    :param arr: numpy array
    :param k:
    :return array of indices of k max values
    sorted by value in ascending order
    """
    ind = np.argpartition(arr, -k)[-k:]
    ind = ind[np.argsort(arr[ind])]
    return ind

def img_to_ndarray(pil_img):
    """
    Converts PIL image to numpy array
    :param img: grayscale or RGB image
    :return: ndarray pixel_array, size of image NxM, type of img depth if valid image was provided,
    otherwise returns -1. Let pixel count k = n * m
    if image is grayscale, then pixel_array.shape == (k, 1)
    if image is RGB, then pixel_array.shape == (k, 3)
    """

    img = np.asarray(pil_img)
    img_dim = img.ndim
    if img_dim == 2:
        depth = 1
    elif img_dim == 3:
        depth = 3
    else:
        return -1

    N, M = img.shape[:2]
    pixel_array = img.reshape(N*M, depth)
    return pixel_array, N, M, depth
