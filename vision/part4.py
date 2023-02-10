#!/usr/bin/python3

from re import I
import numpy as np


def my_conv2d_freq(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Apply the Convolution Theorem to perform the convolution operation. 

    Return 
    - the input image represented in the frequency domain, 
    - the filter represented in the frequency domain,
    - the result of the convolution in the frequency domain, and 
    - the result of the convolution in the spatial domain.

    We will plot and analyze these to gain a better understanding of what is going on.

    Args:
        image: array of shape (m, n)
        filter: array of shape (k, j)
    Returns:
        image_freq: array of shape (m, n)
        filter_freq: array of shape (m, n)
        conv_result_freq: array of shape (m, n)
        conv_result: array of shape (m, n)
    HINTS:
    - Pad your filter in the spatial domain. We want to retain all of the high frequencies after the FFT
    - Return only the real component of the convolution result
    - Numpy considers frequency graphs differently than we have shown them in class. Look into the 
      documentation for np.fft.fft2 for how to account for this in the output image.
    - When applying padding, only use the zero-padding method.
    """

    ############################
    ### TODO: YOUR CODE HERE ###
    image_freq = np.fft.fft2(image)
    filter_freq = np.fft.fft2(filter, image.shape)
    conv_result_freq = np.multiply(image_freq,filter_freq)
    conv_result = np.fft.ifft2(conv_result_freq)
    return np.real(image_freq), np.real(filter_freq), np.real(conv_result_freq), np.real(conv_result)
    raise NotImplementedError(
        "`my_conv2d_freq` function in `part4.py` needs to be implemented"
    )

    ### END OF STUDENT CODE ####
    ############################


def my_deconv2d_freq(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    """
    Apply the Convolution Theorem to perform the convolution operation.

    Return 
    - the input image represented in the frequency domain, 
    - the filter represented in the frequency domain,
    - the result of the deconvolution in the frequency domain, and 
    - the result of the deconvolution in the spatial domain.

    We will plot and analyze these to gain a better understanding of what is going on.

    Args:
        image: array of shape (m, n)
        filter: array of shape (k, j)
    Returns:
        image_freq: array of shape (m, n)
        filter_freq: array of shape (m, n)
        deconv_result_freq: array of shape (m, n)
        deconv_result: array of shape (m, n)
    HINTS:
    - Pad your filter in the spatial domain. We want to retain all of the high frequencies after the FFT
    - Return only the real component of the deconvolution result
    - Numpy considers frequency graphs differently than we have shown them in class. Look into the 
      documentation for np.fft.fft2 to see what this means and to account for this in the output image.
    - When applying padding, only use the zero-padding method.
    """

    ############################
    ### TODO: YOUR CODE HERE ###
    image_freq = np.fft.fft2(image)
    filter_freq = np.fft.fft2(filter, image.shape)
    deconv_result_freq = image_freq/filter_freq
    deconv_result = np.fft.ifft2(deconv_result_freq)
    # deconv_result = np.clip(deconv_result, a_min=0, a_max = 1)
    return np.real(image_freq), np.real(filter_freq), np.real(deconv_result_freq), np.real(deconv_result)
    raise NotImplementedError(
        "`my_deconv2d_freq` function in `part4.py` needs to be implemented"
    )

    ### END OF STUDENT CODE ####
    ############################
