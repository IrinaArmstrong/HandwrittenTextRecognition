import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import math
import cv2
from scipy.signal import argrelmin
from WordSegmentation import createKernel


def lineSegmentation(img, kernelSize=25, sigma=11, theta=7):
    """Scale space technique for lines segmentation proposed by R. Manmatha:
	http://ciir.cs.umass.edu/pubfiles/mm-27.pdf
    Args:
		img: image of the text to be segmented on lines.
        kernelSize: size of filter kernel, must be an odd integer.
		sigma: standard deviation of Gaussian function used for filter kernel.
		theta: approximated width/height ratio of words, filter function is distorted by this factor.
		minArea: ignore word candidates smaller than specified area.
	Returns:
		List of lines (segmented input img)
	"""
    img_tmp = np.transpose(prepareTextImg(img))
    k = createKernel(kernelSize, sigma, theta)
    imgFiltered = cv2.filter2D(img_tmp, -1, k, borderType=cv2.BORDER_REPLICATE)
    img_tmp1 = normalize(imgFiltered)
    # Make summ elements in columns to get function of pixels value for each column
    summ_pix = np.sum(img_tmp1, axis = 0)
    smoothed = smooth(summ_pix, 35)
    mins = np.array(argrelmin(smoothed, order=2))
    found_lines = transpose_lines(crop_text_to_lines(img_tmp, mins[0]))
    return found_lines

def prepareTextImg(img):
    """convert given image to grayscale image (if needed) and normalize it"""
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return normalize(img)

def normalize(img):
    """ Normalize input image:
    img = (img[][]-mean)/ stddev
    using function: cv2.meanStdDev(src[, mean[, stddev[, mask]]]), returns: mean, stddev
    where: mean & stddev - numpy.ndarray[][] """
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s>0 else img
    return img

def smooth(x, window_len=11, window='hanning'):
    """ Image smoothing is achieved by convolving the image with a low-pass filter kernel.
    Such low pass filters as: ['flat', 'hanning', 'hamming', 'bartlett', 'blackman'] can be used
    It is useful for removing noise. It actually removes high frequency content
    (e.g: noise, edges) from the image resulting in edges being blurred when this is filter is applied."""
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='valid')
    return y


def crop_text_to_lines(text, blanks):
    """ Splits the image with text into lines, according to the markup obtained from the created algorithm.
     Very first"""
    x1 = 0
    y = 0
    lines = []
    for i, blank in enumerate(blanks):
        x2 = blank
        line = text[:, x1:x2]
        lines.append(line)
        x1 = blank
    print("Lines found: {0}".format(len(lines)))
    return lines


def transpose_lines(lines):
    res = []
    for l in lines:
        line = np.transpose(l)
        res.append(line)
    return res