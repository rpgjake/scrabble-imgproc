import dippykit as dip
import numpy as np
import skimage.morphology as morph
import scipy.ndimage as ndImage
import skimage.transform as tf
from skimage.util.dtype import dtype_limits

import cv2
from matplotlib import pyplot as plt


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged



image = dip.im_read('images/Test2.png')
image = dip.rgb2gray(image[:, :, 0:3])
image2 = image

#print(I1.shape)
conn = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                 [0, 0, 1, 1, 1, 1, 1, 0, 0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 1, 1, 1, 1, 0],
                 [0, 0, 1, 1, 1, 1, 1, 0, 0],
                 [0, 0, 0, 1, 1, 1, 0, 0, 0]])
conn = np.array([[1, 1],
                 [1, 1]])
conn = np.array([[0, 1, 0],
                 [1, 1, 1],
                 [0, 1, 0]])

levels = [#("Gauss", 1),
          ("Canny", .4),
          ("Dilate", conn),
          ("Gauss", 2)]

for i in range(0, len(levels)):
    trans, param = levels[i]
    if trans == "Canny":
        image = auto_canny(image, param)
        # dtype_max = dtype_limits(image, clip_negative=False)[1]
        # # image = dip.edge_detect(image, 'canny', True, sigma=param)
        # low_threshold = 0.1 * dtype_max
        # high_threshold = 0.3 * dtype_max
        # image = dip.edge_detect(image, 'canny', True, sigma=param, low_threshold=low_threshold, high_threshold=high_threshold)
    elif trans == "Dilate":
        image = morph.dilation(image, selem=param)
    elif trans == "Gauss":
        image = ndImage.gaussian_filter(image, param)
    text = trans + ":  " + str(i)
    if trans != "Dilate":
        text +=  "      " + str(param)
    dip.figure(text)
    dip.imshow(image, 'gray')
dip.show()

Labeled, numobj = ndImage.label(image)
lastSum = 0
displayImage = None
for item in range(1, numobj + 1):
    newImage = (Labeled == item)
    newSum = np.sum(newImage)
    if newSum > lastSum:
        displayImage = newImage
        lastSum = newSum


dip.figure("Largest Blob")
dip.imshow(displayImage, 'gray')

cornerBR = (0, 0)
sumBR = 0
cornerTR = (0, 0)
sumTR = 0
cornerBL = (0, 0)
sumBL = 0
cornerTL = (0, 0)
sumTL = 0
imagey, imagex = displayImage.shape
for x in range(0, imagex):
    for y in range(0, imagey):
        if displayImage[y][x] != 0:
            temp = x + y
            if temp > sumBR:
                sumBR = temp
                cornerBR = (x, y)
            temp = x + imagey - y
            if temp > sumTR:
                sumTR = temp
                cornerTR = (x, y)
            temp = imagex - x + imagey - y
            if temp > sumTL:
                sumTL = temp
                cornerTL = (x, y)
            temp = imagex - x + y
            if temp > sumBL:
                sumBL = temp
                cornerBL = (x, y)
print("TL:  " + str(cornerTL))
print("TR:  " + str(cornerTR))
print("BL:  " + str(cornerBL))
print("BR:  " + str(cornerBR))
dest = np.array([cornerTL, cornerBL, cornerBR, cornerTR])
scale = 15 * 15 * 20
src = np.array([[0, 0], [0, scale], [scale, scale], [scale, 0]])
tform3 = tf.ProjectiveTransform()
tform3.estimate(src, dest)
warped = tf.warp(image2, tform3, output_shape=[scale, scale])
dip.figure("warped")
dip.imshow(warped, 'gray')
dip.show()