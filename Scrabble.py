import dippykit as dip
import numpy as np
import skimage.morphology as morph
import scipy.ndimage as ndImage
from skimage.util.dtype import dtype_limits
image = dip.im_read('images/Board3.png')
print(image.shape)
image = dip.rgb2gray(image[:, :, 0:3])

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

levels = [("Canny", .8),
          ("Dilate", conn),
          ("Gauss", .1)
          ]

for i in range(0, len(levels)):
    trans, param = levels[i]
    if trans == "Canny":
        dtype_max = dtype_limits(image, clip_negative=False)[1]
        low_threshold = 0.2 * dtype_max
        high_threshold = 0.8 * dtype_max
        image = dip.edge_detect(image, 'canny', True, sigma=param, low_threshold=low_threshold, high_threshold=high_threshold)
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


bottomright_x = 0
bottomright_y = 0
topright_x = 0
topright_y = 0
bottomleft_x = 0
bottomleft_y = 0
topleft_x = 0
topleft_y = 0
BRsum = 0
TRsum = 0
BLsum = 0
TLsum = 0
imagey, imagex = displayImage.shape
for y in range(0, imagey):
    for x in range(0, imagex):
        if displayImage[y][x]!=0:
            if x + y > BRsum:
                BRsum = x + y
                bottomright_x = x
                bottomright_y = y
            if x + imagey - y > TRsum:
                TRsum = x + imagey - y
                topright_x = x
                topright_y = y
            if imagex - x + imagey - y > TLsum:
                TLsum = imagex - x + imagey - y
                topleft_x = x
                topleft_y = y
            if imagex - x + y > BLsum:
                BLsum = imagex - x + y
                bottomleft_x = x
                bottomleft_y = y
print("TL x= "+str(topleft_x)+" y= "+str(topleft_y))
print("TR x= "+str(topright_x)+" y= "+str(topright_y))
print("BL x= "+str(bottomleft_x)+" y= "+str(bottomleft_y))
print("BR x= "+str(bottomright_x)+" y= "+str(bottomright_y))
dip.show()