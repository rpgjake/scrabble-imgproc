import dippykit as dip
import numpy as np
import skimage.morphology as morph
import scipy.ndimage as ndImage
I1 = dip.im_read('scrabble/image1.jpeg')
I1 = dip.rgb2gray(I1)

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


Is_1 = dip.edge_detect(I1, 'canny', True, sigma=1)
#Is_P5 = dip.edge_detect(I1, 'canny', True, sigma=0.5)
Dilated = morph.dilation(Is_1, selem=conn)
Dilated = ndImage.gaussian_filter(Dilated, .25)
Dilated = morph.dilation(Dilated, selem=conn)
Dilated = ndImage.gaussian_filter(Dilated, .25)
Dilated = morph.dilation(Dilated, selem=conn)
Gauss = ndImage.gaussian_filter(Dilated, 2)
Labeled, numobj = ndImage.label(Gauss)
# dip.figure('Dilated')
# dip.imshow(Dilated, 'gray')
# dip.figure('Gauss')
# dip.imshow(Gauss, 'gray')
# dip.show()
dip.figure('Label')
dip.imshow(Labeled, 'gray')
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
dip.show()
