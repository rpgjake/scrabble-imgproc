import dippykit as dip
import numpy as np
import skimage.morphology as morph
import scipy.ndimage as ndImage
import skimage.transform as tf
import cv2

image = dip.im_read('images/Words6.jpg')
image = (image[:, :, 2])

x, y = image.shape
if max(x, y) > 1000:
    scale = 1000.0 / max(x, y)
    newSize = (int(scale * y), int(scale * x))
    image = dip.resize(image, newSize)

image2 = np.zeros(image.shape, dtype=np.uint8)
original = image

# conn = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
#                  [0, 0, 0, 1, 1, 1, 0, 0, 0],
#                  [0, 0, 0, 1, 1, 1, 0, 0, 0],
#                  [1, 1, 1, 1, 1, 1, 1, 1, 1],
#                  [1, 1, 1, 1, 1, 1, 1, 1, 1],
#                  [1, 1, 1, 1, 1, 1, 1, 1, 1],
#                  [0, 0, 0, 1, 1, 1, 0, 0, 0],
#                  [0, 0, 0, 1, 1, 1, 0, 0, 0],
#                  [0, 0, 0, 1, 1, 1, 0, 0, 0]])
# conn = np.array([[1, 1],
#                  [1, 1]])
conn = np.array([[0, 1, 0],
                 [1, 1, 1],
                 [0, 1, 0]])
# conn = np.array([[0, 0, 1, 1, 0, 0],
#                  [0, 0, 1, 1, 0, 0],
#                  [1, 1, 1, 1, 1, 1],
#                  [1, 1, 1, 1, 1, 1],
#                  [0, 0, 1, 1, 0, 0],
#                  [0, 0, 1, 1, 0, 0]])





lsd = cv2.createLineSegmentDetector(0)
lines = lsd.detect(image)[0]
for line in lines:
    for (x1, y1, x2, y2) in line:
        length = abs(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        cv2.line(image2, (x1, y1), (x2, y2), 255, 1)
image2Lines = image2
image2 = morph.dilation(image2, selem=conn)

Labeled, numobj = ndImage.label(image2)
lastSum = 0
displayImage = None
for item in range(1, numobj + 1):
    newImage = (Labeled == item)
    newSum = newImage.sum()
    if newSum > lastSum:
        displayImage = newImage
        lastSum = newSum

# dip.figure("Largest Blob")
# dip.imshow(displayImage, 'gray')

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
scale = 1000 # 15 * 15 * 20
src = np.array([[0, 0], [0, scale], [scale, scale], [scale, 0]])
tform3 = tf.ProjectiveTransform()
tform3.estimate(src, dest)
warped = tf.warp(original, tform3, output_shape=[scale, scale])
warped = warped[30 : -30, 30 : -30]

dip.figure("warped")
dip.imshow(warped, 'gray')
dip.show()


warped = dip.float_to_im(warped)
image3 = np.zeros(warped.shape, dtype=np.uint8)
lsd = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_ADV)
lines = lsd.detect(warped)[0]
for line in lines:
    for (x1, y1, x2, y2) in line:
        length = abs(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        cv2.line(image3, (x1, y1), (x2, y2), 255, 3)
# image3 = morph.dilation(image3, selem=conn)


Labeled, numobj = ndImage.label(image3)
lastSum = 0
displayImage = None
testImage = None
for item in range(1, numobj + 1):
    newImage = (Labeled == item)
    newSum = newImage.sum()
    if newSum > lastSum:
        displayImage = newImage
        lastSum = newSum

dip.figure("Display image")
dip.imshow(displayImage, 'gray')
dip.show()



dip.figure("Image 3")
dip.imshow(image3, 'gray')
dip.show()