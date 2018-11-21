import dippykit as dip
import numpy as np
import scipy.ndimage as ndImage
import skimage.transform as tf
import cv2

image = dip.im_read('images/Words4.jpg')
image = (image[:, :, 2])

x, y = image.shape
if min(x, y) > 1000:
    scale = 1000.0 / min(x, y)
    newSize = (int(scale * y), int(scale * x))
    image = dip.resize(image, newSize)

original = image

# Image2 draws the lines that are detected
image2 = np.zeros(image.shape, dtype=np.uint8)
lsd = cv2.createLineSegmentDetector(0)
lines = lsd.detect(image)[0]
for line in lines:
    for (x1, y1, x2, y2) in line:
        length = abs(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        cv2.line(image2, (x1, y1), (x2, y2), 255, 3)
image2Lines = image2

Labeled, numobj = ndImage.label(image2)
lastSum = 0
displayImage = None
for item in range(1, numobj + 1):
    newImage = (Labeled == item)
    newSum = newImage.sum()
    if newSum > lastSum:
        displayImage = newImage
        lastSum = newSum

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
dest = np.array([cornerTL, cornerBL, cornerBR, cornerTR])
scale = 1000 # 15 * 15 * 20
src = np.array([[0, 0], [0, scale], [scale, scale], [scale, 0]])
tform3 = tf.ProjectiveTransform()
tform3.estimate(src, dest)
warped = tf.warp(original, tform3, output_shape=[scale, scale])
warped = warped[30 : -30, 30 : -30]

dip.figure("warped")
dip.imshow(warped, 'gray')
# dip.show()


warped = dip.float_to_im(warped)
image3 = np.zeros(warped.shape, dtype=np.uint8)
lsd = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_ADV)
lines = lsd.detect(warped)[0]
for line in lines:
    for (x1, y1, x2, y2) in line:
        length = abs(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        cv2.line(image3, (x1, y1), (x2, y2), 255, 3)

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
# dip.show()

dimX, dimY = warped.shape

topRow = 0
botRow = dimY - 1
leftCol = 0
rightCol = dimX - 1
while np.amax(displayImage[:, leftCol]) == 0:
    leftCol += 1
while np.amax(displayImage[:, rightCol]) == 0:
    rightCol -= 1
while np.amax(displayImage[topRow]) == 0:
    topRow += 1
while np.amax(displayImage[botRow]) == 0:
    botRow -= 1

lineTop = (topRow, topRow)
lineBot = (botRow, botRow)
lineLeft = (leftCol, leftCol)
lineRight = (rightCol, rightCol)
bestScoreTop = 0
bestScoreBot = 0
bestScoreLeft = 0
bestScoreRight = 0

canvas = np.zeros(warped.shape, dtype=np.uint8)
for i in range(6, 50):
    for j in range(6, 50):
        # Top Row
        canvas.fill(0)
        x1 = 0
        y1 = topRow + i
        x2 = dimX - 1
        y2 = topRow + j
        cv2.line(canvas, (x1, y1), (x2, y2), 255, 10)
        score = np.count_nonzero(np.logical_and(canvas > 0, displayImage))
        if score > bestScoreTop:
            lineTop = (y1, y2)
            bestScoreTop = score

        # Bottom Row
        canvas.fill(0)
        x1 = 0
        y1 = botRow - i
        x2 = dimX - 1
        y2 = botRow - j
        cv2.line(canvas, (x1, y1), (x2, y2), 255, 10)
        score = np.count_nonzero(np.logical_and(canvas > 0, displayImage))
        if score > bestScoreBot:
            lineBot = (y1, y2)
            bestScoreBot = score

        # Left Column
        canvas.fill(0)
        x1 = leftCol + i
        y1 = 0
        x2 = leftCol + j
        y2 = dimY - 1
        cv2.line(canvas, (x1, y1), (x2, y2), 255, 10)
        score = np.count_nonzero(np.logical_and(canvas > 0, displayImage))
        if score > bestScoreLeft:
            lineLeft = (x1, x2)
            bestScoreLeft = score

        # Right Column
        canvas.fill(0)
        x1 = rightCol - i
        y1 = 0
        x2 = rightCol - j
        y2 = dimY - 1
        cv2.line(canvas, (x1, y1), (x2, y2), 255, 10)
        score = np.count_nonzero(np.logical_and(canvas > 0, displayImage))
        if score > bestScoreRight:
            lineRight = (x1, x2)
            bestScoreRight = score

cv2.line(warped, (0, lineTop[0]), (dimY - 1, lineTop[1]), 0, 3)
cv2.line(warped, (0, lineBot[0]), (dimY - 1, lineBot[1]), 0, 3)
xDiff0 = (lineBot[0] - lineTop[0]) / 15.0
xDiff1 = (lineBot[1] - lineTop[1]) / 15.0
yDiff0 = (lineRight[0] - lineLeft[0]) / 15.0
yDiff1 = (lineRight[1] - lineLeft[1]) / 15.0
for i in range(1, 15):
    y1 = int(lineTop[0] + i * xDiff0)
    y2 = int(lineTop[1] + i * xDiff1)
    cv2.line(warped, (0, y1), (dimY - 1, y2), 0, 3)

    x1 = int(lineLeft[0] + i * yDiff0)
    x2 = int(lineLeft[1] + i * yDiff1)
    cv2.line(warped, (x1, 0), (x2, dimX - 1), 0, 3)

cv2.line(warped, (lineLeft[0], 0), (lineLeft[1], dimX - 1), 0, 3)
cv2.line(warped, (lineRight[0], 0), (lineRight[1], dimX - 1), 0, 3)

dip.figure("Lined image")
dip.imshow(warped, 'gray')
dip.show()