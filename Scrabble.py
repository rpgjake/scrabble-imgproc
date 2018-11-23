import dippykit as dip
import numpy as np
import scipy.ndimage as ndImage
import skimage.transform as tf
import cv2
import pytesseract
import os
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files (x86)\Tesseract-OCR\tessdata'

image = dip.im_read('images/Test3.png')
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

# cv2.line(warped, (0, lineTop[0]), (dimY - 1, lineTop[1]), 0, 3)
# cv2.line(warped, (0, lineBot[0]), (dimY - 1, lineBot[1]), 0, 3)
# cv2.line(warped, (lineLeft[0], 0), (lineLeft[1], dimX - 1), 0, 3)
# cv2.line(warped, (lineRight[0], 0), (lineRight[1], dimX - 1), 0, 3)

xDiff0 = (lineBot[0] - lineTop[0]) / 15.0
xDiff1 = (lineBot[1] - lineTop[1]) / 15.0
yDiff0 = (lineRight[0] - lineLeft[0]) / 15.0
yDiff1 = (lineRight[1] - lineLeft[1]) / 15.0
# for i in range(1, 15):
#     y1 = int(lineTop[0] + i * xDiff0)
#     y2 = int(lineTop[1] + i * xDiff1)
#     cv2.line(warped, (0, y1), (dimY - 1, y2), 0, 3)
#
#     x1 = int(lineLeft[0] + i * yDiff0)
#     x2 = int(lineLeft[1] + i * yDiff1)
#     cv2.line(warped, (x1, 0), (x2, dimX - 1), 0, 3)

dip.figure("Lined image")
dip.imshow(warped, 'gray')

for i in range(0, 15):
    for j in range(0, 15):
        amt_i = [i / 15.0, (i + 1) / 15.0]
        amt_i_inv = [1.0 - i / 15.0, 1.0 - (i + 1) / 15.0]
        tl_y = int((lineTop[0] + (i + 0) * xDiff0) * amt_i_inv[0] + (lineTop[1] + (i + 0) * xDiff1) * amt_i[0])
        tr_y = int((lineTop[0] + (i + 0) * xDiff0) * amt_i_inv[1] + (lineTop[1] + (i + 0) * xDiff1) * amt_i[1])
        bl_y = int((lineTop[0] + (i + 1) * xDiff0) * amt_i_inv[0] + (lineTop[1] + (i + 1) * xDiff1) * amt_i[0])
        br_y = int((lineTop[0] + (i + 1) * xDiff0) * amt_i_inv[1] + (lineTop[1] + (i + 1) * xDiff1) * amt_i[1])

        amt_j = [j / 15.0, (j + 1) / 15.0]
        amt_j_inv = [1.0 - j / 15.0, 1.0 - (j + 1) / 15.0]
        tl_x = int((lineLeft[0] + (j + 0) * yDiff0) * amt_j_inv[0] + (lineLeft[1] + (j + 0) * yDiff1) * amt_j[0])
        bl_x = int((lineLeft[0] + (j + 0) * yDiff0) * amt_j_inv[1] + (lineLeft[1] + (j + 0) * yDiff1) * amt_j[1])
        tr_x = int((lineLeft[0] + (j + 1) * yDiff0) * amt_j_inv[0] + (lineLeft[1] + (j + 1) * yDiff1) * amt_j[0])
        br_x = int((lineLeft[0] + (j + 1) * yDiff0) * amt_j_inv[1] + (lineLeft[1] + (j + 1) * yDiff1) * amt_j[1])

        scale = 80
        pad = 10
        total = scale + 2 * pad
        dest = np.array([[pad, pad], [pad, scale + pad], [scale + pad, scale + pad], [scale + pad, pad]])
        src = np.array([[tl_x, tl_y], [bl_x, bl_y], [br_x, br_y], [tr_x, tr_y]])
        tform = tf.ProjectiveTransform()
        tform.estimate(dest, src)
        output = tf.warp(warped, tform, output_shape=[total, total])
        # dip.figure("Square")
        # dip.imshow(warped[tl_y : bl_y, tl_x : tr_x], 'gray')

        output = (output < 0.4)
        Labeled, numobj = ndImage.label(output)
        lastSum = 0
        closestBlob = None
        distance = 99
        for item in range(1, numobj + 1):
            blob = (Labeled == item)
            x, y = output.shape
            for a in range(0, x):
                for b in range(0, y):
                    if blob[a, b] != 0:
                        dist = np.sqrt((a - 50) ** 2 + (b - 50) ** 2)
                        if dist < distance:
                            distance = dist
                            closestBlob = np.logical_not(blob)

        text = pytesseract.image_to_string(closestBlob, config='--oem 0 -c tessedit_char_whitelist=ABCDEFGHIJLKMNOPQRSTUVWXYZ --psm 10')
        # text = pytesseract.image_to_string(closestBlob) #, config='--oem 0 --psm 10')

        dip.figure("Coordinate:  (" + str(j) + ", " + str(i) + ") is " + text)
        dip.imshow(closestBlob, 'gray')
        dip.show()

