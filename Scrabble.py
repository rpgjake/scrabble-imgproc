import dippykit as dip
import numpy as np
import scipy.ndimage as ndImage
import skimage.transform as tf
import cv2
import pytesseract
import os

import Score
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files (x86)\Tesseract-OCR\tessdata'

def imageToBoard(path):
    # Read the image and extract the grayscale and hue channels
    image = dip.im_read(path)
    image = np.rot90(image[:, :, 0:3], 3)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    image = dip.rgb2gray(image)

    # Scale down the image so that the smallest dimension is 1000 pixels
    x, y = image.shape
    if min(x, y) > 1000:
        scale = 1000.0 / min(x, y)
        newSize = (int(scale * y), int(scale * x))
        image = dip.resize(image, newSize)
        hue = dip.resize(hue, newSize)

    # Detect the straight lines and draw them (dilated) on a blank canvas
    image2 = np.zeros(image.shape, dtype=np.uint8)
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(image)[0]
    for line in lines:
        for (x1, y1, x2, y2) in line:
            # length = abs(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
            cv2.line(image2, (x1, y1), (x2, y2), 255, 3)

    # dip.figure("Image")
    # dip.imshow(image, 'gray')
    # dip.figure("Test Lines")
    # dip.imshow(image2, 'gray')
    # dip.show()

    # Find the largest blob in the image to find the board
    Labeled, numobj = ndImage.label(image2)
    lastSum = 0
    displayImage = None
    for item in range(1, numobj + 1):
        newImage = (Labeled == item)
        newSum = newImage.sum()
        if newSum > lastSum:
            displayImage = newImage
            lastSum = newSum

    # Find the four corners of the image.
    # The corners are defined as the maxima of the four functions:
    # (x + y), (X - x + y), (x + Y - y), and (X - x + Y - y)
    # This assumes the image is taken roughly square with the image boundaries, but it can vary somewhat
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
    # Estimate the transformation that would put the board corners on the image corners
    dest = np.array([cornerTL, cornerBL, cornerBR, cornerTR])
    scale = 1000 # 15 * 15 * 20
    src = np.array([[0, 0], [0, scale], [scale, scale], [scale, 0]])
    tform3 = tf.ProjectiveTransform()
    tform3.estimate(src, dest)
    warped = tf.warp(image, tform3, output_shape=[scale, scale])
    hue = tf.warp(hue, tform3, output_shape=[scale, scale])
    warped = warped[7 : -7, 7 : -7]
    hue = hue[7 : -7, 7 : -7]

    # dip.figure("warped")
    # dip.imshow(warped, 'gray')
    # dip.figure("hue")
    # dip.imshow(hue, 'gray')
    # dip.show()

    # Do line detection again to try to fine the best grid lines, particularly on the board borders
    warped = dip.float_to_im(warped)
    gridLines = np.zeros(warped.shape, dtype=np.uint8)
    lsd = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_ADV)
    lines = lsd.detect(warped)[0]
    for line in lines:
        for (x1, y1, x2, y2) in line:
            cv2.line(gridLines, (x1, y1), (x2, y2), 255, 3)

    # dip.figure("warped")
    # dip.imshow(warped, 'gray')
    # dip.figure("gridLines")
    # dip.imshow(gridLines, 'gray')
    # dip.show()

    # Determine the actual rows/cols that start with pixels
    dimX, dimY = warped.shape
    topRow = 0
    botRow = dimY - 1
    leftCol = 0
    rightCol = dimX - 1
    while np.amax(gridLines[:, leftCol]) == 0:
        leftCol += 1
    while np.amax(gridLines[:, rightCol]) == 0:
        rightCol -= 1
    while np.amax(gridLines[topRow]) == 0:
        topRow += 1
    while np.amax(gridLines[botRow]) == 0:
        botRow -= 1

    lineTop = (topRow, topRow)
    lineBot = (botRow, botRow)
    lineLeft = (leftCol, leftCol)
    lineRight = (rightCol, rightCol)
    bestScoreTop = 0
    bestScoreBot = 0
    bestScoreLeft = 0
    bestScoreRight = 0

    # Within a small range from the border, determine the lines that best describe the image borders
    # They are scored by which one has the most overlap with the canvas with the image lines
    canvas = np.zeros(warped.shape, dtype=np.uint8)
    thickness = 13
    for i in range(6, 50):
        for j in range(6, 50):
            # Top Row
            x1 = 0
            y1 = topRow + i
            x2 = dimX - 1
            y2 = topRow + j
            canvas.fill(0)
            cv2.line(canvas, (x1, y1), (x2, y2), 255, thickness)
            score = np.count_nonzero(np.logical_and(canvas > 0, gridLines))
            if score > bestScoreTop:
                lineTop = (y1, y2)
                bestScoreTop = score

            # Bottom Row
            x1 = 0
            y1 = botRow - i
            x2 = dimX - 1
            y2 = botRow - j
            canvas.fill(0)
            cv2.line(canvas, (x1, y1), (x2, y2), 255, thickness)
            score = np.count_nonzero(np.logical_and(canvas > 0, gridLines))
            if score > bestScoreBot:
                lineBot = (y1, y2)
                bestScoreBot = score

            # Left Column
            x1 = leftCol + i
            y1 = 0
            x2 = leftCol + j
            y2 = dimY - 1
            canvas.fill(0)
            cv2.line(canvas, (x1, y1), (x2, y2), 255, thickness)
            score = np.count_nonzero(np.logical_and(canvas > 0, gridLines))
            if score > bestScoreLeft:
                lineLeft = (x1, x2)
                bestScoreLeft = score

            # Right Column
            x1 = rightCol - i
            y1 = 0
            x2 = rightCol - j
            y2 = dimY - 1
            canvas.fill(0)
            cv2.line(canvas, (x1, y1), (x2, y2), 255, thickness)
            score = np.count_nonzero(np.logical_and(canvas > 0, gridLines))
            if score > bestScoreRight:
                lineRight = (x1, x2)
                bestScoreRight = score

    xDiff0 = (lineBot[0] - lineTop[0]) / 15.0
    xDiff1 = (lineBot[1] - lineTop[1]) / 15.0
    yDiff0 = (lineRight[0] - lineLeft[0]) / 15.0
    yDiff1 = (lineRight[1] - lineLeft[1]) / 15.0

    # cv2.line(warped, (0, lineTop[0]), (dimY - 1, lineTop[1]), 0, 3)
    # cv2.line(warped, (0, lineBot[0]), (dimY - 1, lineBot[1]), 0, 3)
    # cv2.line(warped, (lineLeft[0], 0), (lineLeft[1], dimX - 1), 0, 3)
    # cv2.line(warped, (lineRight[0], 0), (lineRight[1], dimX - 1), 0, 3)
    #
    # for i in range(1, 15):
    #     y1 = int(lineTop[0] + i * xDiff0)
    #     y2 = int(lineTop[1] + i * xDiff1)
    #     cv2.line(warped, (0, y1), (dimY - 1, y2), 0, 3)
    #
    #     x1 = int(lineLeft[0] + i * yDiff0)
    #     x2 = int(lineLeft[1] + i * yDiff1)
    #     cv2.line(warped, (x1, 0), (x2, dimX - 1), 0, 3)
    #
    # dip.figure("Lined image")
    # dip.imshow(warped, 'gray')
    # dip.show()

    # Now go through each of the 225 (15 * 15) cells
    grid = []
    for i in range(0, 15):
        grid.append([])
        for j in range(0, 15):
            # Calculate the four corners of the current grid square
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
            # Warp the image so that the grid square becomes the center of the image with some padding on all sides
            dest = np.array([[pad, pad], [pad, scale + pad], [scale + pad, scale + pad], [scale + pad, pad]])
            src = np.array([[tl_x, tl_y], [bl_x, bl_y], [br_x, br_y], [tr_x, tr_y]])
            tform = tf.ProjectiveTransform()
            tform.estimate(dest, src)
            total = scale + 2 * pad
            output = tf.warp(warped, tform, output_shape=[total, total])
            outputHue = tf.warp(hue, tform, output_shape=[total, total])
            # Output hue doesn't use any of the extra padding because it wants the values from the middle of the tile
            outputHue = outputHue[2 * pad : -2 * pad, 2 * pad : -2 * pad]

            # Perform a simple image threshold to determine any text on the tile
            outputBinary = np.logical_not(output < 0.55)
            Labeled, numobj = ndImage.label(outputBinary)
            closestBlob = None
            distance = 20
            for item in range(1, numobj + 1):
                blob = (Labeled != item)
                x, y = output.shape
                for a in range(0, x):
                    for b in range(0, y):
                        if blob[a, b] == 0:
                            dist = np.sqrt((a - 50) ** 2 + (b - 50) ** 2)
                            tot = np.sum(blob)
                            # If the current blob is within a set distance from the middle of the image,
                            # and the total count doesn't indicate a false tile or a blank tile
                            if dist < distance and 9000 < tot and tot < 9950:
                                distance = dist
                                closestBlob = blob
            text = "?"
            # If a blob was detected
            if closestBlob is not None:
                closestBlob = closestBlob.astype(np.uint8) * 255
                # Perform OCR
                text = pytesseract.image_to_string(closestBlob, config='--oem 0 -c tessedit_char_whitelist=ABCDEFGHIJLKMNOPQRSTUVWXYZ|01l --psm 10')
                # Just a precaution to fix any ambiguity with 0s and Os
                text = text.replace("0", "O")
                # Correct the I tile, as a straight line doesn't easily count with vanilla Tesseract
                if text in ['', '|', 'l', '1']:
                    text = "I"
            # If no letter detected and the median hue & grayscale values indicate a blank tile
            med = np.median(outputHue)
            if text == "?" and (med > 0.6 or med < 0.01) and np.median(output) < 0.3:
                text = '_'
            grid[-1].append(text)
    return grid

if __name__ == "__main__":
    after = imageToBoard('images/NewBoard/Img25.jpg')
    for a in after:
        print([b for b in a])
    before = imageToBoard('images/NewBoard/Img24.jpg')
    for a in before:
        print([b for b in a])
    score = Score.score(before, after)
    print("Score is:  " + str(score))