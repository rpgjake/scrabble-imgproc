


img1 = dip.imread('images/Board.png')
img1 = img1[:, :, 0:3]
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1 = cv2.normalize(img1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
img2 = dip.imread('images/Test2.png')
img2 = img2[:, :, 0:3]
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2 = cv2.normalize(img2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

img1 = ndImage.gaussian_filter(img1, 3)
img2 = ndImage.gaussian_filter(img2, 3)

dip.figure("Board")
dip.imshow(img1, 'gray')
dip.figure("Test")
dip.imshow(img2, 'gray')
#dip.show()


# # Initiate SIFT detector
# sift = cv2.xfeatures2d.SIFT_create()
#
# # find the keypoints and descriptors with SIFT
# print(img1.shape)
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
#
# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
#
# # Match descriptors.
# matches = bf.match(des1, des2)
#
# # Sort them in the order of their distance.
# matches = sorted(matches, key=lambda x: x.distance)
#
# # Draw first 10 matches.
# max = 10
# img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:max], None, flags=2)
#
# src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:max]]).reshape(-1, 1, 2)
# dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:max]]).reshape(-1, 1, 2)
#
# print("Found " + str(len(dst_pts)) + " matches ")
# M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
#
# matchesMask = mask.ravel().tolist()
#
# h, w = img1.shape
# # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
# dst = cv2.warpPerspective(image2, M, (w, h))
#
# # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
#
# # M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
# # dst = cv2.perspectiveTransform(img2, M)
#
# dip.figure("Output")
# dip.imshow(dst, 'gray')
# dip.figure("Image 3")
# dip.imshow(img3)


#cv2.warpPerspective(im1, im1Reg, h, im2.size())




# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)
print(matches)

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMask[i] = [1, 0]

goodMatches = []
for i in range(len(matches)):
    if matchesMask[i][0] == 1:
        goodMatches.append(matches[i][0])


draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=0)



img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

plt.imshow(img3,)

goodMatches = sorted(goodMatches, key=lambda x: x.distance)

# Draw first 10 matches.
max = 500
img3 = cv2.drawMatches(img1, kp1, img2, kp2, goodMatches[:max], None, flags=2)
plt.imshow(img3,)

src_pts = np.float32([kp1[m.queryIdx].pt for m in goodMatches[:max]]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodMatches[:max]]).reshape(-1, 1, 2)

print("Found " + str(len(dst_pts)) + " matches ")
M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

matchesMask = mask.ravel().tolist()

h, w = img1.shape
# pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.warpPerspective(image2, M, (w, h))







dip.figure("Original")
dip.imshow(img1)




dest = np.array([(1285, 523), (1306, 2347), (3462, 2567), (3474, 260)])
scale = 15 * 15 * 20
src = np.array([[0, 0], [0, scale], [scale, scale], [scale, 0]])
tform3 = tf.ProjectiveTransform()
tform3.estimate(src, dest)
warped = tf.warp(image2, tform3, output_shape=[scale, scale])
dip.figure("warped")
dip.imshow(warped, 'gray')
dip.show()



exit()


