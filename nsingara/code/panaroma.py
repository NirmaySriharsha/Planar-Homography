import numpy as np
import cv2

# Import necessary functions
import matplotlib.pyplot as plt
from planarH import compositeH
from planarH import computeH_ransac
from matchPics import matchPics
from displayMatch import displayMatched
import skimage.io 
import skimage.color
import scipy
from opts import get_opts

opts = get_opts()
#left = cv2.imread("data\my_left.jpg")
#right = cv2.imread("data\my_right.jpg")
left = cv2.imread("data\pano_left_mine.jpg")
right = cv2.imread("data\pano_right_mine.jpg")
plt.imshow(left)
plt.show()
plt.imshow(right)
plt.show()
left = cv2.resize(left, (1457, 1080))
right = cv2.resize(right, (1457, 1080))
#I figured out the padding basically by trial and error
right = cv2.copyMakeBorder(right, right.shape[1] - left.shape[1], 0, int(0.6*left.shape[1]), 0, cv2.BORDER_CONSTANT, value = 0)
matches, loc1, loc2 = matchPics(left, right, opts)
#displayMatched(opts, left, right)
loc1 = loc1[matches[:, 0], 0:2]
loc2 = loc2[matches[:, 1], 0:2]
bestH2to1, inliers = computeH_ransac(loc1, loc2, opts)
composite_img = compositeH(bestH2to1, left, right)
plt.imshow(composite_img)
plt.show()





# Q4
