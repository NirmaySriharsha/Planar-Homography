import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

# Import necessary functions
import matplotlib.pyplot as plt
from planarH import compositeH
from planarH import computeH_ransac
from matchPics import matchPics
# Q2.2.4

def warpImage(opts):
    cover = cv2.imread('../data/cv_cover.jpg')
    harry_potter = cv2.imread('../data/hp_cover.jpg')
    desk = cv2.imread('../data/cv_desk.png')
    matches, loc1, loc2 = matchPics(cover, desk, opts)
    
    #In answer to step 4, we need the harry potter cover to appear as the same size and perspective as the book cover. 
    #This can be accomplished with an easy resize
    harry_potter_correct = cv2.resize(harry_potter, (cover.shape[1], cover.shape[0]))
    #Isolating the matches from the locations
    loc1 = loc1[matches[:, 0], 0:2]
    loc2 = loc2[matches[:, 1], 0:2]
    bestH2to1, inliers = computeH_ransac(loc1, loc2, opts)
    composite_img = compositeH(bestH2to1, harry_potter_correct, desk)
    plt.imshow(composite_img)
    plt.show()
    



if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)


