#This file mostly just executes questions/programs that I've been asked to fill out. 
import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts
import scipy

opts = get_opts()
#I used the same image as given in the example
#cv_frontal = cv2.imread('../data/cv_cover.jpg')
#cv_obscured = cv2.imread('../data/cv_desk.png')

#matches, loc1, loc2 = matchPics(cv_frontal, cv_obscured, opts)
#plotMatches(cv_frontal, cv_obscured, matches, loc1, loc2)

image = cv2.imread('../data/cv_cover.jpg')
image_90 = scipy.ndimage.rotate(image, 90, reshape = False)
image_180 = scipy.ndimage.rotate(image, 180, reshape = False)
image_270 = scipy.ndimage.rotate(image, 270, reshape = False)
matches, loc1, loc2 = matchPics(image, image_90, opts)
plotMatches(image, image_90, matches, loc1, loc2)
matches, loc1, loc2 = matchPics(image, image_180, opts)
plotMatches(image, image_180, matches, loc1, loc2)
matches, loc1, loc2 = matchPics(image, image_270, opts)
plotMatches(image, image_270, matches, loc1, loc2)