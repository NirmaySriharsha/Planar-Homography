import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
from helper import plotMatches
import scipy
import matplotlib.pyplot as plt

#Q2.1.6
def rotTest(opts):
    opts = get_opts()
    num_of_matches = np.zeros(36)
    #Read the image and convert to grayscale, if necessary
    image = cv2.imread('../data/cv_cover.jpg')
    for i in range(36):
        theta = i*10
        #Rotate Image
        image_rotated = scipy.ndimage.rotate(image, theta, reshape = False)
        #Compute features, descriptors and Match features
        matches, loc_1, loc_2 = matchPics(image, image_rotated, opts)
        num_of_matches[i] = len(matches)
    #Update histogram
    x_axis = np.arange(0, 360, 10)
    plt.bar(x_axis, num_of_matches) 
    plt.xlabel("Rotation Angle in Degrees")
    plt.ylabel("Number of Matches")
    plt.title("Matches per theta of rotation")
    #Display histogram
    plt.show()
if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
