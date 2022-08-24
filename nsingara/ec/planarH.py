from textwrap import wrap
import numpy as np
import cv2


def computeH(x1, x2):
    #Q2.2.1
    #Compute the homography between two sets of points
    n = x1.shape[0] #n is the number of point pairs
    A = np.zeros((2*n, 9)) #2*d because for each point pair we get A_i with two rows
    #9 because as we saw in section 1 h has 9 elts

    for i in range(n):
        #A_2i and A_{2i+1} should be the matrix A_i from section 1 and I'm just plugging in what I got from there
        A[2*i] = [-1*x1[i, 0], -1*x1[i, 1], -1, 0, 0, 0, x1[i, 0]*x2[i, 0], x2[i, 0]*x1[i, 1], x2[i, 0]]
        A[2*i + 1] = [0, 0, 0, -1*x1[i, 0], -1*x1[i, 1], -1, x2[i, 0]*x1[i, 0], x2[i, 1]*x1[i, 0], x2[i, 1]]

    #as suggested in the handout now we just do svd
    U, Sigma , V_transpose = np.linalg.svd(A)
    eigenvalues= Sigma[-1]
    eigenvectors = V_transpose[-1:]/V_transpose[-1, -1] #The solution is the last column of V, i.e, the last row of V_transpose 
    #We divided by the last elememnt of the vector for normalization purposes. 
    H2to1 = eigenvectors.reshape(3, 3)
    return H2to1


def computeH_norm(x1, x2):
    #Q2.2.2
    #Compute the centroid of the points
    cx_1 = np.mean(x1[:, 0])
    cy_1 = np.mean(x1[:, 1])
    cx_2 = np.mean(x2[:, 0])
    cy_2 = np.mean(x2[:, 1])

    #Shift the origin of the points to the centroid
    #Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    #I don't see the point of actually finding the new points; it suffices for me to find the transformations instead, because that's what I need to return, so I'm instead going to directly compute the T_i
    magnitudes_1 = np.zeros(x1.shape[0])
    magnitudes_2 = np.zeros(x2.shape[0])
    for i in range(x1.shape[0]):
        magnitudes_1[i] = np.sqrt((x1[i, 0] - cx_1)**2 + (x1[i, 1] - cy_1)**2)
    for i in range(x2.shape[0]):
        magnitudes_2[i] = np.sqrt((x2[i, 0] - cx_2)**2 + (x2[i, 1] - cy_2)**2)
    
    max_1 = np.max(magnitudes_1)
    max_2 = np.max(magnitudes_2)
    #Scale everything down by max and then multiply by sqrt(2) and we good
    scale_1 = np.sqrt(2)/max_1
    scale_2 = np.sqrt(2)/max_2
    translation_1 = np.array([[1, 0, -cx_1], [0, 1, -cy_1], [0, 0, 1]])
    translation_2 = np.array([[1, 0, -cx_2], [0, 1, -cy_2], [0, 0, 1]])
    scale_1_matrix = np.array([[scale_1, 0, 0], [0, scale_1, 0], [0, 0, 1]])
    scale_2_matrix = np.array([[scale_2, 0, 0], [0, scale_2, 0], [0, 0, 1]])
    #Similarity transform 1
    T_1 = scale_1_matrix@translation_1
    x1new = np.hstack((x1, np.ones((x1.shape[0], 1))))
    x1new = T_1@x1new.T
    #Similarity transform 2
    T_2 = scale_2_matrix@translation_2
    x2new = np.hstack((x2, np.ones((x2.shape[0], 1))))
    x2new = T_2@x2new.T
    #Compute homography
    H2to1_oftransf = computeH(x1new, x2new)
    #Denormalization
    #Using the formula given
    H2to1 = np.linalg.inv(T_2)@(H2to1_oftransf@T_1)
    return H2to1




def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    randompixels_1 = np.zeros((2, 4))
    randompixels_2 = np.zeros((2, 4))
    bestH2to1 = np.zeros((3, 3))
    x1 = locs1
    x2 = locs2
    x1_hom = np.hstack((x1, np.ones((x1.shape[0], 1))))
    x2_hom = np.hstack((x2, np.ones((x2.shape[0], 1))))
    inliers = -1
    for i in range(max_iters):
        num_inliers = 0
        random_index = np.random.choice(locs1.shape[0], 4)
        randompixels_1 = locs1[random_index, :]
        randompixels_2 = locs2[random_index, :]
        H_norm = computeH_norm(randompixels_1, randompixels_2)

        for j in range(x2_hom.shape[0]):
            pred_x2 = H_norm@x1_hom[j].T
            pred_x2[0] = pred_x2[0]/pred_x2[2]
            pred_x2[1] = pred_x2[1]/pred_x2[2]
            err_1 = (x2_hom[j][0] - pred_x2[0])
            err_2 = (x2_hom[j][1] - pred_x2[1])
            err = [err_1, err_2]
            error = np.linalg.norm(err)
            if error <= inlier_tol:
                num_inliers+=1
        
        if num_inliers > inliers:
            bestH2to1 = H_norm
            inliers = num_inliers    
    return bestH2to1, inliers



def compositeH(H2to1, template, img):
    
    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.
    

    #Create mask of same size as template
    mask_ones = cv2.transpose(np.ones(template.shape))
    #Warp mask by appropriate homography
    warped_mask = cv2.transpose(cv2.warpPerspective(mask_ones, H2to1, (img.shape[0], img.shape[1])))
    #Warp template by appropriate homography
    template = cv2.transpose(template)
    indices = np.nonzero(warped_mask)
    warped_template = cv2.transpose(cv2.warpPerspective(template, H2to1, (img.shape[0], img.shape[1])))
    #Use mask to combine the warped template and the image
    img[indices] = warped_template[indices]
    composite_img = img.astype('uint8')
    composite_img = cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB)
    

    
    return composite_img


