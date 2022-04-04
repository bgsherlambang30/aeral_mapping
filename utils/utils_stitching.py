import cv2 as cv
import numpy as np


def DetectAndDescribe(img, method='orb'):
    '''
        Compute key points and feature descriptors using an specific method
    '''
    if method == 'brisk':
        descriptor = cv.BRISK_create()
    elif method == 'orb':
        descriptor = cv.ORB_create(nfeatures=30000)

    ret, mask = cv.threshold(img, 1, 255, cv.THRESH_BINARY)
    # get keypoints and descriptors
    (kps, desc) = descriptor.detectAndCompute(img, mask)

    return(kps, desc)


def createMatcher(method, crossCheck):
    # Create and return a Matcher Object

    if method == 'sift' or method == 'surf':
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=crossCheck)
    return bf


def matchkeypointsBF(descA, descB, method):
    bf = createMatcher(method, crossCheck=True)

    # Match Descriptors
    best_Match = bf.match(descA, descB)

    # Sort the desc in order of distance
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_Match, key=lambda x: x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches


def matchkeypointsKNN(descA, descB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(descA, descB, 2)
    #print("Raw matches (knn):", len(rawMatches))
    matches = []

    # Loop over ther raw matches
    for m, n in rawMatches:
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches


def getHomography(kpsA, kpsB, matches, Threshold):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])

    if len(matches) > 4:

        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])

        (H, status) = cv.findHomography(ptsA, ptsB, cv.RANSAC, Threshold)

        return(H, status)
    else:
        print('went wrong')
        return None


def get2dTrans(kpsA, kpsB, matches, Threshold):
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])

    ptsA = np.float32([kpsA[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
    ptsB = np.float32([kpsB[m.trainIdx] for m in matches]).reshape(-1, 1, 2)

    H = cv.estimateAffinePartial2D(
        ptsA, ptsB, method=cv.RANSAC, ransacReprojThreshold=Threshold)
    return H


def getRotMatrix(H_full, c):
    '''
        Return: Rotation Matrix 2x2
    '''
    n = len(H_full)
    H_rot_list = []
    T_y_list = []
    T_x_list = []

    for i in range(0, n, 1):
        H = H_full[i]
        a = H[0][0]
        b = H[1][0]
        theta = np.arctan2(b, a)
        theta = np.rad2deg(theta)
        if abs(theta) < 3:
            theta = 0
        Mat = cv.getRotationMatrix2D(center=c, angle=theta, scale=1)
        H_rot = Mat[0:2, 0:2]
        T_y = Mat[0][-1]
        T_x = Mat[1][-1]
        H_rot_list.append(H_rot)
        T_y_list.append(T_y)
        T_x_list.append(T_x)
    return H_rot_list, T_y_list, T_x_list


def angle2Rotmatrix2D(center=(0, 0), angle=0):
    c_x = center[0]
    c_y = center[1]
    a = np.cos(angle)
    b = np.sin(angle)
    x = (1-a) * c_x + b * c_y
    y = -b * c_x + (1-a) * c_y
    Mat = np.float32(([a, -b, x], [b, a, y]))
    return Mat


def get_RotMatrix_Correction(center=(0, 0), angle=0):
    c_x = 1.5*center[0]
    c_y = 1.5*center[1]
    a = np.cos(np.deg2rad(angle))
    b = np.sin(np.deg2rad(angle))
    x = (1-a) * c_x + b * c_y
    y = -b * c_x + (1-a) * c_y
    Mat = np.float32(([a, -b, x], [b, a, y], [0, 0, 1]))
    return Mat


def getRotMatrix_seq(H, c):
    '''
        Return: Rotation Matrix 2x2
    '''

    a = H[0][0]
    b = H[1][0]
    theta = np.arctan2(b, a)
    #Mat = cv.getRotationMatrix2D(center=c, angle=-np.rad2deg(theta), scale=1)
    Mat = angle2Rotmatrix2D(center=c, angle=theta)
    H_rot = Mat[0:2, 0:2]
    T_y = Mat[0][-1]
    T_x = Mat[1][-1]

    return H_rot, T_y, T_x, np.rad2deg(theta)


def transform_with_homography(h_mat, points_array):

    # add column of ones so that matrix multiplication with homography matrix is possible
    ones_col = np.ones((points_array.shape[0], 1))
    points_array = np.concatenate((points_array, ones_col), axis=1)
    transformed_points = np.matmul(h_mat, points_array.T)
    epsilon = 1e-7  # very small value to use it during normalization to avoid division by zero
    transformed_points = transformed_points / \
        (transformed_points[2, :].reshape(1, -1) + epsilon)
    transformed_points = transformed_points[0:2, :].T
    return transformed_points


def get_corners_as_array(img_height, img_width):

    corners_array = np.array([[0, 0],
                              [img_width - 1, 0],
                              [img_width - 1, img_height - 1],
                              [0, img_height - 1]])
    return corners_array


def get_crop_points_horz(img_a_h, transfmd_corners_img_b):

    # the four transformed corners of image B
    top_lft_x_hat, top_lft_y_hat = transfmd_corners_img_b[0, :]
    top_rht_x_hat, top_rht_y_hat = transfmd_corners_img_b[1, :]
    btm_rht_x_hat, btm_rht_y_hat = transfmd_corners_img_b[2, :]
    btm_lft_x_hat, btm_lft_y_hat = transfmd_corners_img_b[3, :]

    # initialize the crop points
    # since image A (on the left side) is used as pivot, x_start will always be zero
    x_start, y_start, x_end, y_end = (0, None, None, None)

    if (top_lft_y_hat > 0) and (top_lft_y_hat > top_rht_y_hat):
        y_start = top_lft_y_hat
    elif (top_rht_y_hat > 0) and (top_rht_y_hat > top_lft_y_hat):
        y_start = top_rht_y_hat
    else:
        y_start = 0

    if (btm_lft_y_hat < img_a_h - 1) and (btm_lft_y_hat < btm_rht_y_hat):
        y_end = btm_lft_y_hat
    elif (btm_rht_y_hat < img_a_h - 1) and (btm_rht_y_hat < btm_lft_y_hat):
        y_end = btm_rht_y_hat
    else:
        y_end = img_a_h - 1

    if (top_rht_x_hat < btm_rht_x_hat):
        x_end = top_rht_x_hat
    else:
        x_end = btm_rht_x_hat

    return int(x_start), int(y_start), int(x_end), int(y_end)


def get_crop_points_vert(img_a_w, transfmd_corners_img_b):

    # the four transformed corners of image B
    top_lft_x_hat, top_lft_y_hat = transfmd_corners_img_b[0, :]
    top_rht_x_hat, top_rht_y_hat = transfmd_corners_img_b[1, :]
    btm_rht_x_hat, btm_rht_y_hat = transfmd_corners_img_b[2, :]
    btm_lft_x_hat, btm_lft_y_hat = transfmd_corners_img_b[3, :]

    # initialize the crop points
    # since image A (on the top) is used as pivot, y_start will always be zero
    x_start, y_start, x_end, y_end = (None, 0, None, None)

    if (top_lft_x_hat > 0) and (top_lft_x_hat > btm_lft_x_hat):
        x_start = top_lft_x_hat
    elif (btm_lft_x_hat > 0) and (btm_lft_x_hat > top_lft_x_hat):
        x_start = btm_lft_x_hat
    else:
        x_start = 0

    if (top_rht_x_hat < img_a_w - 1) and (top_rht_x_hat < btm_rht_x_hat):
        x_end = top_rht_x_hat
    elif (btm_rht_x_hat < img_a_w - 1) and (btm_rht_x_hat < top_rht_x_hat):
        x_end = btm_rht_x_hat
    else:
        x_end = img_a_w - 1

    if (btm_lft_y_hat < btm_rht_y_hat):
        y_end = btm_lft_y_hat
    else:
        y_end = btm_rht_y_hat

    return int(x_start), int(y_start), int(x_end), int(y_end)


def get_crop_points(h_mat, img_a, img_b, stitch_direc):

    img_a_h, img_a_w, _ = img_a.shape
    img_b_h, img_b_w, _ = img_b.shape

    orig_corners_img_b = get_corners_as_array(img_b_h, img_b_w)

    transfmd_corners_img_b = transform_with_homography(
        h_mat, orig_corners_img_b)

    if stitch_direc == 1:
        x_start, y_start, x_end, y_end = get_crop_points_horz(
            img_a_w, transfmd_corners_img_b)
    # initialize the crop points
    x_start = None
    x_end = None
    y_start = None
    y_end = None

    if stitch_direc == 1:  # 1 is horizontal
        x_start, y_start, x_end, y_end = get_crop_points_horz(
            img_a_h, transfmd_corners_img_b)
    else:  # when stitching images in the vertical direction
        x_start, y_start, x_end, y_end = get_crop_points_vert(
            img_a_w, transfmd_corners_img_b)
    return x_start, y_start, x_end, y_end


def stitch_two_image(img_a, img_b):
    img_a_gray = cv.cvtColor(img_a, cv.COLOR_BGR2GRAY)
    img_b_gray = cv.cvtColor(img_b, cv.COLOR_BGR2GRAY)

    kpsA, descA = DetectAndDescribe(img_b_gray, method='orb')
    kpsB, descB = DetectAndDescribe(img_a_gray, method='orb')
    matches = matchkeypointsKNN(descA, descB, ratio=0.8, method='orb')
    print(len(matches))
    H_mat, status = getHomography(kpsA, kpsB, matches, Threshold=8)
    H_mat[-1][0] = H_mat[-1][1] = 0
    H_mat[-1][2] = 1
    print(H_mat)
    width = img_a.shape[1]+img_b.shape[1]
    height = img_a.shape[0]
    canvas = cv.warpPerspective(img_b, H_mat, (width, height))
    print(canvas)
    canvas[:, 0:img_a.shape[1], :] = img_a[:, :, :]
    xs, ys, xe, ye = get_crop_points(H_mat, img_a, img_b, 1)
    res = canvas[ys:ye, xs:xe, :]
    return res


def get_whole_Hmat(images, method='Homography'):
    n = len(images)
    H_list = []

    for i in range(0, n-1, 1):
        a = cv.cvtColor(images[i], cv.COLOR_RGB2GRAY)
        b = cv.cvtColor(images[i+1], cv.COLOR_RGB2GRAY)

        kpsA, descA = DetectAndDescribe(a, method='orb')
        kpsB, descB = DetectAndDescribe(b, method='orb')

        match = matchkeypointsKNN(descB, descA, ratio=0.6, method='orb')
        if method == 'Homography':
            H, _ = getHomography(kpsB, kpsA, match, Threshold=6)
            H[-1][0] = H[-1][1] = 0
            H[-1][2] = 1
            H_list.append(H)
    return H_list


def get_whole_Hmat_seq(image_first, image_next, method='Homography', overlap=1):
    h, w = image_first.shape[:-1]
    if overlap != 1:
        im1 = image_first[:, int(w*(1-overlap)):int(w)]
        im2 = image_next[:, 0:int(w*overlap)]
    else:
        im1 = image_first
        im2 = image_next

    a = cv.cvtColor(im1, cv.COLOR_RGB2GRAY)
    b = cv.cvtColor(im2, cv.COLOR_RGB2GRAY)

    kpsA, descA = DetectAndDescribe(a, method='orb')
    kpsB, descB = DetectAndDescribe(b, method='orb')

    if method == '2DTransform':
        match = matchkeypointsKNN(descB, descA, ratio=0.7, method='orb')
        M = get2dTrans(kpsB, kpsA, match, Threshold=6)
        H = np.identity(3, dtype=float)
        H[[0, 1]] = M[0]

    if method == 'Homography':
        match = matchkeypointsKNN(descB, descA, ratio=0.7, method='orb')
        H, _ = getHomography(kpsB, kpsA, match, Threshold=6)
        H[-1][0] = H[-1][1] = 0
        H[-1][2] = 1
    return H


def get_corner(img_a, img_b, H):
    height1, width1 = img_a.shape[:2]
    height2, width2 = img_b.shape[:2]
    corners1 = np.float32(
        ([0, 0], [0, height1], [width1, height1], [width1, 0]))
    corners2 = np.float32(
        ([0, 0], [0, height2], [width2, height2], [width2, 0]))
    warpedCorners2 = np.zeros((4, 2))
    for i in range(0, 4):
        cornerX = corners2[i, 0]
        cornerY = corners2[i, 1]
        # /(H[2,0]*cornerX + H[2,1]*cornerY + H[2,2])
        warpedCorners2[i, 0] = (H[0, 0]*cornerX + H[0, 1]*cornerY + H[0, 2])
        # /(H[2,0]*cornerX + H[2,1]*cornerY + H[2,2])
        warpedCorners2[i, 1] = (H[1, 0]*cornerX + H[1, 1]*cornerY + H[1, 2])
    allCorners = np.concatenate((corners1, warpedCorners2), axis=0)
    [xMin, yMin] = np.int32(allCorners.min(axis=0).ravel() - 0.5)
    [xMax, yMax] = np.int32(allCorners.max(axis=0).ravel() + 0.5)
    return xMin, xMax, yMin, yMax


def center_image(image):
    height = image.shape[0]
    width = image.shape[1]
    wi = (width/2)
    he = (height/2)
    image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    ret, thresh = cv.threshold(image_gray, 95, 255, 0)

    M = cv.moments(thresh)

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    offsetX = (wi-cX)
    offsetY = (he-cY)
    T = np.float32([[1, 0, offsetX], [0, 1, offsetY]])
    centered_image = cv.warpAffine(image, T, (width, height))

    return centered_image


def rot_correction(w1, w2, r):
    x_0 = r*np.cos(np.deg2rad(w1))
    y_0 = r*np.sin(np.deg2rad(w1))
    x_now = r*np.cos(np.deg2rad(w2))
    y_now = r*np.sin(np.deg2rad(w2))

    dx = x_now - x_0
    dy = y_now - y_0
    return dx, dy


def rot_corr(c, angle):
    c_x = 1.75*c[0]
    c_y = 2*c[1]
    a = np.cos(np.deg2rad(angle))
    b = np.sin(np.deg2rad(angle))
    x = (1-a) * c_x + b * c_y
    y = -b * c_x + (1-a) * c_y
    return x, y


def rot_corr_test(c, angle):
    c_x = c[0]
    c_y = c[1]
    a = np.cos(np.deg2rad(angle))
    b = np.sin(np.deg2rad(angle))
    x = (1-a) * c_x + b * c_y
    y = -b * c_x + (1-a) * c_y
    return x, y
