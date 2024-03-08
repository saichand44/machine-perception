import numpy as np
from est_homography import est_homography


def warp_pts(X, Y, interior_pts):
    """
    First compute homography from video_pts to logo_pts using X and Y,
    and then use this homography to warp all points inside the soccer goal

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
        interior_pts: Nx2 matrix of points inside goal
    Returns:
        warped_pts: Nx2 matrix containing new coordinates for interior_pts.
        These coordinate describe where a point inside the goal will be warped
        to inside the penn logo. For this assignment, you can keep these new
        coordinates as float numbers.

    """

    H = est_homography(X, Y)

    # initilize the wraped points (of size Nx2) for interior_pts
    warped_pts = np.zeros((len(interior_pts), 2))

    # iterate over each interior point and find the wrapped point
    for i in range(len(interior_pts)):  
        point = interior_pts[i]
        point = np.append(point, 1)   # homogeneous coordinates

        # wraped point: interior point mapped onto the penn logo
        warped_pt = np.matmul(H, point)
        warped_pt = warped_pt/warped_pt[2]
        warped_pts[i] = warped_pt[:2]

    return warped_pts
