import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """

    # To estimate the world coordinates of the pixels, we follow the method below
    # 1. (xw yw zw).T = lamda*Rwc*K^{-1}(u v 1).T + t_wc
    # 2. zw = 0 for all the points on the table => last row on R.H.S = 0
    # 3. We get lambda by equating the last row on RHS to zero (since zw=0)

    pixel_vector = np.array([pixel for pixel in pixels])
    pixel_vector = np.append(pixel_vector, np.ones((pixel_vector.shape[0], 1)), axis=1)
    
    # rotational components
    C = np.matmul(R_wc, np.dot(np.linalg.inv(K), pixel_vector.T))

    # calculate lambda for correction using scale factor
    L = - t_wc[2] / C[2, :]

    # get the x, y world coordinates of the pixels
    coordinates = L*C + t_wc.reshape(-1, 1)

    # populate the coordinates into the matrix Pw
    Pw = coordinates.T
    Pw[:, -1] = 0              # since the cooridnate system is on the april tag


    return Pw