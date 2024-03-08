import numpy as np

def epipole(flow_x, flow_y, smin, thresh, num_iterations=None):
    """
    Compute the epipole from the flows,
    
    Inputs:
        - flow_x: optical flow on the x-direction - shape: (H, W)
        - flow_y: optical flow on the y-direction - shape: (H, W)
        - smin: confidence of the flow estimates - shape: (H, W)
        - thresh: threshold for confidence - scalar
    	- Ignore num_iterations
    Outputs:
        - ep: epipole - shape: (3,)
    """
    # Logic to compute the points you should use for your estimation
    # We only look at image points above the threshold in our image
    # Due to memory constraints, we cannot use all points on the autograder
    # Hence, we give you valid_idx which are the flattened indices of points
    # to use in the estimation estimation problem 
    good_idx = np.flatnonzero(smin>thresh)
    permuted_indices = np.random.RandomState(seed=10).permutation(
        good_idx
    )
    valid_idx=permuted_indices[:3000]

    ### STUDENT CODE START - PART 1 ###
    
    # 1. For every pair of valid points, compute the epipolar line (use np.cross)
    # Hint: for faster computation and more readable code, avoid for loops! Use vectorized code instead.
    
    # get the x and y dimensions
    y_dim, x_dim = flow_x.shape[0], flow_x.shape[1]

    # generate the meshgrid and considering the origin at the center of the image
    xp = np.linspace(-x_dim//2, x_dim//2, num=x_dim, endpoint=False, dtype=int)
    yp = np.linspace(-y_dim//2, y_dim//2, num=y_dim, endpoint=False, dtype=int)

    Xp, Yp = np.meshgrid(xp, yp)

    # get only permissible pixel positions according to the thresholding
    Xp = np.ravel(Xp)[valid_idx]
    Yp = np.ravel(Yp)[valid_idx]

    # get only permissible flow values according to the thresholding
    U = np.ravel(flow_x)[valid_idx]
    V = np.ravel(flow_y)[valid_idx]

    # initialize the matrices for homogeneous pixel coodinates, flow velocities
    pixel_positions = np.zeros((len(valid_idx), 3))
    flow_vectors = np.zeros((len(valid_idx), 3))

    i = 0   # intialize the counter
    # create homogeneous coordinates of the pixel position, flow vectors
    for row, col in zip(range(len(valid_idx)), range(len(valid_idx))):
        pixel_positions[i, :] = Xp[row], Yp[col], 1
        flow_vectors[i, :] = U[row], V[col], 0
        i+=1
    
    # solve for the epipole using the equation e.T(xp X u) = 0
    # xp X u --> epipolar lines => e is basically the point of intersection of 
    # all the lines
    # It is enough to find the SVD of concatenated matrix xp X u and get the
    # epipole from the last column of SVD

    epipolar_lines = np.cross(pixel_positions, flow_vectors)
    U, S, Vt = np.linalg.svd(epipolar_lines)
    ep = Vt[-1]

    return ep