import numpy as np

def flow_lk_patch(Ix, Iy, It, x, y, size=5):
    """
    Find the Lucas-Kanade optical flow on a single square patch.
    The patch is centered at (y, x), therefore it generally extends
    from x-size//2 to x+size//2 (inclusive), same for y, EXCEPT when
    exceeding image boundaries!
    
    WARNING: Pay attention to how you index the images! The first coordinate
    is actually the y-coordinate, and the second coordinate is the x-coordinate.
    
    Inputs:
        - Ix: image gradient along the x-dimension - shape: (H, W)
        - Iy: image gradient along the y-dimension - shape: (H, W)
        - It: image time-derivative - shape: (H, W)
        - x: SECOND coordinate of patch center - integer in range [0, W-1]
        - y: FIRST coordinate of patch center - integer in range [0, H-1]
        - size: optional parameter to change the side length of the patch in pixels
    
    Outputs:
        - flow: flow estimate for this patch - shape: (2,)
        - conf: confidence of the flow estimates - scalar
    """

    # use the gradient constraint equation for the flow estimation
    # Ix*u + Ty*v + It = 0 => [Ix Iy It] x [u v 1].T = 0 for each pixel
    # construct a size^2 x 2 matrix for spatial gradients
    # construct a size^2 x 1 matrix for temporal gradients

    # get the derivatives for the patch
    height, width = Ix.shape
    x_min_bound, x_max_bound = max(x-size//2, 0), min((x+size//2)+1, width)
    y_min_bound, y_max_bound = max(y-size//2, 0), min((y+size//2)+1, height)

    Ix_patch = Ix[y_min_bound:y_max_bound, x_min_bound:x_max_bound]
    Iy_patch = Iy[y_min_bound:y_max_bound, x_min_bound:x_max_bound]
    It_patch = It[y_min_bound:y_max_bound, x_min_bound:x_max_bound]

    # initialize a matrix to store spatial gradient values at each pixel in patch
    I_patch_spatial = np.zeros((Ix_patch.shape[0]*Ix_patch.shape[1], 2))

    # initialize a matrix to store spatial gradient values at each pixel in patch
    I_patch_temporal = np.zeros((Ix_patch.shape[0]*Ix_patch.shape[1], 1))

    i = 0   # intialize the counter
    # cycle through each pixel in the patch
    for row in range(Ix_patch.shape[0]):
        for col in range(Ix_patch.shape[1]):
            Ix_pixel = Ix_patch[row, col]
            Iy_pixel = Iy_patch[row, col]
            It_pixel = It_patch[row, col]
            I_patch_spatial[i, :] = Ix_pixel, Iy_pixel
            I_patch_temporal[i, :] = It_pixel
            i+=1
    
    # use least square method for best possible solution
    # use equation I_spatial * [u v].T = = -I_temporal (gradient constraint equation)
    solution, _, _, singular_values = np.linalg.lstsq(I_patch_spatial, -I_patch_temporal, rcond=-1)
    
    # normalize the solution
    if solution.shape[0] != 0 and solution.shape[1] != 0:
        u, v = np.ravel(solution)
    else:
        u, v = np.array([0, 0])
    
    # populate the solution
    flow = np.array([u, v])
    conf = min(singular_values)

    return flow, conf


def flow_lk(Ix, Iy, It, size=5):
    """
    Compute the Lucas-Kanade flow for all patches of an image.
    To do this, iteratively call flow_lk_patch for all possible patches.
    
    WARNING: Pay attention to how you index the images! The first coordinate
    is actually the y-coordinate, and the second coordinate is the x-coordinate.
    
    Inputs:
        - Ix: image gradient along the x-dimension - shape: (H, W)
        - Iy: image gradient along the y-dimension - shape: (H, W)
        - It: image time-derivative
    Outputs:
        - image_flow: flow estimate for each patch - shape: (H, W, 2)
        - confidence: confidence of the flow estimates - shape: (H, W)
    """

    # double for-loop to iterate over all patches

    # get the x and y dimensions
    y_dim, x_dim = Ix.shape[0], Ix.shape[1]

    # initialize the image flow and confidence
    image_flow  = np.zeros((y_dim, x_dim, 2))
    confidence  = np.zeros((y_dim, x_dim))

    # loop throught the pixels
    for y_pos in range(0, y_dim):
        for x_pos in range(0, x_dim):
            image_flow[y_pos, x_pos, :], confidence[y_pos, x_pos] = \
            flow_lk_patch(Ix, Iy, It, x_pos, y_pos, size)

    
    return image_flow, confidence