import numpy as np

def depth(flow, confidence, ep, K, thres=10):
    """
    Compute the depth map from the flow and confidence map.
    
    Inputs:
        - flow: optical flow - shape: (H, W, 2)
        - confidence: confidence of the flow estimates - shape: (H, W)
        - ep: epipole - shape: (3,)
        - K: intrinsic calibration matrix - shape: (3, 3)
        - thres: threshold for confidence (optional) - scalar
    
    Output:
        - depth_map: depth at every pixel - shape: (H, W)
    """
    depth_map = np.zeros_like(confidence)
    
    # 1. Find where flow is valid (confidence > threshold)
    # 2. Convert these pixel locations to normalized projective coordinates
    # 3. Same for epipole and flow vectors
    # 4. Now find the depths using the formula from the lecture slides
    
    # find the indices where the flow is valid according to the confidence thereshold
    valid_idx = confidence > thres

    # find the valid pixel locations corresponding to the premissible flows
    valid_pixels = np.argwhere(valid_idx)

    # initialize the matrices for homogeneous pixel coodinates, flow velocities
    pixel_positions = np.zeros((valid_pixels.shape[0], 3))

    # convert to homogeneous coordinates
    for i in range(valid_pixels.shape[0]):
        pixel_positions[i] = [valid_pixels[i, 0], valid_pixels[i, 1], 1]

    # convert to the camera coordinates
    pixels_wrt_camera = np.matmul(np.linalg.inv(K), np.transpose(pixel_positions))

    # convert the epipolar coordinates from pixel to camera coordinates
    ep_camera = np.matmul(np.linalg.inv(K), np.reshape(ep,(3, 1)))

    # compte the valid flows
    valid_flows = flow[valid_idx]

    # use the following equation p_dot = (Vz/Z) (x - Vx/Vz, y - Vy/Vz).T
    # we know that (Vx/Vz, Vy/Vz) = epipole in the camera coordinates
    difference = pixels_wrt_camera - ep_camera

    # using the hint to solve the equations, (||p - FOE|| / ||p_dot||)
    point_depth = np.linalg.norm(difference, axis=0) / np.linalg.norm(valid_flows, axis=1)

    depth_map[valid_idx] = point_depth
    
    ## Truncate the depth map to remove outliers
    
    # require depths to be positive
    truncated_depth_map = np.maximum(depth_map, 0) 
    valid_depths = truncated_depth_map[truncated_depth_map > 0]
    
    # You can change the depth bound for better visualization if you depth is in different scale
    depth_bound = valid_depths.mean() + 10 * np.std(valid_depths)
    print(f'depth bound: {depth_bound}')

    # set depths above the bound to 0 and normalize to [0, 1]
    truncated_depth_map[truncated_depth_map > depth_bound] = 0
    truncated_depth_map = truncated_depth_map / truncated_depth_map.max()

    return truncated_depth_map
