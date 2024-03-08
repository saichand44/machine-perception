import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import cv2
import open3d as o3d


from dataloader import load_middlebury_data
from utils import viz_camera_poses

EPS = 1e-8


def homo_corners(h, w, H):
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)
    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)
    return u_min, u_max, v_min, v_max


def rectify_2view(rgb_i, rgb_j, rect_R_i, rect_R_j, K_i, K_j, u_padding=20, v_padding=20):
    """Given the rectify rotation, compute the rectified view and corrected projection matrix

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    rect_R_i,rect_R_j : [3,3]
        p_rect_left = rect_R_i @ p_i
        p_rect_right = rect_R_j @ p_j
    K_i,K_j : [3,3]
        original camera matrix
    u_padding,v_padding : int, optional
        padding the border to remove the blank space, by default 20

    Returns
    -------
    [H,W,3],[H,W,3],[3,3],[3,3]
        the rectified images
        the corrected camera projection matrix. WE HELP YOU TO COMPUTE K, YOU DON'T NEED TO CHANGE THIS
    """
    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert rgb_i.shape == rgb_j.shape, "This hw assumes the input images are in same size"
    h, w = rgb_i.shape[:2]

    ui_min, ui_max, vi_min, vi_max = homo_corners(h, w, K_i @ rect_R_i @ np.linalg.inv(K_i))
    uj_min, uj_max, vj_min, vj_max = homo_corners(h, w, K_j @ rect_R_j @ np.linalg.inv(K_j))

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(ui_max, uj_max))) - u_padding * 2
    h_max = int(np.floor(min(vi_max - vi_min, vj_max - vj_min))) - v_padding * 2

    assert K_i[0, 2] == K_j[0, 2], "This hw assumes original K has same cx"
    K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
    K_i_corr[0, 2] -= u_padding
    K_i_corr[1, 2] -= vi_min + v_padding
    K_j_corr[0, 2] -= u_padding
    K_j_corr[1, 2] -= vj_min + v_padding

    # from the hint in the homework, 
    # {K^-1}_corr x [urect vrect 1].T = rect_R x K^-1 x [u v 1].T
    # [urect vrect 1].T = K_corr x rect_R x K^-1 x [u v 1].T
    # => H ~ K_corr x rect_R x K^-1

    H_i = np.matmul(K_i_corr, np.matmul(rect_R_i, np.linalg.inv(K_i)))
    H_i = H_i/H_i[-1, -1]

    H_j = np.matmul(K_j_corr, np.matmul(rect_R_j, np.linalg.inv(K_j)))
    H_j = H_j/H_j[-1, -1]

    # wrap the images onto the parallel planes where epipoles are parallel
    rgb_i_rect =  cv2.warpPerspective(rgb_i, H_i, (w_max, h_max))
    rgb_j_rect =  cv2.warpPerspective(rgb_j, H_j, (w_max, h_max))


    return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr


def compute_right2left_transformation(i_R_w, i_T_w, j_R_w, j_T_w):
    """Compute the transformation that transform the coordinate from j coordinate to i

    Parameters
    ----------
    i_R_w, j_R_w : [3,3]
    i_T_w, j_T_w : [3,1]
        p_i = i_R_w @ p_w + i_T_w
        p_j = j_R_w @ p_w + j_T_w
    Returns
    -------
    [3,3], [3,1], float
        p_i = i_R_j @ p_j + i_T_j, B is the baseline
    """

    # i_R_w = rotation matix for world coordinates to i frame
    # i_T_w = translation matix for world coordinates to i frame
    # j_R_w = rotation matix for world coordinates to j frame
    # j_T_w = translation matix for world coordinates to j frame

    # i_R_j = i_R_w x w_R_j => i_R_w x j_R_w^-1
    # i_T_j = i_T_w - j_T_w => i_T_w - i_R_j x j_T_w
    # baseline is the magnitude of the translation vector from j to i
    i_R_j = np.matmul(i_R_w, np.linalg.inv(j_R_w))
    i_T_j = i_T_w - np.matmul(i_R_j, j_T_w)
    B = np.linalg.norm(i_T_j)


    return i_R_j, i_T_j, B


def compute_rectification_R(i_T_j):
    """Compute the rectification Rotation

    Parameters
    ----------
    i_T_j : [3,1]

    Returns
    -------
    [3,3]
        p_rect = rect_R_i @ p_i
    """
    # check the direction of epipole, should point to the positive direction of y axis
    e_i = i_T_j.squeeze(-1) / (i_T_j.squeeze(-1)[1] + EPS) # along y axis

    # the epipole (in y direction) coincides with the translation vector
    r2 = e_i
    r2 = r2 / np.linalg.norm(r2)
    r2 = np.ravel(r2)

    r1 = np.array([i_T_j[1, 0], -i_T_j[0, 0], 0])
    r1 = r1 / (np.linalg.norm(r1)+EPS)
    r1 = np.ravel(r1)

    r3 = np.cross(r1, r2)
    r3 = np.ravel(r3)

    rect_R_i = np.vstack((r1, r2, r3))


    return rect_R_i


def ssd_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]


    # reshape the arrays
    src_reshaped = np.reshape(src, ((src.shape[0], src.shape[1], 3)))
    dst_reshaped = np.reshape(dst, ((dst.shape[0], dst.shape[1], 3)))

    # add a new axis to the src and dst to use vectorization for the difference
    # of each patch in src to all other patches in dst
    src_reshaped = np.expand_dims(src_reshaped, axis=1)
    dst_reshaped = np.expand_dims(dst_reshaped, axis=0)

    # compute the difference
    difference = src_reshaped - dst_reshaped

    # compute the ssd
    ssd = np.sum(np.linalg.norm(difference, axis=2)**2, axis=2)


    return ssd  # M,N


def sad_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    # reshape the arrays
    src_reshaped = np.reshape(src, ((src.shape[0], src.shape[1], 3)))
    dst_reshaped = np.reshape(dst, ((dst.shape[0], dst.shape[1], 3)))

    # add a new axis to the src and dst to use vectorization for the difference
    # of each patch in src to all other patches in dst
    src_reshaped = np.expand_dims(src_reshaped, axis=1)
    dst_reshaped = np.expand_dims(dst_reshaped, axis=0)

    # compute the difference
    difference = src_reshaped - dst_reshaped

    # compute the sad
    sad = np.sum(np.sum(np.abs(difference), axis=2), axis=2)

    return sad  # M,N


def zncc_kernel(src, dst):
    """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    # reshape the arrays
    src_reshaped = np.reshape(src, ((src.shape[0], src.shape[1], 3)))
    dst_reshaped = np.reshape(dst, ((dst.shape[0], dst.shape[1], 3)))

    # compute the means
    src_mean = np.mean(src_reshaped, axis=1)
    src_mean = np.expand_dims(src_mean, axis=1)
    dst_mean = np.mean(dst_reshaped, axis=1)
    dst_mean = np.expand_dims(dst_mean, axis=1)

    # compute the sigma
    src_sigma = np.std(src_reshaped, axis=1)
    src_sigma = np.expand_dims(src_sigma, axis=1)
    dst_sigma = np.std(dst_reshaped, axis=1)
    dst_sigma = np.expand_dims(dst_sigma, axis=0)

    # compute the zero mean
    zero_mean_src = (src_reshaped - src_mean)
    zero_mean_src = np.expand_dims(zero_mean_src, axis=1)
    zero_mean_dst = (dst_reshaped - dst_mean)
    zero_mean_dst = np.expand_dims(zero_mean_dst, axis=0)

    # compute normalized cross correlation
    zncc = np.sum(zero_mean_src*zero_mean_dst, axis=2)
    zncc = zncc / ((src_sigma*dst_sigma)+EPS)
    zncc = np.sum(zncc, axis=2)

    return zncc * (-1.0)  # M,N


def image2patch(image, k_size):
    """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

    Parameters
    ----------
    image : [H,W,3]
    k_size : int, must be odd number; your function should work when k_size = 1

    Returns
    -------
    [H,W,k_size**2,3]
        The patch buffer for each pixel
    """

    # define the padding length
    pad_length = k_size // 2

    # get the dimensions of the image
    height, width, _ = image.shape

    # initialize the patch_buffer with np.zeros
    patch_buffer = np.zeros((height, width, k_size**2, 3))

    # add zero-pad to the the image
    padded_image = np.pad(image, ((pad_length, pad_length), (pad_length, pad_length), 
                                  (0, 0)), mode='constant', constant_values=0)

    # populate the patch_buffer by traversing each pixel coordinates in the given image
    for y_coor in range(height):
        for x_coor in range(width):
            # extract the patch for each pixel location
            image_patch = padded_image[y_coor:y_coor+k_size, x_coor:x_coor+k_size, :]
            patch_buffer[y_coor, x_coor] = image_patch.reshape(-1, 3)

    return patch_buffer  # H,W,K**2,3


def compute_disparity_map(rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel,  img2patch_func=image2patch):
    """Compute the disparity map from two rectified view

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    d0 : see the hand out, the bias term of the disparty caused by different K matrix
    k_size : int, optional
        The patch size, by default 3
    kernel_func : function, optional
        the kernel used to compute the patch similarity, by default ssd_kernel
    img2patch_func: function, optional
        the function used to compute the patch buffer, by default image2patch
        (there is NO NEED to alter this argument)

    Returns
    -------
    disp_map: [H,W], dtype=np.float64
        The disparity map, the disparity is defined in the handout as d0 + vL - vR

    lr_consistency_mask: [H,W], dtype=np.float64
        For each pixel, 1.0 if LR consistent, otherwise 0.0
    """
    
    # NOTE: when computing patches, please use the syntax:
    # patch_buffer = img2patch_func(image, k_size)
    # DO NOT DIRECTLY USE: patch_buffer = image2patch(image, k_size), as it may cause errors in the autograder

    #NOTE: large portion of this code taken from .ipynb file (section - 1.2)

    # get the dimensions of the image
    h, w = rgb_i.shape[:2]

    #initialize the disp_map and lr_consistency_mask
    disp_map = np.zeros((h, w), dtype=np.float64)
    lr_consistency_mask = np.zeros((h, w), dtype=np.float64)

    patches_i = img2patch_func(rgb_i.astype(float) / 255.0, k_size)  # [h,w,k*k,3]
    patches_j = img2patch_func(rgb_j.astype(float) / 255.0, k_size)  # [h,w,k*k,3]

    vi_idx, vj_idx = np.arange(h), np.arange(h)
    disp_candidates = vi_idx[:, None] - vj_idx[None, :] + d0
    valid_disp_mask = disp_candidates > 0.0

    # loop along the width (axis=1)(hint from the given stub code in .ipynb)
    for col in tqdm(range(w)):
        buf_i, buf_j = patches_i[:, col], patches_j[:, col]
        value = kernel_func(buf_i, buf_j)
        _upper = value.max() + 1.0
        value[~valid_disp_mask] = _upper

        # search for the least index along the column for best-matched right patch
        # corresponding to each left patch
        best_matched_right_pixel = np.argmin(value, axis=1) # along the column

        # search for the least index along the row for best-matched left patch
        # corresponding to each right patch
        best_matched_left_pixel = np.argmin(value[:, best_matched_right_pixel], axis=0)

        # populate the disparity map
        disp_map[:, col] = disp_candidates[vi_idx, best_matched_right_pixel]

        # populate the lr_consistency_mask
        lr_consistency_mask[:, col] = (best_matched_left_pixel == vi_idx)

    return disp_map, lr_consistency_mask


def compute_dep_and_pcl(disp_map, B, K):
    """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
    compute the depth map and backprojected point cloud

    Parameters
    ----------
    disp_map : [H,W]
        disparity map
    B : float
        baseline
    K : [3,3]
        camera matrix

    Returns
    -------
    [H,W]
        dep_map
    [H,W,3]
        each pixel is the xyz coordinate of the back projected point cloud in camera frame
    """

    # using the equation disparity = b*f/Z => Z = b*f / disparity
    dep_map = (K[1, 1] * B) / disp_map   # assuming fx=fy

    # generate a mesh grid where we can back propagate 3D points from each pixel in the
    # mesh grid
    u, v = np.meshgrid(np.arange(disp_map.shape[1]), np.arange(disp_map.shape[0]))

    # from camera projection equation, 
    # u = f * Xc/Zc + u0 and v = f * Yc/Zc + v0 where f, u0, v0 can  be obtained from K
    x_cam = (u - K[0, 2])*dep_map/K[0, 0]
    y_cam = (v - K[1, 2])*dep_map/K[1, 1]

    xyz_cam = np.stack((x_cam, y_cam, dep_map), axis=2)

    return dep_map, xyz_cam


def postprocess(
    dep_map,
    rgb,
    xyz_cam,
    c_R_w,
    c_T_w,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near=0.45,
    z_far=0.65,
):
    """
    Goal in this function is: 
    given pcl_cam [N,3], c_R_w [3,3] and c_T_w [3,1]
    compute the pcl_world with shape[N,3] in the world coordinate
    """

    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

    # constraint z-near, z-far
    mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(float)
    # imageio.imsave("./debug_dep_mask.png", mask_dep)

    mask = np.minimum(mask_dep, mask_hsv)
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)
    # imageio.imsave("./debug_before_xyz_mask.png", mask)

    # filter xyz point cloud
    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    _pcl_mask = np.zeros(pcl_cam.shape[0])
    _pcl_mask[ind] = 1.0
    pcl_mask = np.zeros(xyz_cam.shape[0] * xyz_cam.shape[1])
    pcl_mask[mask.reshape(-1) > 0] = _pcl_mask
    mask_pcl = pcl_mask.reshape(xyz_cam.shape[0], xyz_cam.shape[1])
    # imageio.imsave("./debug_pcl_mask.png", mask_pcl)
    mask = np.minimum(mask, mask_pcl)
    # imageio.imsave("./debug_final_mask.png", mask)

    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]
    
    # c_R_w = Rotation matrix for world coordinates wrt camera
    # c_T_w = Translation matrix for world coordinates wrt camera
    # pcl_world = w_R_c x pcl_cam + w_T_c
    w_R_c = np.transpose(c_R_w)
    w_T_c = -np.matmul(np.transpose(c_R_w), c_T_w)
    pcl_world = np.matmul(w_R_c, pcl_cam.T).T + w_T_c.T

    # np.savetxt("./debug_pcl_world.txt", np.concatenate([pcl_world, pcl_color], -1))
    # np.savetxt("./debug_pcl_rect.txt", np.concatenate([pcl_cam, pcl_color], -1))

    return mask, pcl_world, pcl_cam, pcl_color


def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline

    # * 1. rectify the views
    i_R_w, i_T_w = view_i["R"], view_i["T"][:, None]  # p_i = i_R_w @ p_w + i_T_w
    j_R_w, j_T_w = view_j["R"], view_j["T"][:, None]  # p_j = j_R_w @ p_w + j_T_w

    i_R_j, i_T_j, B = compute_right2left_transformation(i_R_w, i_T_w, j_R_w, j_T_w)
    assert i_T_j[1, 0] > 0, "here we assume view i should be on the left, not on the right"

    rect_R_i = compute_rectification_R(i_T_j)

    rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        rect_R_i,
        rect_R_i @ i_R_j,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert K_i_corr[1, 1] == K_j_corr[1, 1], "This hw assumes the same focal Y length"
    assert (K_i_corr[0] == K_j_corr[0]).all(), "This hw assumes the same K on X dim"
    assert (
        rgb_i_rect.shape == rgb_j_rect.shape
    ), "This hw makes rectified two views to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        rgb_i_rect,
        rgb_j_rect,
        d0=K_j_corr[1, 2] - K_i_corr[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        dep_map,
        rgb_i_rect,
        xyz_cam,
        rect_R_i @ i_R_w,
        rect_R_i @ i_T_w,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, dep_map


def main():
    DATA = load_middlebury_data("data/templeRing")
    # viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return


if __name__ == "__main__":
    main()
