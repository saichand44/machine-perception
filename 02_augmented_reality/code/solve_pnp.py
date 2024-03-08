from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    # Homography Approach
    # Following slides: Pose from Projective Transformation

    # initialize the R and t that describe the camera w.r.t world
    R = np.zeros((3, 3))
    t = np.zeros((3, 1))

    # initialize the Rc and tc that describe the world w.r.t camera
    Rc = np.zeros((3, 3))
    tc = np.zeros((3, 1))

    # compute homography matrix (remove the last column of Pw to have (x,y) format)
    H = est_homography(Pw[:, :2], Pc)

    # compute H^{1} = K^{-1}*H
    K_inverse = np.linalg.inv(K)
    H_dash = np.matmul(K_inverse, H)

    # To minimize the function ||(a b c) - lambda*(r1 r2 T)||, we use
    # SVD on (a b) and then extract r1, r2 from it
    # use the scaling / digonal matrix to coompute lambda and then T = c/lambda
    U, S, V_T = np.linalg.svd(H_dash[:, :2], full_matrices=False)

    r1 = np.matmul(U, V_T)[:, 0]         # first column of rotation matrix
    r1 = (1/np.linalg.norm(r1))*r1       # normalize the column matrix

    r2 = np.matmul(U, V_T)[:, 1]         # second column of rotation matrix
    r2 = (1/np.linalg.norm(r2))*r2       # normalize the column matrix

    L = np.sum(S)/2                      # lambda 

    Rc[:, 0] = r1                        # populate the first column of Rc
    Rc[:, 1] = r2                        # populate the second column of Rc
    Rc[:, 2] = np.cross(r1, r2)          # populate the third column of Rc
    Rc[:, 2] = (1/np.linalg.det(Rc))*Rc[:, 2]  # to make determinant of Rc = 1
    tc = (1/L)*H_dash[:, -1]             # populate the translation matrix tc

    # Rc and tc are related to world w.r.t the camera and
    # R and t are related to camera w.r.t the world
    # => R = Rc.T and t = -Rc.T * tc   (reference class notes)
    R = np.linalg.inv(Rc)
    t = -np.matmul(np.transpose(Rc), tc)

    return R, t
