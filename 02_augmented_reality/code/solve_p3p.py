import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    def _get_distance_angle(world_point1, world_point2, pixel1, pixel2, K):
        '''
        Compute the distance between world points and the cosine of the angle 
        between the rays calculated from the corresponding pixels.
        '''
        pixel1_vec = np.hstack((pixel1[0], pixel1[1], 1))
        pixel2_vec = np.hstack((pixel2[0], pixel2[1], 1))

        ray1 = np.matmul(np.linalg.inv(K), pixel1_vec.T)
        ray2 = np.matmul(np.linalg.inv(K), pixel2_vec.T)

        ray1_unit = np.array([ray1 / np.linalg.norm(ray1)])
        ray2_unit = np.array([ray2 / np.linalg.norm(ray2)])

        # Compute cosine of the angle
        cos_angle = np.dot(ray1_unit, ray2_unit.T)

        # Compute distance
        distance = np.linalg.norm(world_point1 - world_point2)

        return distance, cos_angle

    def _get_unit_vector(pixel, K):
        '''
        Compute the unit vector of the ray joining pixel and camera center
        '''
        pixel_vec = np.hstack((pixel[0], pixel[1], 1))
        ray = np.matmul(np.linalg.inv(K), pixel_vec.T)
        ray_unit = np.array([ray / np.linalg.norm(ray)])

        return ray_unit

    # Choose the vertices a, b, and d on the April tag as the three points
    pw1 = Pw[0]              # world coordinates
    pw2 = Pw[1]              # world coordinates          
    pw3 = Pw[2]              # world coordinates
    pw4 = Pw[3]              # world coordinates

    pc1 = Pc[0]              # pixel coordinates
    pc2 = Pc[1]              # pixel coordinates          
    pc3 = Pc[2]              # pixel coordinates
    pc4 = Pc[3]              # pixel coordinates

    # Compute known parameters (distances and cosine of angles)
    # Note: Here alpha, beta, gamma are not angles but cosines of the angles
    d23, alpha = _get_distance_angle(pw3, pw4, pc3, pc4, K)
    d13, beta  = _get_distance_angle(pw2, pw4, pc2, pc4, K)
    d12, gamma = _get_distance_angle(pw2, pw3, pc2, pc3, K)

    # Let s2 = u*s1 and s3 = v*s1 (ref. P3P slides) => a 4th degree polynomial in v
    # A4*v^4 + A3*v^3 + A2*v^2 + A1*v + A0 = 0
    A0 = (1 + (d23 ** 2 - d12 ** 2) / d13 ** 2) ** 2 - 4 * (d23 ** 2 / d13 ** 2) * gamma ** 2
    A1 = 4 * (-((d23 ** 2 - d12 ** 2) / d13 ** 2) * (1 + ((d23 ** 2 - d12 ** 2) / d13 ** 2)) * beta + 2 * (d23 ** 2 / d13 ** 2) * (gamma ** 2) * beta - (1 - ((d23 ** 2 + d12 ** 2) / d13 ** 2)) * alpha * gamma)
    A2 = 2 * (((d23 ** 2 - d12 ** 2) / d13 ** 2) ** 2 - 1 + 2 * ((d23 ** 2 - d12 ** 2) / d13 ** 2) ** 2 * beta** 2 + 2 * ((d13 ** 2 - d12 ** 2) / d13 ** 2) * alpha ** 2 - 4 * ((d23 ** 2 + d12 ** 2) / d13 ** 2) * alpha * beta * gamma + 2 * ((d13 ** 2 - d23 ** 2) / d13 ** 2) * gamma ** 2)
    A3 = 4 * (((d23 ** 2 - d12 ** 2) / d13 ** 2) * (1 - ((d23 ** 2 - d12 ** 2) / d13 ** 2)) * beta - (1 - ((d23 ** 2 + d12 ** 2) / d13 ** 2)) * alpha * gamma + 2 * (d12 ** 2 / d13 ** 2) * (alpha ** 2) * beta)
    A4 = ((d23 ** 2 - d12 ** 2) / d13 ** 2 - 1) ** 2 - 4 * (d12 ** 2 / d13 ** 2) * alpha ** 2

    coefficients = [A4, A3, A2, A1, A0]

    # initilaize a list to store the values of u, v, s1, s2, s3
    u  = []
    v  = []
    s1 = []
    s2 = []
    s3 = []

    # calculate the roots
    roots = np.roots(np.ravel(np.array(coefficients)))
    roots = np.real(roots[roots > 0])

    for root in roots:
        v.append(root)

    # compute the value of u corresponding to each v
    for root in v:
        u_val = ((-1 + (d23**2 - d12**2) / d13**2) * root**2 - 2 * ((d23**2 - d12**2) / d13**2) * beta * root + 1 + ((d23**2 - d12**2) / d13**2)) / (2 * (gamma - root * alpha))
        u.append(u_val)

    # compute the values of s1, s2, s3 from u, v
    # s2 = u*s1 and s3 = v*s1 
    for u_value, v_value in zip(u, v):
        s1_val = np.sqrt((d13**2) / (1 + v_value**2 - 2 * v_value * beta))
        s1.append(s1_val)
        s2.append(u_value * s1_val)
        s3.append(v_value * s1_val)

    # define the parameter to capture the least error norm of the reprojection
    least_norm = None

    # extract the camera coordinates from the values of s1, s2, s3
    for s1_val, s2_val, s3_val in zip(s1, s2, s3):
        Pc_3d = np.vstack((
            s1_val * _get_unit_vector(pc2, K),
            s2_val * _get_unit_vector(pc3, K),
            s3_val * _get_unit_vector(pc4, K),
        ))

        # compute Rc, tc that describe the world w.r.t camera
        Rc, tc = Procrustes(Pw[1:], Pc_3d)

        # estimate the pixel coordinate of the remaining world point
        Pc_estimate = np.dot(K, np.dot(Rc, pw1.T) + tc)
        Pc_estimate = (1 / Pc_estimate[-1]) * Pc_estimate
        Pc_estimate = Pc_estimate[:-1]
        
        # calculate the error of the reprojection
        error_norm = np.linalg.norm(Pc_estimate - pc1)

        # search for the combination of R, t that gives the least error
        if least_norm is None:
            least_norm = error_norm
            R_true = Rc
            t_true = tc
        elif error_norm < least_norm:
            least_norm = error_norm
            R_true = Rc
            t_true = tc

    # Rc and tc are related to world w.r.t the camera and
    # R and t are related to camera w.r.t the world
    # => R = Rc.T and t = -Rc.T * tc   (reference class notes)
    R = np.linalg.inv(R_true)
    t = -np.matmul(R, t_true)

    return R, t

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    # compute the mean of the X, Y
    X_mean = np.mean(X, axis = 0)
    Y_mean = np.mean(Y, axis = 0)
    
    # find the centered values of matrices X, Y
    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    # Now we have to minimize the frobenius norm of ||Y_centered - R*X_centered||^2
    # This is equivalent to finding the nearest orthogonal matrix to X_centered.T*Y_centered
    # i.e. ||X_centered.T*Y_centered.T - R||^2
    U, S, Vt = np.linalg.svd(np.matmul(np.transpose(X_centered), Y_centered))

    # calculate the determinant of U*Vt
    det_UVt = np.linalg.det(np.matmul(U, Vt))

    # R is a rotational matrix, so determinant of R has to be positive
    S_modified = np.eye(S.shape[0])
    S_modified[-1, -1] = det_UVt
    
    # compute the rotational matrix
    R = np.matmul(np.transpose(Vt), S_modified)
    R = np.matmul(R, np.transpose(U))

    # compute the translation
    t = Y_mean - np.matmul(R, X_mean)

    return R, t