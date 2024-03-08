import numpy as np

def est_homography(X, Y):
    """ 
    Calculates the homography H of two planes such that Y ~ H*X
    If you want to use this function for hw5, you need to figure out 
    what X and Y should be. 
    Input:
        X: 4x2 matrix of (x,y) coordinates 
        Y: 4x2 matrix of (x,y) coordinates
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X
        
    """
    
    def _get_equations(goal_pt, logo_pt):
        '''
        This function gives out two linear equations for a point in terms of H.

        1. Consider (x1,y1) be the points in soccer video that has to be mapped
        tp (x2, y2) in the penn logo.

        2. For a homography matrix transformation H, 
                (x2 y2 1).T ~ H*(x1 y1 1).T

            =>  l*(x2 y2 1).T = H*(x1 y1 1).T , where l is the scaling factor
        
        3. Consider the matrix H = [[h00 h01 h02]
                                    [h10 h11 h12]
                                    [h20 h21 h22]]

        4. Solving the above and eliminating l, we get
           x2 = (h00*x1 + h01*y1 + h02)/(h20*x1 + h21*y1 + h22)
           y2 = (h10*x1 + h11*y1 + h12)/(h20*x1 + h21*y1 + h22)
        
        => -h00*x1 -h01*y1 -h02 + h20*x1*x2 + h21*y1*x2 + h22*x2 = 0  --> [1]
           -h10*x1 -h11*y1 -h12 + h20*x1*y2 + h21*y1*y2 + h22*y2 = 0  --> [2]
        
        => (ax ay).T * h = 0
           where   ax = (-x1 -y1 -1 0 0 0 x1*x2 y1*x2 x2)
                   ay = (0 0 0 -x1 -y1 -1 x1*y2 y1*y2 y2)   
                   h = (h00 h01 h02 h10 h11 h12 h20 h21 h22).T
        
        5. After concatenating the matrix A, with corresponding ax and ay for 
           every point, we get a 8x9 matrix. Now use SVD to solve A*H = 0
        
        6. U, S, V =  np.linalg.svd(A) and the last column of the matrix V gives
           out the homography matrix transformation H

        '''

        # get the x1, y1 coorinates of the point in the soccer video
        x1, y1 = goal_pt[0], goal_pt[1]

        # get the x2, y2 coorinates of the point in the penn logo
        x2, y2 = logo_pt[0], logo_pt[1]

        # get the two equation matrix for the given set of points
        ax = np.array([-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2])
        ay = np.array([0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2])

        return np.array([ax, ay])
    
    # initilize the matrix containing corresponding equations (size = 8x9)
    A = np.zeros((8,9))

    # iterate over the four points & populate ax, ay matrices in A in column wise
    for i in range(len(X)):
        ax, ay = _get_equations(X[i], Y[i])
        A[2*i] = ax
        A[2*i+1] = ay

    # use SVD to find the homography matrix transformation
    U, S, V = np.linalg.svd(A)

    # last column of V gives homography matrix transformation and reshape to 3x3
    H = np.reshape(V[-1], (3,3))

    # normalize the homography matrix transformation using H[2][2]
    H = H/H[2][2]

    return H
