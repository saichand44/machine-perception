from lse import least_squares_estimation
import numpy as np
from tqdm import tqdm # for test

def ransac_estimator(X1, X2, num_iterations=60000): #60000
    sample_size = 8

    eps = 10**-4

    best_num_inliers = -1
    best_inliers = None
    best_E = None

    def _get_distance(pt_x1, pt_x2, E):
        '''
        This function calculates the distance between the point x2 and the epipolar
        line corresponding to the point x2 in image 2 i.e. E*x1
        '''
        # assuming the points are already calibrated

        # compute the epipolar line coefficients (E*x1)
        line_coeff = np.matmul(E, np.reshape(pt_x1, (3,1)))    

        # compute the distance of the point x2 from the epipolar line
        e3 = np.array([0, 0, 1])
        e3_skew = np.array([[     0, -e3[2],  e3[1]],
                            [ e3[2],      0, -e3[0]],
                            [-e3[1],  e3[1],      0]])

        numerator = np.matmul(np.reshape(pt_x2, (1,3)), line_coeff)
        denominator = np.matmul(e3_skew, line_coeff)
        distance = np.sqrt(np.ravel(numerator[0])**2 / np.linalg.norm(denominator, 'fro')**2) 
        
        return distance 

    def _count_inliers(test_pts_x1, test_pts_x2, E, eps = 10**-4):
        '''
        This function counts the total number of inliers whose residual is less 
        than a certain acceptable threshold.
        The residual is calculated as given in .pdf d(x2,epi(E,x1))**2 + d(x1,epi(E,x2))**2
        '''
        inliers_indices = []
        
        for i, (test_pt_x1, test_pt_x2) in enumerate(zip(test_pts_x1, test_pts_x2)):
            # get the distance of point x2 for epipolar line in image 2
            d21 = _get_distance(test_pt_x1, test_pt_x2, E)

            # get the distance of point x1 for epipolar line in image 1
            d12 = _get_distance(test_pt_x2, test_pt_x1, np.transpose(E))

            # compute the resultant of the residual
            residual = d21**2 + d12**2

            if residual <= eps:
                inliers_indices.append(i)
        
        return np.array([inliers_indices])


    for i in tqdm(range(num_iterations), desc="Processing items", unit="item"):
        # permuted_indices = np.random.permutation(np.arange(X1.shape[0]))
        permuted_indices = np.random.RandomState(seed=(i*10)).permutation(np.arange(X1.shape[0]))
        sample_indices = permuted_indices[:sample_size]
        test_indices = permuted_indices[sample_size:]
        
        # get a sample of 8 point correspondences for the estimation of E
        sample_X1 = X1[sample_indices]
        sample_X2 = X2[sample_indices]

        # compute the matrix E
        E = least_squares_estimation(sample_X1, sample_X2)

        # get the inliers
        test_X1 = X1[test_indices]
        test_X2 = X2[test_indices]
        inliers_indices = test_indices[[idx for idx in _count_inliers(test_X1, test_X2, E, eps)]]

        # append the current inliers indices to the sample indices as they are also inliers
        inliers_indices = np.append(sample_indices, inliers_indices)
        inliers = inliers_indices

        if inliers.shape[0] > best_num_inliers:
            best_num_inliers = inliers.shape[0]
            best_E = E
            # this line is redudant as the shape is already taken care of in line 78
            # best_inliers = np.reshape(np.ravel(inliers), (len(inliers), 2))
            best_inliers = inliers_indices


    return best_E, best_inliers