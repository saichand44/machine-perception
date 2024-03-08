import numpy as np

def least_squares_estimation(X1, X2):
  """
  """
  def _get_equations(calibrated_pt_x1, calibrated_pt_x2):
    '''
    This function gives a linear equation in terms of the entries of E.

    1. Consider the equation for the essential matrix, x2.T * E * x1 = 0 --> [1]

    2. Consider the matrix E = [[e00 e01 e02]
                                [e10 e11 e12]
                                [e20 e21 e22]]

                          x2 = [x2 y2 z2] and x1 = [x1 y1 z1]

    3. Solving the linear equation (1) gives us
       a.x = 0 where
       a = [x1x2 y1x2 z1x2 x1y2 y1y2 z1y2 x1z2 y1z2 z1z2]
       x = (e00 e01 e02 e10 e11 e12 e20 e21 e22).T

    4. After concatenating the matrix A using "a" from above equation using a set 
       of N points, we get a Nx9 matrix. Now use SVD to solve Ax = 0
 
    5. Finally ensure the rank of E matrix is 2 by equating the last term of the
       singular matrix to zero. (This comes from the fact that all the different
       epipolar lines pass through a single point i.e. epipolar point => [1] should
       satisfy for all different points in image2 for e2 and image1 for e1 and hence
       we are dealing with the null space which means rank(E) must be 2)
    '''
    
    # get the coordinates from the calibrated points
    x1, y1, z1 = calibrated_pt_x1
    x2, y2, z2 = calibrated_pt_x2

    # get the equation a from the point 3
    equ = np.array([x1*x2, y1*x2, z1*x2, x1*y2, y1*y2, z1*y2, x1*z2, y1*z2, z1*z2])

    return equ

  # get the number of points
  number_of_pts = len(X1)

  # initilize the matrix containing corresponding equations (size = len(X1) x 9)
  A = np.zeros((number_of_pts,9))

  # iterate over the points & populate "a" matrices in A in row wise manner
  for i in range(number_of_pts):
      a  = _get_equations(X1[i], X2[i])
      A[i] = a

  # use SVD to find the E matrix
  U, S, Vt = np.linalg.svd(A)

  # last column of the Vt matrix gives the E matrix
  E = np.reshape(Vt[-1], (3, 3))      # reshape the column vector to a 3x3 matrix

  # decompose the matrix E using svd
  Ue, Se, Vet = np.linalg.svd(E)

  # re-compute E to impose the rank constraint
  Se_new = np.diag(np.array([1, 1, 0]))
  E = np.matmul(np.matmul(Ue, Se_new), Vet)
  
  return E
