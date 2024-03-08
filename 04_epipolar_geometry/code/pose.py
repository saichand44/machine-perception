import numpy as np

def pose_candidates_from_E(E):
  transform_candidates = []
  ##Note: each candidate in the above list should be a dictionary with keys "T", "R"

  def _compute_R_t(U, Y, Vt, T ):
    '''
    Compute the R, T pose of the camera for the given orthogonal matrices U, Vt
    (obtained from E), Y, T
    '''
    # initialize an empty dictionary to store the values of R and T
    pose_dict = {}

    R = np.matmul(np.matmul(U, Y), Vt)

    return {'T':T, 'R':R}

  # compute the svd for the E matrix
  U, S, Vt = np.linalg.svd(E)
  
  # define the matrix Y (Rz{pi/2})
  Y = np.array([[0, -1, 0],
                [1,  0, 0],
                [0,  0, 1]])
  
  # Now decompose E to give out the T^ and R matrices
  # get the translation matrix from U (i.e last column of U i.e u3) (and -u3)
  # get the translation matrix from U * Y * Vt (two possiblities for Y --> Rz{pi/2}, Rz{-pi/2})
  T = U[:, -1]

  # consider all the four possible cases of poses R, T
  transform_candidates.append(_compute_R_t(U, np.transpose(Y), Vt, T))
  transform_candidates.append(_compute_R_t(U, Y, Vt, T))
  transform_candidates.append(_compute_R_t(U, np.transpose(Y), Vt, -T))
  transform_candidates.append(_compute_R_t(U, Y, Vt, -T))

  return transform_candidates