import numpy as np

def reconstruct3D(transform_candidates, calibrated_1, calibrated_2):
  """This functions selects (T,R) among the 4 candidates transform_candidates
  such that all triangulated points are in front of both cameras.
  """

  best_num_front = -1
  best_candidate = None
  best_lambdas = None
  for candidate in transform_candidates:
    R = candidate['R']
    T = candidate['T']

    lambdas = np.zeros((2, calibrated_1.shape[0]))

    # Cycle through all the points to get lambda1 and lambda2 for each pair
    # Use the equation lambda2*qi = lambda1*R*pi + T
    # rearrange the equation (qi -Rpi) * (lambda1 lambda2).T = T  --> [1]
    # let the matrices be  A * X = B  where A = 3x2 ; B = 3x1 and X = 2x1
    for i, (pt1, pt2) in enumerate(zip(calibrated_1, calibrated_2)):
      pt1 = np.reshape(pt1, (3, 1))
      pt2 = np.reshape(pt2, (3, 1))

      A = np.hstack((pt2, -np.matmul(R, pt1)))

      # solve for lambdas from the equation [1]
      lambdas[:, i] = np.matmul(np.linalg.pinv(A), T)

    num_front = np.sum(np.logical_and(lambdas[0]>0, lambdas[1]>0))

    if num_front > best_num_front:
      best_num_front = num_front
      best_candidate = candidate
      best_lambdas = lambdas
      print("best", num_front, best_lambdas[0].shape)
    else:
      print("not best", num_front)


  P1 = best_lambdas[1].reshape(-1, 1) * calibrated_1
  P2 = best_lambdas[0].reshape(-1, 1) * calibrated_2
  T = best_candidate['T']
  R = best_candidate['R']
  return P1, P2, T, R