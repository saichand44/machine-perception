import numpy as np

def est_Pw(s):
    """
    Estimate the world coordinates of the April tag corners, assuming the world origin
    is at the center of the tag, and that the xy plane is in the plane of the April
    tag with the z axis in the tag's facing direction. See world_setup.jpg for details.
    Input:
        s: side length of the April tag

    Returns:
        Pw: 4x3 numpy array describing the world coordinates of the April tag corners
            in the order of a, b, c, d for row order. See world_setup.jpg for details.

    """

    # With reference to world_setup.jpg, the origin is at the center of the april tag,
    # hence we can extract the coordinates of corners using the origin and side length.

    # initialize the array containing the world coorindates of the corners of april tag
    Pw = np.zeros((4,3))

    # the origin is at the center of the april tag and with x, y, z directions
    # as shown in world_setup.jpg
    # world coordinates (z_{w} = 0 since the cooridnate system is on the april tag)
    Pw[0] = np.array([-s/2, -s/2, 0])  # point a
    Pw[1] = np.array([ s/2, -s/2, 0])  # point b
    Pw[2] = np.array([ s/2,  s/2, 0])  # point c
    Pw[3] = np.array([-s/2,  s/2, 0])  # point d

    return Pw
