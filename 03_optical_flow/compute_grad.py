import numpy as np
from scipy.ndimage import convolve1d

# Use the following kernels for computing the image gradients
# G: a Gaussian smoothing kernel, used to smooth across the axes orthogonal to the gradient AFTER the gradient is computed
# H: the derivative kernel, that outputs a measure of the difference between neighboring pixels (derivative of a Gaussian)
KERNEL_G = np.array([0.015625, 0.093750, 0.234375, 0.312500, 0.234375, 0.093750, 0.015625])
KERNEL_H = np.array([0.03125, 0.12500, 0.15625, 0, -0.15625, -0.1250, -0.03125])

def compute_Ix(imgs):
    """
    Compute the gradient of the images along the x-dimension.
    
    WARNING: The first coordinate of the gradient is actually the y-coordinate,
    which means here you're computing the gradient along axis=1, NOT 0!
    
    Inputs:
        - imgs: image volume, axis semantics: first = y, second = x, third = t - shape: (H, W, N)
    Outputs:
        - Ix: image gradient along the x-dimension - shape: (H, W, N)
    """
    

    Ix = None

    # cycle through the images
    for frame in range(imgs.shape[2]):
        # initialize an empty array to populate the gradients in x direction
        if Ix is None:
            Ix = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2]))

        # get a particular frame in the sequence
        img = imgs[:, :, frame]

        #=======================================================================
        # CONVOLUTION USING GRADIENT OF G IN X DIRECTION
        #=======================================================================
        # cycle through the rows in img to perform guassian operations i.e x dir
        Ix[:, :, frame] = convolve1d(img, weights=KERNEL_H, axis=1)
        
        # cycle through the columns in Ix[:,:,n] to perform guassian operations i.e x dir
        Ix[:, :, frame] = convolve1d(Ix[:, :, frame], weights=KERNEL_G, axis=0)

    #===========================================================================
    # CONVOLUTION USING Gt
    #===========================================================================
    # cycle through the time frames in  Ix[:,:,n] to perform guassian operations
    for row_index in range(img.shape[0]):
        for col_index in range(img.shape[1]):
            temporal_data = Ix[row_index, col_index, :]
            Ix[row_index, col_index, :] = convolve1d(temporal_data, weights=KERNEL_G)
    
    return Ix

def compute_Iy(imgs):
    """
    Compute the gradient of the images along the y-dimension.
    
    WARNING: The first coordinate of the gradient is actually the y-coordinate,
    which means here you're computing the gradient along axis=0, NOT 1!
    
    Inputs:
        - imgs: image volume, axis semantics: first = y, second = x, third = t - shape: (H, W, N)
    Outputs:
        - Iy: image gradient along the y-dimension - shape: (H, W, N)
    """

    Iy = None

    # cycle through the images
    for frame in range(imgs.shape[2]):
        # initialize an empty array to populate the gradients in y direction
        if Iy is None:
            Iy = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2]))

        # get a particular frame in the sequence
        img = imgs[:, :, frame]

        #=======================================================================
        # CONVOLUTION USING GRADIENT OF G IN Y DIRECTION
        #=======================================================================
        # cycle through the columns in img to perform guassian operations i.e y dir
        Iy[:, :, frame] = convolve1d(img, weights=KERNEL_H, axis=0)

        # cycle through the rows in Ix[:,:,n] to perform guassian operations i.e y dir
        Iy[:, :, frame] = convolve1d(Iy[:, :, frame], weights=KERNEL_G, axis=1)
        
    #===========================================================================
    # CONVOLUTION USING G
    #===========================================================================
    # cycle through the time frames in  Ix[:,:,n] to perform guassian operations
    for row_index in range(img.shape[0]):
        for col_index in range(img.shape[1]):
            temporal_data = Iy[row_index, col_index, :]
            Iy[row_index, col_index, :] = convolve1d(temporal_data, weights=KERNEL_G)
    
    return Iy

def compute_It(imgs):
    """
    Compute the gradient of the images along the x-dimension.
    
    WARNING: The first coordinate of the gradient is actually the y-coordinate,
    which means here you're computing the gradient along axis=0, NOT 1!
    
    Inputs:
        - imgs: image volume, axis semantics: first = y, second = x, third = t - shape: (H, W, N)
    Outputs:
        - It: temporal image gradient - shape: (H, W, N)
    """
    
    
    It = None

    #===========================================================================
    # CONVOLUTION USING GRADIENT OF G IN TIME DIRECTION
    #===========================================================================

    # initialize an empty array to populate the gradients in t
    if It is None:
        It = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2]))
    
    # temporal derivative
    for row_index in range(imgs.shape[0]):
        for col_index in range(imgs.shape[1]):
            temporal_data = imgs[row_index, col_index, :]
            It[row_index, col_index, :] = convolve1d(temporal_data, weights=KERNEL_H)

    #===========================================================================
    # CONVOLUTION USING G IN X, Y DIRECTION
    #===========================================================================
    for frame in range(imgs.shape[2]):
        # smoothing in Y direction
        It[:, :, frame] = convolve1d(It[:, :, frame], weights=KERNEL_G, axis=0)
        # smoothing in X direction
        It[:, :, frame] = convolve1d(It[:, :, frame], weights=KERNEL_G, axis=1) 
          
    
    return It
