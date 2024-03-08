import numpy as np
import matplotlib.pyplot as plt

def plot_flow(image, flow_image, confidence, threshmin=10):
    """
    Plot a flow field of one frame of the data.
    
    Inputs:
        - image: grayscale image - shape: (H, W)
        - flow_image: optical flow - shape: (H, W, 2)
        - confidence: confidence of the flow estimates - shape: (H, W)
        - threshmin: threshold for confidence (optional) - scalar
    """
    
    # Useful function: np.meshgrid()
    # Hint: Use plt.imshow(<your image>, cmap='gray') to display the image in grayscale
    # Hint: Use plt.quiver(..., color='red') to plot the flow field on top of the image in a visible manner
    
    # get the x and y dimensions
    y_dim, x_dim = image.shape[0], image.shape[1]

    # generate the meshgrid
    x = np.linspace(0, x_dim, num=x_dim, endpoint=False)
    y = np.linspace(0, y_dim, num=y_dim, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # get the gradients for the arrow direction
    U = flow_image[:, :, 0]
    V = flow_image[:, :, 1]
    
    # filter the values
    # use the pre-existing code from epipole.py to filter out the values using
    # confidence and threshold and choose at random among the permissible values
    good_idx = np.flatnonzero(confidence>threshmin)
    permuted_indices = np.random.RandomState(seed=10).permutation(good_idx)
    valid_idx=permuted_indices[:3000]
    X = np.ravel(X)[valid_idx]
    Y = np.ravel(Y)[valid_idx]
    U = np.ravel(U)[valid_idx]
    V = np.ravel(V)[valid_idx]

    # plot the original image
    plt.imshow(image, cmap='gray', alpha=0.5)

    # create an optical flow plot
    plt.quiver(X, Y, U, V, scale=15, width=0.003, color='red', alpha=0.5, angles='xy', headwidth=10)

    # plot title and plot show are already included in main!!

    # this function has no return value
    return





    

