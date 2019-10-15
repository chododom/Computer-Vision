import cv2
import numpy as np

# Harris detection function to assign R-values to pixels which are considered to be edges
def HarrisDetector(img,k = 0.04):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # kernels
    vertical_sobel = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal_sobel = np.transpose(vertical_sobel)
    gaussian = cv2.getGaussianKernel(ksize=5, sigma=3)
    
    Ix = cv2.filter2D(grey, cv2.CV_16S, vertical_sobel)
    Iy = cv2.filter2D(grey, cv2.CV_16S, horizontal_sobel)
    
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy
    
    g_Ixx = cv2.filter2D(Ixx, cv2.CV_16S, gaussian)
    g_Iyy = cv2.filter2D(Iyy, cv2.CV_16S, gaussian)
    g_Ixy = cv2.filter2D(Ixy, cv2.CV_16S, gaussian)
    
    height = grey.shape[0]
    width = grey.shape[1]
    
    r_vals = np.zeros((height, width))
    for x in range(height):
        for y in range(width):
            A = [ [ g_Ixx[x][y], g_Ixy[x][y] ], [ g_Ixy[x][y], g_Iyy[x][y] ] ]
            r_vals[x][y] = np.linalg.det(A) - k * (np.trace(A) ** 2)         
    
    return r_vals


def SuppressNonMax(Rvals, numPts):
    '''
    Args:
    
    -   Rvals: A numpy array of shape (m,n,1), containing Harris response values
    -   numPts: the number of responses to return

    Returns:

     x: A numpy array of shape (N,) containing x-coordinates of interest points
     y: A numpy array of shape (N,) containing y-coordinates of interest points
     confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
   '''
    pass 


