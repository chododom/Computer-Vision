import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_interest_points(image, feature_width):
    confidences, scales, orientations = None, None, None

    Responses = HarrisDetector( image )

    x, y, distances = SuppressNonMax(np.where(Responses>0.2*Responses.max(),Responses,0),5000)

    return x,y,None,None,None


def SobelKernels():
    Kx = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    return Kx,Ky

def ImageDerivative( image, Kx, Ky ):

    Gx = cv2.filter2D( image, cv2.CV_16S, Kx )
    Gy = cv2.filter2D( image, cv2.CV_16S, Ky )

    return Gx,Gy

def ConvolveWithGauss( image, GK, GK_T ):

    output = cv2.filter2D( image, cv2.CV_16S, GK )
    return cv2.filter2D( output, cv2.CV_16S, GK_T )

def HarrisDetector(img,k = 0.04):
    grayscale = img.copy()

    Kx, Ky = SobelKernels()
    Ix, Iy = ImageDerivative( grayscale, Kx, Ky ) 

    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy 

    GaussKernel = cv2.getGaussianKernel(ksize=5, sigma=1)
    GaussKernel_T = GaussKernel.transpose()

    Ixx = ConvolveWithGauss( Ixx, GaussKernel, GaussKernel_T ).astype( np.int32 )
    Ixy = ConvolveWithGauss( Ixy, GaussKernel, GaussKernel_T ).astype( np.int32 )
    Iyy = ConvolveWithGauss( Iyy, GaussKernel, GaussKernel_T ).astype( np.int32 )
 
    #formula
    R2 = ( Ixx * Iyy - Ixy * Ixy - k * ( Ixx + Iyy ) ** 2 ).astype( np.int32 )
  
    return R2 

def SuppressNonMax(Rvals, numPts):
    indices = np.nonzero( Rvals )
    vals = Rvals[ indices ]

    #connect the arrays into one
    indices = np.stack( ( indices[0], indices[1], vals ), axis=-1 )

    #sort by the R value
    i_sorted = indices[indices[:,2].argsort()[::-1]]

    ranges = np.copy( i_sorted )

    indices = i_sorted[:,0:2]

    ranges[0] = i_sorted[0]
    ranges[0][2] = np.iinfo( np.int64 ).max

    for i in range( 1, i_sorted.shape[0] ):
        ranges[i][2] = np.linalg.norm(indices[0:i]-indices[i],axis=1).min()
   
    r_sorted = ranges[ranges[:,2].argsort()[::-1]]
    return r_sorted[:numPts,1],r_sorted[:numPts,0],r_sorted[:numPts,2]



