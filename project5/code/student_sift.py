#!/usr/bin/env python3

import numpy as np
import cv2
import sys

def Magnitude( Gx, Gy ):
    ''' @input: Gx - 2d np arr, Gy - 2d np arr
    @output: magnitude of the image, 2d np arr
    '''
    return np.clip( np.hypot( Gx, Gy ), 0, 255 ).astype(np.uint8)

# Gradient Direction
def Grad_Direction( Gx, Gy ):
    ''' @input: Gx - 2d np arr, Gy - 2d np arr
    @output: arctan of Gx and Gy converted to degrees - 2d np arr
    '''
    return np.degrees( np.arctan2( Gy, Gx ) )

def Normalize_Direction( Grad_Dir ):
    ''' @input: Gradient Dir - 2d np arr
    @output: normalized Gradient Direction - 2d np arr
    '''
    #tmp = np.rint( Grad_Dir / 45 ).astype('int16') * 45
    #return np.where( tmp < 0, tmp + 180, tmp )
    return Grad_Dir + 180

def Filtered_Image( image ):
    ''' @input: image - 2d np arr
    @output: Gx - 2d np arr, Gy 2d np arr
    - only to make the code shorter
    '''

    GaussKernel = cv2.getGaussianKernel(ksize=5, sigma=1)

    result = cv2.filter2D(image, -1, GaussKernel)
    result = cv2.filter2D(result, -1, GaussKernel.transpose())

    Kx = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


    Gx = cv2.filter2D( result, cv2.CV_16S, Kx )
    Gy = cv2.filter2D( result, cv2.CV_16S, Ky )

    return Gx, Gy

def get_features(image, x_coord, y_coord, f_w, scales=None):

    assert image.ndim == 2, 'Image must be grayscale'

    discretization = 8

    Gx, Gy = Filtered_Image( image )
    Grad = Magnitude( Gx, Gy )
    Grad_Dir = Grad_Direction( Gx, Gy )
    Norm_Dir = Normalize_Direction( Grad_Dir )

    x_coord = x_coord.astype( np.int32 )
    y_coord = y_coord.astype( np.int32 )
  
    features = np.zeros( ( x_coord.shape[0], ((f_w//4) ** 2) * discretization ) )
    hists = np.zeros( ( (f_w//4)**2, discretization ) )
  
    pad_img = np.pad( Norm_Dir, (f_w//2, f_w//2), 'constant')
  
  
    for i in range( y_coord.shape[0] ):
        patch = pad_img[int(y_coord[i]) : int(y_coord[i] + 2*f_w//2), int(x_coord[i]) : int(x_coord[i] + 2*f_w//2)]
        index = 0
    
        for j in range( f_w // 4 ):
            for k in range( f_w // 4 ):
                square = patch[ j * f_w//4 : j * f_w//4 + f_w // 4, k * f_w//4 : k * f_w//4 + f_w // 4 ].flatten()
                hist, e = np.histogram( square, bins=8, range=(0,360) ) #jeste zvazit
                hists[index] = hist
                index += 1
    
        features[i] = hists.flatten()    
  
    return features
