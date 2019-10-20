import cv2
import numpy as np

# applies linearly separable Gaussian filters
def gaussianThatShit(img):
    gaussian = cv2.getGaussianKernel(ksize=5, sigma=3)
    img = cv2.filter2D(img, cv2.CV_16S, gaussian)
    img = cv2.filter2D(img, cv2.CV_16S, gaussian.T)
    return img

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
    
    g_Ixx = gaussianThatShit(Ixx)
    g_Iyy = gaussianThatShit(Iyy)
    g_Ixy = gaussianThatShit(Ixy)
    
    height = grey.shape[0]
    width = grey.shape[1]
    
    r_vals = np.zeros((height, width))
    for x in range(height):
        for y in range(width):
            A = [ [ g_Ixx[x][y], g_Ixy[x][y] ], [ g_Ixy[x][y], g_Iyy[x][y] ] ]
            r_vals[x][y] = np.linalg.det(A) - k * (np.trace(A) ** 2)         
    
    return r_vals


# suppresses local R-values into one maximum
def SuppressNonMax(Rvals, numPts, top_x_percent):
    height = len(Rvals)
    width = len(Rvals[0])
    
    r_vals = np.zeros( height * width, dtype=([('x', int), ('y', int), ('r', float), ('sup_r', float)]) )
    
    index = 0
    max_r = Rvals.max()
    for i in range(height):
        for j in range(width):
            if Rvals[i][j] > (1 - top_x_percent) * max_r:
                r_vals[index] = (i, j, Rvals[i][j], 0)
                index += 1

    r_vals = r_vals[:index]
    
    r_vals = (np.sort(r_vals, order='r'))[::-1]   
    r_vals[0][3] = np.inf
    
    for i in range(1, r_vals.shape[0]):
        min_sup_r = np.inf
        for j in range(0, i):
            distance = np.hypot(r_vals[i][0] - r_vals[j][0], r_vals[i][1] - r_vals[j][1])
            if distance < min_sup_r:
                min_sup_r = distance
        r_vals[i][3] = min_sup_r
    
    r_vals = (np.sort(r_vals, order='sup_r'))[::-1]
    x_vals = np.zeros(numPts)
    y_vals = np.zeros(numPts)
    
    for i in range(numPts):
        x_vals[i] = r_vals[i][0]
        y_vals[i] = r_vals[i][1]
        
    return x_vals, y_vals

############################################################################################################################################

def im2single(im):
    im = im.astype(np.float32) / 255
    return im

def single2im(im):
    im *= 255
    im = im.astype(np.uint8)
    return im

def load_image(path):
    return im2single(cv2.imread(path))[:, :, ::-1]

def save_image(path, im):
    return cv2.imwrite(path, single2im(im.copy())[:, :, ::-1])