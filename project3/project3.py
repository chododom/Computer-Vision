import cv2
import math
import numpy as np

def reduce_channels(img):
    arr = np.ndarray((len(img), len(img[0])), dtype="uint8")
    for i in range(len(img)):
        for j in range(len(img[0])):
            arr[i][j] = img[i][j][0]
    return arr

def detect_edges(img):
    vertical_sobel = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal_sobel = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gaussian = cv2.getGaussianKernel(ksize=10, sigma=5)
    blur = cv2.filter2D(img, -1, gaussian)
    v_dst = cv2.filter2D(img, -1, vertical_sobel)
    h_dst = cv2.filter2D(img, -1, horizontal_sobel)
    return get_edge_magnitudes(v_dst, h_dst)

def get_edge_magnitudes(vertical, horizontal):
    return np.clip(np.hypot(vertical, horizontal), 0, 255).astype(np.uint8)
    
def calculate_orientation(vertical, horizontal):
    return np.arctan2(horizontal, vertical)

# greyscale values get separated along the value of 128 to either be black or white
def thresholding(img):
    x = len(img)
    y = len(img[0,:])
    thresholdArr = np.zeros((x, y))
    
    for i in range(x):
        for j in range(y):
            val = img[i][j]
            if val < 128:
                thresholdArr[i][j] = 0
            else:
                thresholdArr[i][j] = 255

    return thresholdArr

def nearest_rho(val, rho_res):
    return int(np.rint(val / rho_res) * rho_res)

# parts of this function inspired by source: https://alyssaq.github.io/2014/understanding-hough-transform/
def myHoughLines(image, rho_res, theta_res, threshold):
    theta_vals = np.arange(0, math.pi, theta_res)   
    l = int(np.ceil(np.hypot(len(image), len(image[0]))))
    rhos = np.arange(-l, l, 2 * l)
    
    acc = np.zeros((2 * l, len(theta_vals)), dtype=np.uint32)
    y_edges, x_edges = np.nonzero(image)  # row and column indexes of edges
    
    for i in range(len(x_edges)):
        x = x_edges[i]
        y = y_edges[i]
        
        for t in range(len(theta_vals)):
            rho = nearest_rho(x * np.cos(theta_vals[t]) + y * np.sin(theta_vals[t]), rho_res)
            acc[rho][t] += 1
    
    cnt = 0
    for i in range(len(acc)):
        for j in range(len(theta_vals)):
            if acc[i][j] > threshold:
                cnt += 1
                
    lines = np.ndarray((cnt, 2))
    index = 0
    for i in range(len(acc)):
        for j in range(len(theta_vals)):
            if acc[i][j] > threshold:
                lines[index] = i, j
                index += 1
                
    print('Strong line count: ' + str(cnt))
    
    # shape according to cv2's documentation of HoughLines function
    strong_lines = np.ndarray((cnt, 1, 2))
                
    for i in range(cnt):
        strong_lines[i][0] = lines[i][0], lines[i][1] * theta_res
   
    return strong_lines

# non-maximum suppression function to choose strongest pixels in local neighbourhoods
def non_max_suppression(magnitudes, orientations):
    padded_magnitudes = np.pad(magnitudes, (1, 1), 'constant')
    
    h, w = magnitudes.shape
    for x in range(1, h + 1):
        for y in range(1, w + 1):
            
            # cmp to east and west
            if round(orientations[x - 1][y - 1]) == 0:
                if padded_magnitudes[x][y + 1] > padded_magnitudes[x][y] or padded_magnitudes[x][y - 1] > padded_magnitudes[x][y]:
                    magnitudes[x - 1][y - 1] = 0
                    
            # cmp to north and south
            elif round(orientations[x - 1][y - 1]) == 90:
                if padded_magnitudes[x - 1][y] > padded_magnitudes[x][y] or padded_magnitudes[x + 1][y] > padded_magnitudes[x][y]:
                    magnitudes[x - 1][y - 1] = 0
                    
            # cmp to north-east and south-west
            elif round(orientations[x - 1][y - 1]) == 45:
                if padded_magnitudes[x - 1][y + 1] > padded_magnitudes[x][y] or padded_magnitudes[x + 1][y - 1] > padded_magnitudes[x][y]:
                    magnitudes[x - 1][y - 1] = 0
                    
            # cmp to north-west and south-east
            elif round(orientations[x - 1][y - 1]) == 135:
                if padded_magnitudes[x - 1][y - 1] > padded_magnitudes[x][y] or padded_magnitudes[x + 1][y + 1] > padded_magnitudes[x][y]:
                    magnitudes[x - 1][y - 1] = 0
                    
            else:
                continue
                 
    return magnitudes    
    
# double thresholding function to separate lines according to their strength
def magnitude_threshold(val, t_h, t_l):
    if val < t_l:
        val = 0
    elif val > t_h:
        val = 255
    else:
        return val
    return val

# edge tracking function to thin out edge lines
def track_edges(img, t_h, t_l):
    # get x and y indices for pixels with weak edges (t_l <= val <= t_h)
    #weak_edges_x, weak_edges_y = np.nonzero(np.where(np.logical_and(t_l <= img, img <= t_h), img, 0))
    
    padded_img = np.pad(img, (1, 1), 'constant')
    #weak_edges_x += 1
    #weak_edges_y += 1
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            # if at least one neighbour is strong (val > t_h), keep current pixel, otherwise 0
            if t_l <= img[x][y] and img[x][y] <= t_h:
                if padded_img[x - 1][y - 1] <= t_h and padded_img[x - 1][y] <= t_h and padded_img[x - 1][y + 1] <= t_h and padded_img[x][y - 1] <= t_h and padded_img[x][y + 1] <= t_h and padded_img[x + 1][y - 1] <= t_h and padded_img[x + 1][y] <= t_h and padded_img[x + 1][y + 1] <= t_h:
                    img[x - 1][y - 1] = 0  
            
    return img

# wrapper for all functions included in the Canny edge detection tool
def myCanny(img, t_h, t_l):
    # kernels
    vertical_sobel = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal_sobel = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gaussian = cv2.getGaussianKernel(ksize=5, sigma=1)

    # blur image with Gaussian filter
    img = cv2.filter2D(img, cv2.CV_16S, gaussian)
    
    v_dst = cv2.filter2D(img, cv2.CV_16S, vertical_sobel)
    h_dst = cv2.filter2D(img, cv2.CV_16S, horizontal_sobel)

    magnitudes = np.hypot(v_dst, h_dst)
    orientations = calculate_orientation(v_dst, h_dst)
    
    n_m_sup = non_max_suppression(magnitudes, orientations)
    
    threshold_magnitudes = np.vectorize(magnitude_threshold)(n_m_sup, t_h, t_l)
    
    return track_edges(threshold_magnitudes, t_h, t_l)
    
    
    
    
