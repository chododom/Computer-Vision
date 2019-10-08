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

def nearest_rho(val, rhos):
    id = (np.abs(rhos - val)).argmin()
    return rhos[id]

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
            rho = int(round(x * np.cos(theta_vals[t]) + y * np.sin(theta_vals[t])))
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

    
def non_max_suppression(magnitudes, orientations):
    padded_magnitudes = np.pad(magnitudes, (1, 1), 'constant')
    
    x, y = magnitudes.shape
    for i in range(1, x):
        for j in range(1, y):
            if round(orientations[i - 1][j - 1]) == 0:
                if padded_magnitudes[i][j - 1] > padded_magnitudes[i][j] or padded_magnitudes[i][j + 1] > padded_magnitudes[i][j]:
                    magnitudes[i - 1][j - 1] = 0
            elif round(orientations[i - 1][j - 1]) == 90:
                if padded_magnitudes[i - 1][j] > padded_magnitudes[i][j] or padded_magnitudes[i + 1][j] > padded_magnitudes[i][j]:
                    magnitudes[i - 1][j - 1] = 0
            elif round(orientations[i - 1][j - 1]) == 45:
                if padded_magnitudes[i + 1][j - 1] > padded_magnitudes[i][j] or padded_magnitudes[i - 1][j + 1] > padded_magnitudes[i][j]:
                    magnitudes[i - 1][j - 1] = 0
            elif round(orientations[i - 1][j - 1]) == 135:
                if padded_magnitudes[i - 1][j - 1] > padded_magnitudes[i][j] or padded_magnitudes[i + 1][j + 1] > padded_magnitudes[i][j]:
                    magnitudes[i - 1][j - 1] = 0
            else:
                continue
                 
    return magnitudes
    '''
    for i in range(x):
        for j in range(y):
            if magnitudes[i][j] < padded_magnitudes[i + 1 - 1][j + 1 - 1] or magnitudes[i][j] < padded_magnitudes[i + 1 - 1][j + 1 + 1] or magnitudes[i][j] < padded_magnitudes[i + 1 + 1][j + 1 + 1] or magnitudes[i][j] < padded_magnitudes[i + 1 + 1][j + 1 - 1] or magnitudes[i][j] < padded_magnitudes[i + 1][j + 1 - 1] or magnitudes[i][j] < padded_magnitudes[i + 1][j + 1 + 1] or magnitudes[i][j] < padded_magnitudes[i + 1 - 1][j + 1] or magnitudes[i][j] < padded_magnitudes[i + 1 + 1][j + 1]:
                magnitudes[i][j] = 0
    
    return magnitudes'''
    
    
def magnitude_threshold(val, t_h, t_l):
    if val < t_l:
        val = 0
    elif val > t_h:
        val = 255
    else:
        return val
    return val

def track_edges(img):
    weak_edges = np.where(100 < img < 200)
    print(weak_edges)
    return None

    
    
    
    
    
    
