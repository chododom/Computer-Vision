import cv2
import math
import numpy as np

def detect_edges(img):
    vertical_sobel = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal_sobel = np.asarray([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gaussian = cv2.getGaussianKernel(ksize=10, sigma=5)
    blur = cv2.filter2D(img, -1, gaussian)
    v_dst = cv2.filter2D(img, -1, vertical_sobel)
    h_dst = cv2.filter2D(img, -1, horizontal_sobel)
    return get_edge_magnitudes(v_dst, h_dst)

def get_edge_magnitudes(vertical, horizontal):
    return np.sqrt(np.square(vertical) + np.square(horizontal))
    
def calculate_orientation(vertical, horizontal):
    return np.arctan2(horizontal, vertical)
    
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# greyscale values get separated along the value of 128 to either be black or white
def thresholding(img):
    x = len(img)
    y = len(img[0,:])
    thresholdArr = np.zeros((x, y))
    
    for i in range(x):
        for j in range(y):
            val = img[i][j][0]
            if val < 128:
                thresholdArr[i][j] = 0
            else:
                thresholdArr[i][j] = 255

    return thresholdArr

def myHoughLines(image, rho_res, theta_res, threshold):
    # apply Gaussian blur and Sobel edge detector
    detected_edges = detect_edges(image)
    
    edges = thresholding(detected_edges)
    
    theta_vals = np.linspace(0, math.pi, 8) #8 vals between 0 and pi
    lines = {}
    
    for x in range(len(image)):
        for y in range(len(image[0])):
            for t in theta_vals:
                p = x * math.cos(t) + y * math.sin(t)
                p = round(p, 1)
                if (p, t) in lines.keys():
                      lines[(p, t)] += 1
                else:
                      lines[(p, t)] = 0
    
    # shape according to cv2's documentation of HoughLines function
    strong_lines = np.ndarray((len(lines), 1, 2), dtype="uint8")
    index = 0
    for key in lines:
        if lines[key] > threshold:
            strong_lines[index][0] = [key[0], key[1]]
            index += 1
            # print("Line with rho: " + str(key[0]) + " and theta: " + str(key[1]) + " has " + str(lines[key]) + " votes")
            
    return strong_lines
    
def non_max_suppression(magnitudes, orientations):
    padded_magnitudes = np.pad(magnitudes, (1, 1), 'constant')
    
    x, y, channels = magnitudes.shape
    for i in range(1, x):
        for j in range(1, y):
            if round(orientations[i - 1][j - 1][0]) == 0:
                if padded_magnitudes[i][j - 1][0] > padded_magnitudes[i][j][0] or padded_magnitudes[i][j + 1][0] > padded_magnitudes[i][j][0]:
                    magnitudes[i - 1][j - 1][0] = 0
            elif round(orientations[i - 1][j - 1][0]) == 90:
                if padded_magnitudes[i - 1][j][0] > padded_magnitudes[i][j][0] or padded_magnitudes[i + 1][j][0] > padded_magnitudes[i][j][0]:
                    magnitudes[i - 1][j - 1][0] = 0
            elif round(orientations[i - 1][j - 1][0]) == 45:
                if padded_magnitudes[i + 1][j - 1][0] > padded_magnitudes[i][j][0] or padded_magnitudes[i - 1][j + 1][0] > padded_magnitudes[i][j][0]:
                    magnitudes[i - 1][j - 1][0] = 0
            elif round(orientations[i - 1][j - 1][0]) == 135:
                if padded_magnitudes[i - 1][j - 1][0] > padded_magnitudes[i][j][0] or padded_magnitudes[i + 1][j + 1][0] > padded_magnitudes[i][j]:
                    magnitudes[i - 1][j - 1][0] = 0
            else:
                continue
                
    padded_magnitudes = np.pad(magnitudes, (1, 1), 'constant')
        
    for i in range(x):
        for j in range(y):
            if magnitudes[i][j][0] < padded_magnitudes[i + 1 - 1][j + 1 - 1][0] or magnitudes[i][j][0] < padded_magnitudes[i + 1 - 1][j + 1 + 1][0] or magnitudes[i][j][0] < padded_magnitudes[i + 1 + 1][j + 1 + 1][0] or magnitudes[i][j][0] < padded_magnitudes[i + 1 + 1][j + 1 - 1][0] or magnitudes[i][j][0] < padded_magnitudes[i + 1][j + 1 - 1][0] or magnitudes[i][j][0] < padded_magnitudes[i + 1][j + 1 + 1][0] or magnitudes[i][j][0] < padded_magnitudes[i + 1 - 1][j + 1][0] or magnitudes[i][j][0] < padded_magnitudes[i + 1 + 1][j + 1][0]:
                magnitudes[i][j][0] = 0
    
    return magnitudes
    
    
    
    
    
    
    
    
    
