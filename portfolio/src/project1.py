#project1.py

import numpy as np
import matplotlib.pyplot as plt
import cv2


# given a filename, function returns array containing data aquired from an image in PPM format
def loadppm(filename):
    file = open(filename, "r")
    
    case = 0
    x = 0
    y = 0
    rgb = 0
    
    while True:
        line = file.readline()
        if line == "":
            break
        arr = line.split()
        # ignore comment lines
        if arr[0].startswith("#"):
            continue
        elif case == 0:
            if line != "P3\n":
                raise Exception("Incorrect PPM file header")
        elif case == 1:
            width, height = int(arr[0]), int(arr[1])
            img = np.zeros((height, width, 3), dtype="uint8")
        elif case == 2:
            max_val = arr[0]
        else:
            # load actual image pixels
            for i in range(len(arr)):
                img[(x, y, rgb)] = arr[i]
                rgb += 1
                if rgb == 3:
                    rgb = 0
                    y += 1
                if y == width:
                    y = 0
                    x += 1
        case += 1
    file.close()
    return img;


# returns array representing the green channel
def GetGreenPixels(img):
    return img[:,:,1]

# returns array representing the blue channel
def GetBluePixels(img):
    return img[:,:,2]

# returns array representing the red channel
def GetRedPixels(img):
    return img[:,:,0]

# converts RGB image to greyscale by averaging all 3 channel values
def naiveGreyscale(img):
    red = GetRedPixels(img)
    green = GetGreenPixels(img)
    blue = GetBluePixels(img)
    
    x = len(red)
    y = len(red[0,:])
    greyArr = np.zeros((x, y))
    
    for i in range(x):
        for j in range(y):
            greyArr[i][j] = (int(red[i][j]) + int(green[i][j]) + int(blue[i][j])) // 3

    return greyArr

# converts RGB image to greyscale by contributing channels in ratio R:G:B = 30:59:11
def advancedGreyscale(img, r, g, b):
    red = GetRedPixels(img)
    green = GetGreenPixels(img)
    blue = GetBluePixels(img)
    
    x = len(red)
    y = len(red[0,:])
    greyArr = np.zeros((x, y))
    
    for i in range(x):
        for j in range(y):
            greyArr[i][j] = (int(red[i][j])*r + int(green[i][j])*g + int(blue[i][j])*b)

    return greyArr


# greyscale values get separated along the value of 128 to either be black or white
def thresholding(img):
    red = GetRedPixels(img)
    green = GetGreenPixels(img)
    blue = GetBluePixels(img)
    
    x = len(red)
    y = len(red[0,:])
    thresholdArr = np.zeros((x, y))
    
    for i in range(x):
        for j in range(y):
            value = (int(red[i][j]) + int(green[i][j]) + int(blue[i][j])) // 3
            if value < 128:
                thresholdArr[i][j] = 0
            else:
                thresholdArr[i][j] = 255

    return thresholdArr


# normalizes values of greyscale image in order to spread them out more evenly across all pixels
def normalize(greyArr):
    x = len(greyArr[:,0])
    y = len(greyArr[0,:])
    cdfArr = np.zeros(256)
    cdfValArr = np.zeros(256)
    cdfCnt = 0
    cdfMin = -1
    equalizedArr = np.zeros((x, y))
    
    for i in range(x):
        for j in range(y):
            cdfArr[int(greyArr[i][j])] += 1
    
    first = True
    for i in range(256):
        if cdfArr[i] != 0:
            if first:
                cdfMin = cdfArr[i]
            cdfValArr[i] = cdfArr[i] + cdfCnt
            cdfCnt += cdfArr[i]
    
    for i in range(x):
        for j in range(y):
            equalizedArr[i][j] = round(((cdfValArr[int(greyArr[i][j])] - cdfMin) / (x * y - cdfMin)) * 255)
    
    return equalizedArr


# function to equally scale down the values of all pictures to a certain max value
def scaleDown(img, top):
    red = GetRedPixels(img)
    green = GetGreenPixels(img)
    blue = GetBluePixels(img)
    
    maxVal = 0
    it = np.nditer(red, flags=['f_index'])
    while not it.finished:
        if it[0] > maxVal:
            maxVal = it[0]
        it.iternext()
        
    it = np.nditer(green, flags=['f_index'])
    while not it.finished:
        if it[0] > maxVal:
            maxVal = it[0]
        it.iternext()
        
    it = np.nditer(blue, flags=['f_index'])
    while not it.finished:
        if it[0] > maxVal:
            maxVal = it[0]
        it.iternext()
    
    ratio = maxVal / top
    
    for i in range(len(img)):
        for j in range(len(img[0])):
            for k in range(len(img[(i, j)])):
                img[(i, j, k)] = img[(i, j, k)] // ratio
    
    return img


# function merges three separate RGB channels into one 3D image
def mergeChannels(r, g, b):
    if len(r) != len(g) or len(g) != len(b):
        raise Exception("Cannot merge channels of different sizes")
        
    height = len(r)
    width = len(r[0])
    img = np.zeros((height, width, 3), dtype="uint8")
    for i in range(height):
        for j in range(width):
            img[(i, j, 0)] = r[i][j]
            img[(i, j, 1)] = g[i][j]
            img[(i, j, 2)] = b[i][j]
            
    scaleDown(img, 255)
    return img    

if __name__ == "__main__":
  # testing done in jupyter notebook
  pass
