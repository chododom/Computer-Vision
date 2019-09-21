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
        if arr[0] == "#":
            continue
        elif case == 0:
            if line != "P3\n":
                raise Exception("Incorrect PPM file header")
        elif case == 1:
            width, height = int(arr[0]), int(arr[1])
            img = np.zeros((height, width, 3), dtype=int)
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
        
    return img;


def GetGreenPixels(img):
    '''given a numpy 3d array containing an image, return the green channel'''
    return img[:,:,1]

def GetBluePixels(img):
    '''given a numpy 3d array containing an image, return the blue channel'''
    return img[:,:,2]

def GetRedPixels(img):
    '''given a numpy 3d array containing an image, return the red channel'''
    return img[:,:,0]

def naiveGreyscale(img):
    red = GetRedPixels(img)
    green = GetGreenPixels(img)
    blue = GetBluePixels(img)
    
    x = len(red)
    y = len(red[0,:])
    greyArr = np.zeros((x, y))
    
    for i in range(x):
        for j in range(y):
            greyArr[i][j] = (red[i][j] + green[i][j]+ blue[i][j]) // 3

    return greyArr

def thresholdArr(img):
    pass



if __name__ == "__main__":
  #put any command-line testing code you want here.
  pass
