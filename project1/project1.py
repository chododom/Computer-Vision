#project1.py
import numpy as np
import matplotlib.pyplot as plt
import cv2



def loadppm(filename):
    '''Given a filename, return a numpy array containing the ppm image
    input: a filename to a valid ascii ppm file 
    output: a properly formatted 3d numpy array containing a separate 2d array 
            for each color
    notes: be sure you test for the correct P3 header and use the dimensions and depth 
            data from the header
            your code should also discard comment lines that begin with #
    '''
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
  # return the numpy 3d array



def GetGreenPixels(img):
    '''given a numpy 3d array containing an image, return the green channel'''
    pass

def GetBluePixels(img):
    '''given a numpy 3d array containing an image, return the blue channel'''
    pass

def GetRedPixels(img):
    '''given a numpy 3d array containing an image, return the red channel'''
    pass


if __name__ == "__main__":
  #put any command-line testing code you want here.
  pass
