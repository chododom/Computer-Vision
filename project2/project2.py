import numpy as np

# Functions from my previous project

def GetGreenPixels(img):
    return img[:,:,1]

# returns array representing the blue channel
def GetBluePixels(img):
    return img[:,:,2]

# returns array representing the red channel
def GetRedPixels(img):
    return img[:,:,0]

# function merges three separate RGB channels into one 3D image
def mergeChannels(r, g, b):
    if len(r) != len(g) or len(g) != len(b):
        raise Exception("Cannot merge channels of different sizes")
        
    height = len(r)
    width = len(r[0])
    img = np.zeros((height, width, 3))
    for i in range(height):
        for j in range(width):
            img[(i, j, 0)] = r[i][j]
            img[(i, j, 1)] = g[i][j]
            img[(i, j, 2)] = b[i][j]
            
    return img   

################################################################################################################################

'''
  Apply filter to a given channel.
'''
def convolve(channel, filter):
    m = len(channel)
    n = len(channel[0])
    x = len(filter)
    y = len(filter[0])
    
    if x > y:
        padding = x // 2
    else:
        padding = y // 2
        
    padded_channel = np.pad(channel, (padding, padding), 'constant')
    filtered_channel = np.zeros((m,n))
    
    for i in range(m):
        for j in range(n):
            # padding can be either x or y (depends which is larger), so adding and subtracting half of size of kernel is done separately
            # NumPy documentation was used to find a fast way of performing this task
            filtered_channel[(i,j)] = np.sum(np.multiply(filter, padded_channel[i + padding - x // 2 : i + padding + x // 2 + 1, j + padding - y // 2 : j + padding + y // 2 + 1]))
                
    return filtered_channel

"""
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (x, y)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that I can finish grading 
   before the heat death of the universe. 
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
"""
def my_imfilter(image, filter):
    
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1
    
    m, n, rgb = image.shape
    
    filtered_img = np.zeros(image.shape)
    
    red = GetRedPixels(image)
    green = GetGreenPixels(image)
    blue = GetBluePixels(image)

    red_filter = convolve(red, filter)
    green_filter = convolve(green, filter)
    blue_filter = convolve(blue, filter)
                
    filtered_img = mergeChannels(red_filter, green_filter, blue_filter)
    return filtered_img

"""
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
"""
def create_hybrid_image(image1, image2, filter):
    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]
    
    low_frequency1 = my_imfilter(image1, filter)
    low_frequency2 = my_imfilter(image2, filter)
    high_frequency2 = np.clip(np.subtract(image2,low_frequency2), 0, 1)
    hybrid_image = np.clip(np.add(low_frequency1, high_frequency2), 0, 1)

    return low_frequency1, high_frequency2, hybrid_image
