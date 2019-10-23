from PIL import Image
import pytesseract
import cv2
import os
import matplotlib.pyplot as plt

# function to convert an image of text into an actual string equal to the content of the image
def img2text(filename, file_ext, blur=False, language='eng'):
    # laod image in grayscale
    img = cv2.imread(filename + file_ext, 0)
    
    if not blur:
        # threshold the image and save
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    else:
        # blur image with Gaussian filter
        img = cv2.GaussianBlur(img, (9, 9), 0)
    
    cv2.imwrite(filename + '_gray' + file_ext, img)
    
    # load the image as a PIL Image and apply pytesseract's OCR
    text = pytesseract.image_to_string(Image.open(filename + '_gray' + file_ext), lang=language)
    
    plt.imshow(img, cmap='gray')
    return text