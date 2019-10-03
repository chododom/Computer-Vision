import cv2
import numpy as np

def get_edge_magnitudes(vertical, horizontal):
    return np.sqrt(np.square(vertical) + np.square(horizontal))
    
    
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))