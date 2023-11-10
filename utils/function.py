import cv2
from matplotlib import pyplot as plt
import numpy as np

def get_noise_map(img):
    blur = cv2.bilateralFilter(img,9,75,75)
    noise_map = img-blur
    return noise_map