import cv2
import matplotlib.pyplot as plt
import numpy as np

def translation(img):
    shift_x = 50
    shift_y = 50
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated_img = cv2.warpAffine(img, M, (cols, rows))
    plt.imshow(translated_img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def rotation(img):
    height, width = img.shape[:2]
    angle = 45
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 0.5)
    rotated_img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_CONSTANT)

    plt.imshow(rotated_img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

img = cv2.imread('newyork.jpeg')
assert img is not None, "file could not be read"
rotation(img)