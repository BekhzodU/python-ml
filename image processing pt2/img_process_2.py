import cv2
import matplotlib.pyplot as plt
import numpy as np

def translation(img):
    shift_x = 50
    shift_y = 50
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated_img = cv2.warpAffine(img, M, (cols, rows))
    image = cv2.cvtColor(translated_img, cv2.COLOR_RGB2BGR)

    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def rotation(img):
    height, width = img.shape[:2]
    angle = 45
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 0.5)
    rotated_img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_CONSTANT)
    image = cv2.cvtColor(rotated_img, cv2.COLOR_RGB2BGR)

    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def affineTransformation(img):
    height, width = img.shape[:2]
    src_points = np.float32([[0, 0], [width-1, 0], [0, height-1]])
    dst_points = np.float32([[0, 0], [width * 0.6, 0], [width * 0.4, height - 1]])
    M = cv2.getAffineTransform(src_points, dst_points)
    transformed_img = cv2.warpAffine(img, M, (width, height))
    image = cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR)

    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def mirroring(img):
    horizontal_mirror = cv2.flip(img, 1)
    image = cv2.cvtColor(horizontal_mirror, cv2.COLOR_RGB2BGR)
    
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def projectiveTransform(img):
    src_pts = np.float32([[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]])
    dst_pts = np.float32([[0, 0], [img.shape[1], 0], [300, img.shape[0]], [img.shape[1]-300, img.shape[0]]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    projected_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    image = cv2.cvtColor(projected_img, cv2.COLOR_RGB2BGR)

    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.show()

img = cv2.imread('newyork.jpeg')
assert img is not None, "file could not be read"
affineTransformation(img)