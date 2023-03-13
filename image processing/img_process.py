import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.color import rgb2gray
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
from skimage.feature import blob_dog, blob_log, blob_doh

# rgb with numpy histogram manual
def rgbChannels(img):
    nb_bins = 256
    x = np.array(img).transpose(2,0,1)
    counts = list()
    hists = list()
    for i in range(0,3):
        hists.append(np.histogram(x[i], bins=nb_bins, range=[0, 255]))
        counts.append(np.zeros(nb_bins) + hists[i][0])
    
    bins = hists[0][1]
    colors = ('r', 'g', 'b')
    fig = plt.figure()
    for i, el in enumerate(counts):
        plt.bar(bins[:-1], el, color=colors[i], alpha=0.4)

    plt.show()

# rgb histogram with matplotlib
def rgbChannelsMatlotlib(img):
    color = ('blue','red', 'green')
    for i, col in enumerate(color):
        plt.hist(img[:,:,i].ravel(), bins=256, color=col, alpha=0.3+i*0.05)

    plt.show()


#split into 3 channels
def splitRgb(img):
    height, width, channels = img.shape
    if channels >= 3:
        B, G, R = cv2.split(img[:, :, :3])  # Only split the first three channels
    elif channels == 1:  # Grayscale image
        R = G = B = img
    else:
        raise ValueError("Input image must have at least 3 channels")

    zeroMatrix = np.zeros((height, width), dtype="uint8")
    R = cv2.merge([R, zeroMatrix, zeroMatrix])
    G = cv2.merge([zeroMatrix, G, zeroMatrix])
    B = cv2.merge([zeroMatrix, zeroMatrix, B])
    rgb = [B,G,R]
    for i,el in enumerate(rgb):
        plt.subplot(1,3,i+1)
        plt.imshow(el)
        plt.xticks([])
        plt.yticks([])
    plt.show()
    

def contour(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = np.flipud(np.asarray(img))
    data = np.asarray(img)
    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    plt.subplot(1,3,2)
    plt.contour(data, linestyles='solid', colors='black',extent=[0, img.shape[1], 0, img.shape[0]])
    plt.subplot(1,3,3)
    plt.contourf(data,linestyles='solid', cmap='inferno')
    plt.tight_layout()
    plt.show()


def convolution(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_size = 15
    sigma = 5
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel = np.outer(kernel, kernel.transpose())
    blurred = cv2.filter2D(gray, -1, gaussian_kernel)
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    edges = cv2.filter2D(blurred, -1, laplacian_kernel)
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image ')

    plt.subplot(1, 3, 2)
    plt.imshow(blurred, cmap='gray')
    plt.title('Box Blur')

    plt.subplot(1, 3, 3)
    plt.imshow(edges, cmap='gray')
    plt.title('Laplace Edge Detection')
    plt.tight_layout()
    plt.show()


def harrisCorners(img):
    original = img
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w, _ = img.shape
    img = np.array(grayImg)
    coords = corner_peaks(corner_harris(grayImg), min_distance=15, threshold_rel=0.02)
    _, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
    ax.plot(coords[:, 1], coords[:, 0], color='red', marker='o', linestyle='None', markersize=3)
    ax.axis((0, w, h, 0))
    plt.xticks([])
    plt.yticks([])
    plt.show()


def multi_dil(im, num, element):
    for i in range(num):
        im = dilation(im, element)
    return im

def multi_ero(im, num, element):
    for i in range(num):
        im = erosion(im, element)
    return im

def morphologicalPreprocess(img):
    binary = rgb2gray(img)<0.15
    element = np.array([[0,0,0,0,0,0,0],
                    [0,0,1,1,1,0,0],
                    [0,1,1,1,1,1,0],
                    [0,1,1,1,1,1,0],
                    [0,1,1,1,1,1,0],
                    [0,0,1,1,1,0,0],
                    [0,0,0,0,0,0,0]])
    multi_eroded = multi_ero(binary, 2, element)
    opened = opening(multi_eroded, element)
    multi_diluted = multi_dil(opened, 2, element)
    area_morphed = area_opening(area_closing(multi_diluted, 1000), 1000)
    return area_morphed
    
class obj:
    def __init__(self, title, x, y, blobList, color):
        self.title = title
        self.x = x
        self.y = y
        self.blobList = blobList
        self.color = color

def blobDetection(img):
    cleanedImg = morphologicalPreprocess(img)
    blobsL = blob_log(cleanedImg, min_sigma=15)
    blobsD = blob_dog(cleanedImg, min_sigma=5, threshold=0.1)
    blobsH = blob_doh(cleanedImg, min_sigma=35)

    params = list()
    params.append(obj('Original Image', 0, 0, list(), ''))
    params.append(obj('Blobs with Laplacian of Gaussian', 0, 1, blobsL, 'yellow'))
    params.append(obj('Blobs with Difference of Gaussian', 1, 0, blobsD, 'green'))
    params.append(obj('Blobs with Determinant of Hessian', 1, 1, blobsH, 'red'))

    fig, ax = plt.subplots(2,2,figsize=(7,7))

    for el in params:
        ax[el.x,el.y].imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        ax[el.x,el.y].set_title(el.title)
        ax[el.x,el.y].set_xticks([])
        ax[el.x,el.y].set_yticks([])
        for blob in el.blobList:
            y, x, area = blob
            ax[el.x,el.y].add_patch(plt.Circle((x, y), area*np.sqrt(2), color=el.color, fill=False))
    
    plt.tight_layout()
    plt.show()


img = cv2.imread('flash.png')
assert img is not None, "file could not be read"
blobDetection(img)

