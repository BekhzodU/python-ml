import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from IPython import display
from skimage import data, img_as_float
from skimage.feature import corner_harris, corner_subpix, corner_peaks

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


img = cv2.imread('photo-sample.png')
assert img is not None, "file could not be read"
harrisCorners(img)