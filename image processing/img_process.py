import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython import display

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
    

img = cv2.imread('photo_sample.png')
assert img is not None, "file could not be read"
rgbChannels(img)
splitRgb(img)