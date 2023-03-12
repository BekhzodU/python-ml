import cv2
import matplotlib.pyplot as plt
import numpy as np

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

img = cv2.imread('photo_sample.png')
assert img is not None, "file could not be read"
rgbChannels(img)