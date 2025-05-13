import numpy as np
from skimage.morphology import erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, black_tophat, skeletonize, convex_hull_image, thin
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from skimage.filters.rank import entropy, enhance_contrast_percentile
from PIL import Image
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
import math
from skimage import data, filters
import cv2
from matplotlib import pyplot as plt

def get_edges(height, width, i, j, n):
    return max(0, i - n), min(height, i + n + 1), max(0, j - n), min(width , j + n + 1)

def phansalskar_more_sabale_method(img, n = 5, k = 0.25, r = 0.5, p = 2, q = 10):
    n = int(n / 2)
    a = img.copy()
    b = img.copy()
    a = a / 255
    b = b / 255

    height = len(a)
    width = len(a[0])

    for i in range(height):
        for j in range(width):
            i_st, i_nd, j_st, j_nd = get_edges(height, width, i, j, n)

            mu = b[i_st:i_nd, j_st:j_nd].mean()
            sigma = b[i_st:i_nd, j_st:j_nd].std()

            threshold = mu * (1 + p * np.exp(-q * mu) + k * (sigma / r - 1))
            a[i][j] = (255 if a[i][j] <= threshold else 0)

    
    return a.astype(img.dtype)

def my_segmentation(img, img_mask, seuil):
    mask = img_mask.astype(np.uint8) * 255
    return phansalskar_more_sabale_method(img, n = 13, q = 15, p = 0.5, k = 0.15, r = 0.5) & mask
    # img_out = (img_mask & (img < seuil))
    # return img_out

n_range = np.arange(1, 25, 2)
q_range = np.arange(1, 50, 4)
p_range = np.arange(0.1, 1, 0.1)
k_range = np.arange(0.05, 0.25, 0.01)
def choose_parameters(img, img_mask):
    best_ACCU = 0
    best_RECALL = 0
    best_img_out = None
    best_n = 0
    best_q = 0
    best_p = 0
    best_k = 0

    for n in n_range:
        print("n = {}/{}".format(n, n_range[-1]))
        for q in q_range:
            for p in p_range:
                for k in k_range:
                    img_out = phansalskar_more_sabale_method(img, n=n, q=q, p=p, k=k, r=0.5)
                    ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out, img_GT)
                    if ACCU > best_ACCU:
                        print('Best parameters so far: n =', n, ', q =', q, ', p =', p, ', k =', k)
                        print('Accuracy =', ACCU, ', Recall =', RECALL)
                        best_ACCU = ACCU
                        best_RECALL = RECALL
                        best_img_out = img_out
                        best_n = n
                        best_q = q
                        best_p = p
                        best_k = k

    return best_img_out, (best_n, best_q, best_p, best_k)

def evaluate(img_out, img_GT):
    GT_skel = skeletonize(img_GT) # On reduit le support de l'evaluation...
    img_out_skel = skeletonize(img_out) # ...aux pixels des squelettes
    TP = np.sum(img_out_skel & img_GT) # Vrais positifs
    FP = np.sum(img_out_skel & ~img_GT) # Faux positifs
    FN = np.sum(GT_skel & ~img_out) # Faux negatifs

    ACCU = TP / (TP + FP) # Precision
    RECALL = TP / (TP + FN) # Rappel
    return ACCU, RECALL, img_out_skel, GT_skel

#Ouvrir l'image originale en niveau de gris
img =  np.asarray(Image.open('./images_IOSTAR/star01_OSC.jpg')).astype(np.uint8)
print(img.shape)

nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
#On ne considere que les pixels dans le disque inscrit 
img_mask = (np.ones(img.shape)).astype(np.bool_)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
img_mask[invalid_pixels] = 0

img_out = my_segmentation(img,img_mask,80)

#Ouvrir l'image Verite Terrain en booleen
img_GT =  np.asarray(Image.open('./images_IOSTAR/GT_01.png')).astype(np.uint32)
print(choose_parameters(img, img_mask))

ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out, img_GT)
print('Accuracy =', ACCU,', Recall =', RECALL)

plt.subplot(231)
plt.imshow(img,cmap = 'gray')
plt.title('Image Originale')
plt.subplot(232)
plt.imshow(img_out)
plt.title('Segmentation')
plt.subplot(233)
plt.imshow(img_out_skel)
plt.title('Segmentation squelette')
plt.subplot(235)
plt.imshow(img_GT)
plt.title('Verite Terrain')
plt.subplot(236)
plt.imshow(GT_skel)
plt.title('Verite Terrain Squelette')
plt.show()