import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from PIL import Image
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

    # cv2.imshow("a", a)
    return a.astype(img.dtype)

def mode_method(img, n = 5, f = 0.5):
    n = int(n / 2)
    a = img.copy()

    height = len(a)
    width = len(a[0])

    for i in range(height):
        for j in range(width):
            i_st, i_nd, j_st, j_nd = get_edges(height, width, i, j, n)

            kernel = img[i_st:i_nd, j_st:j_nd]
            mode = kernel.sum() > (len(kernel.flatten()) * f) * 255

            a[i][j] = mode * 255

    # cv2.imshow("b", a)
    return a.astype(img.dtype)

def my_segmentation(img, img_mask, seuil):
    mask = img_mask.astype(np.uint8) * 255
    # cv2.imshow("mask", mask)
    # seg = phansalskar_more_sabale_method(img, n = 11, q = 20, p = 0.5, k = 0.08, r = 0.5) & mask
    # seg = phansalskar_more_sabale_method(img, n = 23, q = 17, p = 0.9, k = 0.2, r = 0.5) & mask
    # seg = mode_method(seg, n = 3, f = 0.2) & mask
    # high_acc = seg.copy()
    seg = phansalskar_more_sabale_method(img, n = 23, q = 17, p = 0.9, k = 0.12, r = 0.5) & mask
    high_rec = seg.copy()
    high_rec = remove_small_objects(high_rec.astype(bool), min_size=10, connectivity=2).astype(np.uint8) * 255
    cv2.imshow("high_rec", high_rec)
    return high_rec

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
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (-10 + nrows / 2)**2)
img_mask[invalid_pixels] = 0

img_out = my_segmentation(img,img_mask,80)

#Ouvrir l'image Verite Terrain en booleen
img_GT =  np.asarray(Image.open('./images_IOSTAR/GT_01.png')).astype(np.uint32)

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