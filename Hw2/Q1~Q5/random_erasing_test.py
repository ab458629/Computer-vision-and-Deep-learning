import PIL
import time
import numpy as np
import tensorflow as tf
from scipy.misc import imsave
import cv2
import random
import math


def random_erasing(img, probability=0.5, sl=0.02, sh=0.4, r1=0.3):

    height = img.shape[0]
    width = img.shape[1]
    channel = img.shape[2]

    area = height * width

    Se = random.uniform(sl, sh) * area
    re = random.uniform(r1, 1 / r1)

    He = int(round(math.sqrt(Se * re)))
    We = int(round(math.sqrt(Se / re)))

    xe = random.randint(0, img.shape[0] - He)
    ye = random.randint(0, img.shape[1] - We)

    if xe < img.shape[1] and ye < img.shape[0]:
        img[xe:xe+He, ye:ye+We,
            :] = np.random.randint(0, 255, size=(He, We, channel)).astype(np.uint8)

    return img


if __name__ == '__main__':
    img = cv2.imread("test.jpg")
    img = random_erasing(img)
    cv2.namedWindow("1", 0)
    cv2.resizeWindow("1", 400, 400)
    cv2.imshow("1", img)
    cv2.waitKey(0)
