'''
main.py
航太所碩一 P46091204 蔡承穎  Copyright (C) 2020
1. Background substraction using single gaussian distribution
2. Find 7 feature points
   , then use LK flow to track points
3. Use Aruco Marker to realize AR
4. Use PCA to do dimension reduction
   , and calculate the reconstruction error
'''

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDockWidget, QListWidget
from PyQt5.QtGui import *
from Ui_hw2 import Ui_MainWindow

import math
import time
from numpy.linalg import norm
from scipy.stats import multivariate_normal as gaussian

import cv2
import cv2 as cv
import numpy as np
import matplotlib
from matplotlib import pyplot
import glob
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from matplotlib.pyplot import suptitle
from matplotlib.pyplot import savefig


class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    # __init__:解構函式，也就是類被建立後就會預先載入的專案。
    # 馬上執行，這個方法可以用來對你的物件做一些你希望的初始化。
    def __init__(self):
        # 這裡需要過載一下mywindow，同時也包含了QtWidgets.QMainWindow的預載入項。
        super(mywindow, self).__init__()
        self.setupUi(self)


def button1_clicked():

    cap = cv2.VideoCapture('Hw2\\Q1_Image\\bgsub.mp4')

    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    col, row = frame_gray.shape

    count = 0
    gray_sum = np.zeros((col, row))
    mean = np.zeros((col, row))
    variance = np.zeros((col, row))

    while (count < 50):

        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gray_sum += frame_gray

        count += 1

    cap.release()

    mean = gray_sum / 50

    cap2 = cv2.VideoCapture('Hw2\\Q1_Image\\bgsub.mp4')
    ret, frame = cap2.read()
    count = 0

    while (count < 50):

        ret, frame = cap2.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        variance += (frame_gray - mean)**2

        count += 1

    variance /= 50
    # print(variance)

    variance = np.where(variance < 25, 25, variance)
    # print(np.max(variance))

    a = np.uint8([255])  # white (foreground)
    b = np.uint8([0])  # black (background)

    while (1):
        result = np.zeros((col, row))

        ret, frame = cap2.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not ret:
            break

        bool1 = (frame_gray > mean + 5 * (variance ** 0.5))
        bool2 = (frame_gray > mean - 5 * (variance ** 0.5))

        result = np.bitwise_xor(bool1, bool2)
        result = np.where(result == False, a, b)

        cv.imshow("Original frame", frame)
        cv.imshow("Background substraction", result)

        k = cv2.waitKey(33) & 0xff
        if k == 27:
            break

    cap2.release()
    cv2.destroyAllWindows()


def button2_clicked():

    global points
    points = np.zeros((7, 1, 2))    # 共有七個特徵點
    params = cv2.SimpleBlobDetector_Params()

    # 以下設定是個小坑

    # Area (Size)
    params.filterByArea = True
    params.minArea = 40
    params.maxArea = 54
    # Circularity (if perfect circle, then it is 1.0)
    params.filterByCircularity = True
    params.minCircularity = 0.8
    # Convexity (if no gap, then it is 1.0)
    params.filterByConvexity = True
    params.minConvexity = 0.9

    detector = cv2.SimpleBlobDetector_create(params)

    capature = cv2.VideoCapture('Hw2\\Q2_Image\\opticalFlow.mp4')
    ret, frame = capature.read()
    keypoints = detector.detect(frame)

    # Total : 7 blue circules
    for i in range(0, len(keypoints)):
        x, y = np.int(keypoints[i].pt[0]), np.int(keypoints[i].pt[1])
        points[i, 0, 0] = keypoints[i].pt[0]
        points[i, 0, 1] = keypoints[i].pt[1]
        # Dram the cross markers
        img = cv2.rectangle(frame, (x-5, y-5), (x+5, y+5),
                            color=(0, 0, 255), thickness=1)
        img = cv2.line(frame, (x, y-5), (x, y+5),
                       color=(0, 0, 255), thickness=1)
        img = cv2.line(frame, (x-5, y), (x+5, y),
                       color=(0, 0, 255), thickness=1)

    cv2.imshow('img', img)
    cv2.waitKey(0)


def button3_clicked():
    capature = cv2.VideoCapture('Hw2\\Q2_Image\\opticalFlow.mp4')
    ret, old_frame = capature.read()

    # Mat mask = Mat::zeros(old_frame.size(), old_frame.type());
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    p0 = np.float32(points)
    mask = np.zeros_like(old_frame)
    '''
    vid_writer = cv.VideoWriter("HW02_02.avi", cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (round(
        capature.get(cv.CAP_PROP_FRAME_WIDTH)), round(capature.get(cv.CAP_PROP_FRAME_HEIGHT))))
    '''
    while(1):
        ret, frame = capature.read()
        if(ret != True):
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, winSize=(15, 15), maxLevel=4)   # defalut : winSize(21, 21) maxLevel = 3    運動太大要改層數
        good_new = p1
        good_old = p0

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), (0, 0, 255), 2)
            frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        # vid_writer.write(img.astype(np.uint8))
        k = cv2.waitKey(33) & 0xff
        if k == 27:
            break
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv2.destroyAllWindows()
    capature.release()
    # vid_writer.release()


def button4_clicked():
    im_src = cv2.imread('Hw2\\Q3_Image\\rl.jpg')
    cap = cv2.VideoCapture('Hw2\\Q3_Image\\test4perspective.mp4')
    '''
    outputFile = "HW02_03.avi"

    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (round(
        2*cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    print("Storing it as :", outputFile)
    '''
    while (1):
        hasFrame, frame = cap.read()

        if not hasFrame:
            break

        dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
        parameters = cv.aruco.DetectorParameters_create()
        markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(
            frame, dictionary, parameters=parameters)

        # Find markers 25, 33
        # 因為影片有幾張是偵測不到marker的，故當有一個偵測不到我就進下一個loop

        index = np.squeeze(np.where(markerIds == 25))
        if (np.all(index == 0)):
            continue
        refPt1 = np.squeeze(markerCorners[index[0]])[1]

        index = np.squeeze(np.where(markerIds == 33))
        if (np.all(index == 0)):
            continue
        refPt2 = np.squeeze(markerCorners[index[0]])[2]

        distance = np.linalg.norm(refPt1-refPt2)

        scalingFac = 0.02
        pts_dst = [
            [refPt1[0] - round(scalingFac*distance), refPt1[1] - round(scalingFac*distance)]]
        pts_dst = pts_dst + \
            [[refPt2[0] + round(scalingFac*distance),
              refPt2[1] - round(scalingFac*distance)]]

        index = np.squeeze(np.where(markerIds == 30))
        refPt3 = np.squeeze(markerCorners[index[0]])[0]
        pts_dst = pts_dst + \
            [[refPt3[0] + round(scalingFac*distance),
              refPt3[1] + round(scalingFac*distance)]]

        index = np.squeeze(np.where(markerIds == 23))
        if (np.all(index == 0)):
            continue
        refPt4 = np.squeeze(markerCorners[index[0]])[0]

        # pts_dst are four points
        pts_dst = pts_dst + \
            [[refPt4[0] - round(scalingFac*distance),
              refPt4[1] + round(scalingFac*distance)]]

        pts_src = [[0, 0], [im_src.shape[1], 0], [
            im_src.shape[1], im_src.shape[0]], [0, im_src.shape[0]]]

        pts_src_m = np.asarray(pts_src)
        pts_dst_m = np.asarray(pts_dst)

        # Find Homography Matrix
        h, status = cv.findHomography(pts_src_m, pts_dst_m)

        warped_image = cv.warpPerspective(
            im_src, h, (frame.shape[1], frame.shape[0]))

        mask = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8)
        cv.fillConvexPoly(mask, np.int32(
            [pts_dst_m]), (255, 255, 255), cv.LINE_AA)

        element = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        mask = cv.erode(mask, element, iterations=3)

        warped_image = warped_image.astype(float)
        mask3 = np.zeros_like(warped_image)
        for i in range(0, 3):
            mask3[:, :, i] = mask/255

        warped_image_masked = cv.multiply(warped_image, mask3)
        frame_masked = cv.multiply(frame.astype(float), 1-mask3)
        im_out = cv.add(warped_image_masked, frame_masked)

        concatenatedOutput = cv.hconcat([frame.astype(float), im_out])
        cv.imshow("Use Aruco markers to realize AR",
                  concatenatedOutput.astype(np.uint8))
        # vid_writer.write(concatenatedOutput.astype(np.uint8))

        k = cv2.waitKey(33) & 0xff
        if k == 27:
            break

    cv.destroyAllWindows()
    cap.release()
    # vid_writer.release()


def button5_clicked():
    global array_of_img
    array_of_img = []

    for filename in glob.glob("Hw2\\Q4_Image\\*.jpg"):
        img = cv2.imread(filename)
        array_of_img.append(img)

    def pca(image):
        cov_mat = image - np.mean(image, axis=1)
        eig_val, eig_vec = np.linalg.eigh(
            np.cov(cov_mat))  # 得到eigenvalue and eigenvector
        p = np.size(eig_vec, axis=1)
        idx = np.argsort(eig_val)
        idx = idx[::-1]  # 順序相反可得最大eigen value
        eig_vec = eig_vec[:, idx]
        eig_val = eig_val[idx]
        principal_components = 70
        if principal_components < p or principal_components > 0:
            eig_vec = eig_vec[:, range(principal_components)]
        score = np.dot(eig_vec.T, cov_mat)
        reconstruction = np.dot(eig_vec, score) + np.mean(image, axis=1).T
        recon_img_mat = np.uint8(np.absolute(reconstruction))
        return recon_img_mat

    global array_of_img_recon
    array_of_img_recon = []
    fig, ax = plt.subplots(4, 17)

    for i in range(34):
        data = array_of_img[i]
        r = data[:, :, 0]
        g = data[:, :, 1]
        b = data[:, :, 2]
        a_r_recon, a_g_recon, a_b_recon = pca(r), pca(g), pca(b)
        recon_color_img = np.dstack((a_r_recon, a_g_recon, a_b_recon))
        recon_color_img = Image.fromarray(recon_color_img)
        array_of_img_recon.append(recon_color_img)
        if i == 8:
            ax[0][i].text(-3, -20, "Original")
            ax[1][i].text(-3, -20, "Reconstrction")
            ax[2][i].text(-3, -20, "Original")
            ax[3][i].text(-3, -20, "Reconstrction")
        if i < 17:
            ax[0][i].imshow(array_of_img[i])
            ax[0][i].set_xticks([])
            ax[0][i].set_yticks([])
            ax[1][i].imshow(recon_color_img)
            ax[1][i].set_xticks([])
            ax[1][i].set_yticks([])
        else:
            ax[2][i-17].imshow(array_of_img[i])
            ax[2][i-17].set_xticks([])
            ax[2][i-17].set_yticks([])
            ax[3][i-17].imshow(recon_color_img)
            ax[3][i-17].set_xticks([])
            ax[3][i-17].set_yticks([])

    plt.show()


def button6_clicked():
    gray_img = []
    gray_img_recon = []

    for filename in glob.glob("Hw2\\Q4_Image\\*.jpg"):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        gray_img.append(img)

    err_arr = []

    for i in range(34):
        img = cv2.cvtColor(np.asarray(
            array_of_img_recon[i]), cv2.COLOR_BGR2GRAY)
        gray_img_recon.append(img)
        err = (gray_img_recon[i] - gray_img[i]).sum()
        err_arr.append(err)

    print(err_arr)


if __name__ == '__main__':  # 如果整個程式是主程式
    app = QtWidgets.QApplication(sys.argv)  # 初始化GUI介面
    window = mywindow()  # 呼叫mywindow物件，並給這個物件名字window
    window.setWindowTitle("HW02")
    # 有了例項，就得讓它顯示，show()是QWidget的方法，用於顯示視窗。
    window.show()
    window.pushButton.clicked.connect(button1_clicked)
    window.pushButton_2.clicked.connect(button2_clicked)
    window.pushButton_3.clicked.connect(button3_clicked)
    window.pushButton_4.clicked.connect(button4_clicked)
    window.pushButton_5.clicked.connect(button5_clicked)
    window.pushButton_6.clicked.connect(button6_clicked)
    # 呼叫sys庫的exit退出方法，條件是app.exec_()，也就是整個視窗關閉。
    # 有時候退出程式後，sys.exit(app.exec_())會報錯，改用app.exec_()就沒事
    sys.exit(app.exec_())
