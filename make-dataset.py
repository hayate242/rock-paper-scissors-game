import cv2
# from PIL import Image
import numpy as np


def colorRange(hsvImg):
    HSV_MIN = np.array([0, 20, 40])
    HSV_MAX = np.array([20, 220, 255])
    hsv_mask = cv2.inRange(hsvImg, HSV_MIN, HSV_MAX)
    kernel = np.ones((2, 2), np.uint8)
    ksize = 5
    hsv_mask = cv2.medianBlur(hsv_mask, ksize)
    hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_CLOSE, kernel)
    hsv_mask = cv2.dilate(hsv_mask, kernel, iterations=1)
    return hsv_mask


def addGaussianNoise(src):
    row, col, ch = src.shape
    mean = 0
    var = 0.5
    sigma = 50
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    a = 1
    noisy = a*src + (4-a)*gauss
    return noisy


def augmentedImage(img):
    orgH, orgW = img.shape[:2]
    aImg = np.zeros((orgH*2, orgW*2, 3), np.uint8)
    aImg[0:orgH, 0:orgW] = img
    # 90度
    transpose_img = img.transpose(1, 0, 2)
    clockwise = transpose_img[:, ::-1]
    aImg[orgH:orgH*2, 0:orgW] = clockwise
    # -90度
    counter_clockwise = transpose_img[::-1]
    aImg[0:orgH, orgW:orgW*2] = counter_clockwise
    # 180度
    xAxis = cv2.flip(img, 0)
    yAxis = cv2.flip(img, 1)
    xyAxis = cv2.flip(img, -1)
    aImg[orgH:orgH*2, orgW:orgW*2] = xyAxis
    return aImg


if __name__ == "__main__":
    img = cv2.imread('./dataset/rock/IMG_1118.JPG', cv2.IMREAD_COLOR)
    rows, cols, channels = img.shape
    # HSVに変換
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    cv2.imshow('original2', img2)

    # color_range = colorRange(img2)
    # dst = cv2.add(img, color_range)
    # print(dst)
    # img[0:rows, 0:cols] = dst
    # cv2.imshow('exracted hand', img)
    # gaussian_noise = addGaussianNoise(color_range)
    # augmented_img = augmentedImage(img)
    # print(color_range)

    # cv2.namedWindow('drifting!!')
    cv2.imshow('original', img)
    # cv2.imshow('color_range', color_range)
    # cv2.imshow('gaussian_noise!!', gaussian_noise)
    # cv2.imshow('augmented_img!!', augmented_img)
    # cv2.imshow('edged!!', edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
