import cv2
import numpy as np

kernal = np.ones((5, 5), "uint8")


def f_dark_red_mask(hsv_img, img):
    dark_red_lower = np.array([140, 20, 60], np.uint8)
    dark_red_upper = np.array([180, 255, 255], np.uint8)
    dark_red_mask = cv2.inRange(hsv_img, dark_red_lower, dark_red_upper)
    dark_red_mask = cv2.dilate(dark_red_mask, kernal)
    return dark_red_mask


def f_orange_mask(hsv_img, img):
    orange_lower = np.array([11, 80, 50], np.uint8)
    orange_upper = np.array([21, 255, 255], np.uint8)
    orange_mask = cv2.inRange(hsv_img, orange_lower, orange_upper)
    orange_mask = cv2.dilate(orange_mask, kernal)
    return orange_mask


def f_yellow_mask(hsv_img, img):
    yellow_lower = np.array([21, 110, 45], np.uint8)
    yellow_upper = np.array([29, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsv_img, yellow_lower, yellow_upper)
    yellow_mask = cv2.dilate(yellow_mask, kernal)
    return yellow_mask


def f_green_mask(hsv_img, img):
    green_lower = np.array([30, 70, 20], np.uint8)
    green_upper = np.array([90, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsv_img, green_lower, green_upper)
    green_mask = cv2.dilate(green_mask, kernal)
    return green_mask


def f_black_mask(hsv_img, img):
    black_lower = np.array([0, 0, 0], np.uint8)
    black_upper = np.array([255, 255, 95], np.uint8)
    black_mask = cv2.inRange(hsv_img, black_lower, black_upper)
    black_mask = cv2.dilate(black_mask, kernal)
    return black_mask


def f_red_mask(hsv_img, img):
    red_lower = np.array([0, 130, 50], np.uint8)
    red_upper = np.array([20, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsv_img, red_lower, red_upper)
    red_mask = cv2.dilate(red_mask, kernal)
    return red_mask


def white(img):
    white_gummies = 0
    img = cv2.resize(img, (0, 0), fx=0.21, fy=0.21, interpolation=4)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imgGry = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

    mask_orange = f_orange_mask(imgHSV, img)
    mask_yellow = f_yellow_mask(imgHSV, img)
    mask_green = f_green_mask(imgHSV, img)
    mask_black = f_black_mask(imgHSV, img)
    mask_dark = f_dark_red_mask(imgHSV, img)
    mask_red = f_red_mask(imgHSV, img)

    mask_orange = 255 - mask_orange
    mask_yellow = 255 - mask_yellow
    mask_green = 255 - mask_green
    mask_black = 255 - mask_black
    mask_dark = 255 - mask_dark
    mask_red = 255 - mask_red

    edges = cv2.Canny(imgGry, 40, 211)
    res_orange = cv2.bitwise_and(edges, edges, mask=mask_orange)
    res_yellow = cv2.bitwise_and(res_orange, res_orange, mask=mask_yellow)
    res_green = cv2.bitwise_and(res_yellow, res_yellow, mask=mask_green)
    res_black = cv2.bitwise_and(res_green, res_green, mask=mask_black)
    res_dark = cv2.bitwise_and(res_black, res_black, mask=mask_dark)
    res_red = cv2.bitwise_and(res_dark, res_dark, mask=mask_red)

    kernel = np.ones((25, 25), np.uint8)
    img_dilation = cv2.dilate(res_red, kernel, iterations=1)
    img_erode = cv2.erode(img_dilation, kernel, iterations=1)
    opening = cv2.morphologyEx(img_erode, cv2.MORPH_OPEN, kernel=(5, 5))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel=(7, 7))

    img_erode = cv2.blur(closing, (3, 3))

    ret, labels = cv2.connectedComponents(img_erode)
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    imgGrey = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(imgGrey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if 200 < area < 2600:
            x, y, w, h = cv2.boundingRect(contour)
            if 20 <= w < 79 and 20 <= h < 79:
                white_gummies += 1
        if area > 3500:
            white_gummies += 2

    return white_gummies
