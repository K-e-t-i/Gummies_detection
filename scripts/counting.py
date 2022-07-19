import cv2
import numpy as np
from scripts.processing import *
from scripts.detection import *
from scripts.color_masks import *
from scripts.detect_white import white


def counting_gummies(img):
    img = cv2.resize(img, (0, 0), fx=0.19, fy=0.19, interpolation=4)
    gray = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

    edges = cv2.Canny(gray, 45, 222)

    kernel = np.ones((6, 6), np.uint8)
    img_dilation = cv2.dilate(edges, kernel, iterations=1)
    img_erode = cv2.erode(img_dilation, kernel, iterations=1)
    img_erode = cv2.blur(img_erode, (6, 6))

    ret, labels = cv2.connectedComponents(img_erode)
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0

    cv2.imshow('p', labeled_img)
    cv2.waitKey(0)

    return ret - 1


def present_result(image):
    cnt_gummies = white(image)
    img = cv2.resize(image, (0, 0), fx=0.2, fy=0.2, interpolation=4)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    sum_all_circles = detect_all_circles(img)

    dred_c = detection_gummies(img, f_dark_red_mask(hsv_img, img))[1]
    dred_g = detection_gummies(img, f_dark_red_mask(hsv_img, img))[0]
    red_c = detection_gummies(img, f_red_mask(hsv_img, img))[1]
    red_g = detection_gummies(img, f_red_mask(hsv_img, img))[0]
    orange_c = detection_gummies(img, f_orange_mask(hsv_img, img))[1]
    orange_g = detection_gummies(img, f_orange_mask(hsv_img, img))[0]
    yellow_c = detection_gummies(img, f_yellow_mask(hsv_img, img))[1]
    yellow_g = detection_gummies(img, f_yellow_mask(hsv_img, img))[0]
    yellow_s = detection_snakes(img, f_yellow_mask(hsv_img, img))
    green_c = detection_gummies(img, f_green_mask(hsv_img, img))[1]
    green_g = detection_gummies(img, f_green_mask(hsv_img, img))[0]
    green_s = detection_snakes(img, f_green_mask(hsv_img, img))
    black_s = detection_snakes(img, f_black_mask(hsv_img, img))

    sum_color_circles = red_c + dred_c + orange_c + green_c + yellow_c

    white_c = sum_all_circles - sum_color_circles
    white_g = cnt_gummies - white_c

    if white_g < 0:
        white_g = 0

    result = [red_g, dred_g, orange_g, green_g, yellow_g, white_g, red_c, dred_c, orange_c, green_c, yellow_c, white_c,
              green_s, black_s, yellow_s]

    return result
