import cv2
import numpy as np

kernal = np.ones((5, 5), "uint8")


def f_dark_red_mask(hsv_img, img):
    # Set range for dark red color and
    # define mask
    dark_red_lower = np.array([170, 180, 60], np.uint8)
    dark_red_upper = np.array([180, 255, 133], np.uint8)
    dark_red_mask = cv2.inRange(hsv_img, dark_red_lower, dark_red_upper)

    dark_red_mask = cv2.dilate(dark_red_mask, kernal)
    res_dark_red = cv2.bitwise_and(img, img, mask=dark_red_mask)

    return dark_red_mask


def f_red_mask(hsv_img, img):
    # Set range for red color and
    # define mask
    red_lower = np.array([1, 190, 105], np.uint8)
    red_upper = np.array([7, 255, 180], np.uint8)
    red_mask = cv2.inRange(hsv_img, red_lower, red_upper)

    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(img, img, mask=red_mask)

    return red_mask


def f_orange_mask(hsv_img, img):
    # Set range for orange color and
    # define mask
    orange_lower = np.array([12, 155, 50], np.uint8)
    orange_upper = np.array([18, 255, 240], np.uint8)
    orange_mask = cv2.inRange(hsv_img, orange_lower, orange_upper)

    orange_mask = cv2.dilate(orange_mask, kernal)
    res_orange = cv2.bitwise_and(img, img, mask=orange_mask)

    return orange_mask


def f_yellow_mask(hsv_img, img):
    # Set range for yellow color and
    # define mask
    yellow_lower = np.array([21, 150, 45], np.uint8)
    yellow_upper = np.array([29, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsv_img, yellow_lower, yellow_upper)

    yellow_mask = cv2.dilate(yellow_mask, kernal)
    res_yellow = cv2.bitwise_and(img, img, mask=yellow_mask)

    return yellow_mask


def f_green_mask(hsv_img, img):
    # Set range for green color and
    # define mask
    green_lower = np.array([34, 120, 20], np.uint8)
    green_upper = np.array([80, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsv_img, green_lower, green_upper)

    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(img, img, mask=green_mask)

    return green_mask


def f_black_mask(hsv_img, img):
    # Set range for black color and
    # define mask
    black_lower = np.array([0, 0, 0], np.uint8)
    black_upper = np.array([255, 255, 45], np.uint8)
    black_mask = cv2.inRange(hsv_img, black_lower, black_upper)

    black_mask = cv2.dilate(black_mask, kernal)
    res_black = cv2.bitwise_and(img, img, mask=black_mask)

    return black_mask


def f_white_mask(hsv_img, img):
    # Set range for white color and
    # define mask
    white_lower = np.array([255, 255, 255], np.uint8)
    white_upper = np.array([255, 255, 255], np.uint8)
    white_mask = cv2.inRange(hsv_img, white_lower, white_upper)

    white_mask = cv2.dilate(white_mask, kernal)
    res_white = cv2.bitwise_and(img, img, mask=white_mask)

    return white_mask
