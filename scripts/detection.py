import cv2
import numpy as np
from scripts.processing import *


def detection_gummies(img, mask):
    # Creating contour to track orange color
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sum_bears = 0
    sum_circles = 0
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 4)
            # print(w)
            # print(h)
            if 20 <= w < 68 and 20 <= h < 68:
                crop_img = img[y - 10:y + h + 10, x - 10:x + w + 10]

                gray = cv2.cvtColor(contrast(crop_img), cv2.COLOR_BGR2GRAY)
                gray = cv2.blur(gray, (3, 3))
                output = crop_img.copy()
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=52, param2=30, minRadius=10,
                                           maxRadius=30)
                if circles is not None:
                    sum_circles += 1
                else:
                    sum_bears += 1
    all_gummies_color = [sum_bears, sum_circles]
    return all_gummies_color


def detection_snakes(img, mask):
    # Creating contour to track orange color
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    sum_snakes = 0
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= 68 or h >= 68:
                sum_snakes += 1

    return sum_snakes
