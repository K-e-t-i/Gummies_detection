import cv2
import numpy as np


def contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.3, tileGridSize=(3, 3))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final


def detect_all_circles(img):
    all_circles = 0

    gray = cv2.cvtColor(contrast(img), cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=30)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for i in circles:
            all_circles += 1

    return all_circles

