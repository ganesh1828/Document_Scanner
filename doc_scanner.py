from transform import four_point_transform
from skimage.filters import threshold_local
from stackImages import stackImages
import numpy as np
import argparse
import cv2
import imutils

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required = True, help = 'image path')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
origImg = image.copy()
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

def get_edged(img):
    print('STEP 1: Edge Detection')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    
    return edged

def get_contours(img):
    print('STEP 2: Contour Detection')
    cnts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    # finding largest contour
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            return approx

    print('Unable to detect contours')
    return None

def get_warped(img):
    print('STEP 3: Perspective Transform')
    warped = four_point_transform(img, contours.reshape(4, 2) * ratio)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 7, offset = 7, method = 'gaussian')
    warped = (warped > T).astype('uint8') * 255
    return warped

def get_thres(img):
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
    return thresh

edged = get_edged(image)
contours = get_contours(edged)
if contours is None:
    print('Unable to find document')
    exit(0)
warped = get_warped(origImg)

cv2.drawContours(image, [contours], -1, (0, 255, 0), 3)

outputImg = stackImages(0.3, ([origImg, edged], [image, warped]))
cv2.imshow('Output image', outputImg)
cv2.imwrite('./output/Output3.jpg', warped)
cv2.imwrite('./output/Output_Stacked3.jpg', outputImg)
# cv2.imshow('Original Image', image)
# cv2.imshow('Edged Image', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
