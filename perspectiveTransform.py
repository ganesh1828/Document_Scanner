from transform import four_point_transform
import numpy as np
import argparse
import cv2

# parsing required arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', help = 'path to image file')
ap.add_argument('-c', '--cords', help = 'comma separated points list')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
pts = np.array(eval(args['cords']))

warped = four_point_transform(image, pts)

cv2.imshow('Original Image', image)
cv2.imshow('Warped Image', warped)
cv2.waitKey(0)

print('All Done')
