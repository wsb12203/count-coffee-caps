import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

user_name = "user"
main_path = fr"C:\Users\{user_name}\Downloads"
input_file = fr"{main_path}\input_image.jpg"
output_file = fr"{main_path}\output_image.jpg"

img = cv2.imread(input_file)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (0, 37, 127), (16, 255,255))
im = cv2.bitwise_and(img,img, mask=mask)

H, W = im.shape[:2]
# Remove noise
im = cv2.medianBlur(im, 3)
im = cv2.GaussianBlur(mask, (5, 5), 21)
im = cv2.adaptiveThreshold(im, 25, cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY, 31, 2)

# Fill area with black to find seeds
cv2.floodFill(im, np.zeros((H+2, W+2), np.uint8), (0, 0), 0)
im = cv2.dilate(im, np.ones((9, 9)))
im = cv2.erode(im, np.ones((7, 7)))
plt.imshow(im)
plt.show()
# Find seeds
cnts, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Convert GRAY instances to BGR
im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

# Draw circle around detected seeds
c = 0
for cnt in cnts:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.circle(img, (x+w//2, y+h//2), max(w, h)//2, (c, 150, 255-c), 3)
    c += 5
cv2.imwrite(output_file, img)

print(len(cnts))
plt.imshow(img)
plt.show()