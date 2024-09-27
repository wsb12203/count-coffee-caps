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
# sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[1])
# Convert GRAY instances to BGR
im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

# Draw circle around detected seeds
c = 0
for cnt in cnts:
    CENTER = (x+w//2, y+h//2)
    x, y, w, h = cv2.boundingRect(cnt)
    # cv2.circle(img, (x+w//2, y+h//2), max(w, h)//2, (0, 0, 255), 3)
    
    cv2.ellipse(img, (x+w//2, y+h//2), (max(w, h)//2,min(w, h)//2), 0, 0, 360, (0, 0, 255), 3)
    # cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
    c += 1
    
    
    TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
    TEXT_SCALE = 1
    TEXT_THICKNESS = 2
    TEXT = str(c)
    text_size, _ = cv2.getTextSize(TEXT, TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
    text_origin = (CENTER[0] - text_size[0] // 2 - 4, CENTER[1] + text_size[1] // 2)
    cv2.putText(img, TEXT, CENTER, TEXT_FACE, TEXT_SCALE, (127,255,127), TEXT_THICKNESS, cv2.LINE_AA)

cv2.imwrite(output_file, img)

print(len(cnts))
plt.imshow(img)
plt.show()
