import cv2

# Set to an image you wish to draw a grid on.
IMAGE = ""
# File path of where to save the image
OUTPUT = ""

"""
This file is used to test the grid system on a given image. 

"""

img = cv2.imread(IMAGE)

img_height, img_width = img.shape[:2]

# Number of grid cells you want to split the image into.
S = 16

# How big each cell is (w,h) of the cell
S_W = img_width / S
S_H = img_height / S

print("Cell width: {} height: {}".format(S_W, S_H))

for cell in range(0, S):
    print(cell)

    img = cv2.line(img, (int(S_W * cell), int(0)), (int(S_W * cell), img_height), (0, 0, 0), 2)
    img = cv2.line(img, (0, int(S_H * cell)), (img_width, int(S_H * cell)), (0, 0, 0), 2)

x = int(0.8375 * img_width)
y = int(0.63888889 * img_height)
width = int(0.20155644 * img_width / 2)
height = int(0.41331989 * img_height / 2)

# Write image to disk
cv2.imwrite(OUTPUT, img)

# Display Image
cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)
cv2.imshow("Display Window", img)
cv2.waitKey(0)
cv2.waitKey(0)
