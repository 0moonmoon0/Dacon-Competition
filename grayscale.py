import cv2
import numpy as np

# Loads image that you want to mask
img = cv2.imread('1.jpg')

# Converts to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Determine threshold input. This process turns pixel value to 0 (black) or 255 (white).
# If pixel value is greater than thresh, assign maxvalue, otherwise 0. 
# cv2.threshold(src, thresh, maxval, type [, dst]) -> retval, dst
mask = cv2.threshold(gray,155,255, cv2.THRESH_BINARY)[1]

# Invert color masking
mask = 255 - mask

kernel = np.ones((3,3), np.uint8) # 3x3 kernel pixel size
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # Erosion followed by dilation. Removes noise from outside
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Dilation followed by erosion. Removes noise from inside

# Use gaussian blur so that sharp edges in images are smoothed while minimizing too much blurring. 
# If kernal size is set to (0,0), then it is computed from sigma values. SigmaX refers to kernal standard deviation from X-axis.
mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)

# Use linear stretch for contrast enhancement. Pixel value of 127.5 goes to 0, but 255 stays 255
mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)

# Resize opencv images
mask=cv2.resize(mask,(400,400))
img=cv2.resize(img,(400,400))

# Image pop up
cv2.imshow("Original", img)
cv2.imshow("MASK", mask)

# Save mask image
cv2.imwrite("Mask.jpg", mask)

cv2.waitKey(5000) # Display image with a time limit (1000 = 1 second)
cv2.destroyAllWindows() # Destroy all pop up images at once