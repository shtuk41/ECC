import cv2
import numpy as np

# Read the images to be aligned
im1 =  cv2.imread("Centered.tif");
im2 =  cv2.imread("FocalRotation.tif");

im1 = im1[2366 : 3424 , 3296 : 4354]
im2 = im2[2366 : 3424 , 3296 : 4354]

# Convert images to grayscale
im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

# Find size of image1
sz = im1.shape

# Define the motion model
warp_mode = cv2.MOTION_AFFINE

# Define 2x3 or 3x3 matrices and initialize the matrix to identity
if warp_mode == cv2.MOTION_HOMOGRAPHY :
    warp_matrix = np.eye(3, 3, dtype=np.float32)
else :
    warp_matrix = np.eye(2, 3, dtype=np.float32)

# Specify the number of iterations.
number_of_iterations = 5000;

# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-3;

# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

# Run the ECC algorithm. The results are stored in warp_matrix.
(cc, warp_matrix) = cv2.findTransformECC(im1_gray,
                                         im2_gray,
                                         warp_matrix,
                                         warp_mode,
                                         criteria,
                                         inputMask=None,
                                         gaussFiltSize=1)

print (warp_matrix)

if warp_mode == cv2.MOTION_HOMOGRAPHY:
    # Use warpPerspective for Homography
    im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
else :
    # Use warpAffine for Translation, Euclidean and Affine
    im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);


# Show final results
im1S = cv2.resize(im1, (960, 540))
cv2.imshow("Image 1", im1S)
im2S = cv2.resize(im2, (960, 540))
cv2.imshow("Image 2", im2S)
im2Aligned = cv2.resize(im2_aligned, (960, 540))
cv2.imshow("Aligned Image 2", im2Aligned)
cv2.waitKey(0)
