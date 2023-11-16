# build a document scanner that takes an image of a document 
# and transforms it into a scanned document by applying transformations


import cv2
from scanner_functions import order_points, four_point_transform

def main():
    image_path = input("enter the path to your file: \n")
    # Read the image
    image = cv2.imread(image_path)
    #image = cv2.imread("document.jpg")

    # Resize the image to improve processing speed (optional)
    image = cv2.resize(image, (800, 600))

    #   convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour (presumably the document)
    contour = max(contours, key=cv2.contourArea)

    # Get the approximate contour
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Apply perspective transformation
    warped = four_point_transform(image, approx.reshape(4, 2))

    # Display the original and scanned images
    cv2.imshow("Original", image)
    cv2.imshow("Scanned", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()