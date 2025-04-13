import cv2
import numpy as np

# Global variables to store the start and end points
start_point = None
end_point = None

# Mouse callback function to capture the start and end points of the ruler
def select_points(event, x, y, flags, param):
    global start_point, end_point, image_copy
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # First click, record the start point
        if start_point is None:
            start_point = (x, y)
            cv2.circle(image_copy, start_point, 5, (0, 255, 0), -1)  # Draw a circle at the start point
        else:
            # Second click, record the end point and draw the line
            end_point = (x, y)
            cv2.circle(image_copy, end_point, 5, (0, 0, 255), -1)  # Draw a circle at the end point
            cv2.line(image_copy, start_point, end_point, (0, 255, 0), 2)  # Draw the line
            cv2.imshow("Image with Ruler", image_copy)

# Load the image
image = cv2.imread('sample2.jpeg')  # Replace with your image path

# Get the screen resolution (width and height) using OpenCV
screen_width = 1920  # You can change this to your screen width
screen_height = 1080  # You can change this to your screen height

# Resize the image to fit within the screen resolution
aspect_ratio = image.shape[1] / image.shape[0]
new_width = screen_width - 100  # Leave a small margin
new_height = int(new_width / aspect_ratio)

# Resize image if it's larger than the screen size
if new_height > screen_height:
    new_height = screen_height - 100  # Leave a small margin
    new_width = int(new_height * aspect_ratio)

resized_image = cv2.resize(image, (new_width, new_height))

# Create a copy of the resized image
image_copy = resized_image.copy()

# Create a window and set the mouse callback function
cv2.imshow("Image with Ruler", resized_image)
cv2.setMouseCallback("Image with Ruler", select_points)

# Wait until two points are selected
cv2.waitKey(0)
cv2.destroyAllWindows()

# Ensure that both points have been selected
if start_point and end_point:
    # Calculate the pixel distance between the two points
    pixel_distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
    print(f"Ruler length in pixels: {pixel_distance:.2f} pixels")

    # Example: Known real-world length of the ruler (in mm)
    reference_length_mm = 150  # You should input the actual length of your ruler in mm

    # Calculate the scale (mm per pixel)
    scale = reference_length_mm / pixel_distance
    print(f"Scale: {scale:.4f} mm per pixel")
else:
    print("Please select both the start and end points of the ruler.")
