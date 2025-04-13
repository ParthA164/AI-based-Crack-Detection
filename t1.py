
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Constants
# MIN_CRACK_AREA = 2000  # Minimum area threshold for cracks
# MIN_CRACK_LENGTH = 30  # Minimum crack length in pixels
# MIN_CRACK_WIDTH = 5    # Minimum crack width in pixels
# SCALE_FACTOR_MM = 0.03341  # Scale factor in mm per pixel (adjust accordingly)

# def preprocess_image(image_path):
#     """Load and preprocess the image."""
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Use adaptive histogram equalization for better contrast
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     gray = clahe.apply(gray)
    
#     # Enhance the image
#     enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
    
#     return image, enhanced

# def detect_cracks(image):
#     """Detect cracks using edge detection."""
#     blurred = cv2.GaussianBlur(image, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#     return edges

# def segment_crack(image):
#     """Segment the crack from the background using adaptive thresholding."""
#     thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#     return thresh

# def get_longest_thread_length(contour):
#     """Calculate the longest thread length using the convex hull of the contour."""
#     hull = cv2.convexHull(contour)
#     max_dist = 0
#     for i in range(len(hull)):
#         for j in range(i + 1, len(hull)):
#             dist = np.linalg.norm(hull[i] - hull[j])
#             max_dist = max(max_dist, dist)
    
#     return max_dist

# def get_crack_width(contour, binary_image):
#     """Estimate the width of the crack by measuring distances perpendicular to the skeleton."""
#     # Create a blank image to draw the contour
#     mask = np.zeros_like(binary_image)

#     # Draw the contour on the mask
#     cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

#     # Apply distance transform to get width of dark regions
#     dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

#     # Find the maximum width of the crack
#     max_width = np.max(dist_transform) *2  # Multiply by 2 to get full width

#     return max_width

# def convert_to_mm(pixels):
#     """Convert pixels to millimeters."""
#     return pixels * SCALE_FACTOR_MM

# def analyze_crack_dimensions(image, original_image):
#     """Analyze and measure cracks in the image."""
#     contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     crack_info = []
#     for contour in contours:
#         area = cv2.contourArea(contour)
        
#         # Filter out very small cracks based on area, length, and width
#         if area >= MIN_CRACK_AREA:
#             # Calculate actual length (longest thread method)
#             length_pixels = get_longest_thread_length(contour)
#             width_pixels = get_crack_width(contour, image)
            
#             # Filter out cracks that are too small in length or width
#             if length_pixels >= MIN_CRACK_LENGTH and width_pixels >= MIN_CRACK_WIDTH:
#                 # Convert to millimeters
#                 length_mm = convert_to_mm(length_pixels)
#                 width_mm = convert_to_mm(width_pixels)

#                 # Store crack information
#                 crack_info.append({'length_pixels': length_pixels, 'width_pixels': width_pixels, 'length_mm': length_mm, 'width_mm': width_mm})

#                 # Draw the contour on the image
#                 cv2.drawContours(original_image, [contour], -1, (0, 255, 0), 2)

#     return crack_info, original_image

# def visualize_results(original_image, processed_image, edges, final_image):
#     """Visualize the processing steps."""
#     fig, ax = plt.subplots(1, 4, figsize=(20, 5))
#     titles = ["Original", "Preprocessed", "Edge Detection", "Detected Cracks"]
#     images = [
#         cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
#         processed_image,
#         edges,
#         cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
#     ]

#     for i, (img, title) in enumerate(zip(images, titles)):
#         ax[i].imshow(img, cmap='gray' if i != 3 else None)
#         ax[i].set_title(title)
#         ax[i].axis("off")

#     plt.show()

# # Example usage
# image_path = "D:/Desktop/cap1/new2.jpeg"  # Replace with your image path
# original_image, processed_image = preprocess_image(image_path)
# edges = detect_cracks(processed_image)
# segmented_image = segment_crack(edges)
# crack_dimensions, final_image = analyze_crack_dimensions(segmented_image, original_image)

# # Show results
# visualize_results(original_image, processed_image, edges, final_image)

# # Print crack dimensions
# if crack_dimensions:
#     print("\nMajor Cracks Detected (in mm):")
#     for idx, crack in enumerate(crack_dimensions):
#         print(f"Crack {idx+1}: Length={crack['length_mm']:.2f}cm, Width={crack['width_mm']:.2f}cm")
# else:
#     print("\nNo cracks detected.")



import cv2
import numpy as np
import matplotlib.pyplot as plt

# Constants
MIN_CRACK_AREA = 2000  # Minimum area threshold for cracks
MIN_CRACK_LENGTH = 30  # Minimum crack length in pixels
MIN_CRACK_WIDTH = 5    # Minimum crack width in pixels
SCALE_FACTOR_MM = 0.03341  # Scale factor in mm per pixel (adjust accordingly)

def preprocess_image(image_path):
    """Load and preprocess the image."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use adaptive histogram equalization for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Enhance the image
    enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
    
    return image, enhanced

def detect_cracks(image):
    """Detect cracks using edge detection."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def segment_crack(image):
    """Segment the crack from the background using adaptive thresholding."""
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def get_longest_thread_length(contour):
    """Calculate the longest thread length using the convex hull of the contour."""
    hull = cv2.convexHull(contour)
    max_dist = 0
    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            dist = np.linalg.norm(hull[i] - hull[j])
            max_dist = max(max_dist, dist)
    
    return max_dist

def get_crack_width(contour, binary_image):
    """Estimate the width of the crack by measuring distances perpendicular to the skeleton."""
    # Create a blank image to draw the contour
    mask = np.zeros_like(binary_image)

    # Draw the contour on the mask
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Apply distance transform to get width of dark regions
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Find the maximum width of the crack
    max_width = np.max(dist_transform)  # Multiply by 2 to get full width
    return max_width

def convert_to_mm(pixels):
    """Convert pixels to millimeters."""
    return pixels * SCALE_FACTOR_MM

def analyze_crack_dimensions(image, original_image):
    """Analyze and measure cracks in the image."""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crack_info = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter out very small cracks based on area, length, and width
        if area >= MIN_CRACK_AREA:
            # Calculate actual length (longest thread method)
            length_pixels = get_longest_thread_length(contour)
            width_pixels = get_crack_width(contour, image)
            
            # Filter out cracks that are too small in length or width
            if length_pixels >= MIN_CRACK_LENGTH and width_pixels >= MIN_CRACK_WIDTH:
                # Convert to millimeters
                length_mm = convert_to_mm(length_pixels)
                width_mm = convert_to_mm(width_pixels)-0.45

                # Store crack information
                crack_info.append({'length_pixels': length_pixels, 'width_pixels': width_pixels, 'length_mm': length_mm, 'width_mm': width_mm})

                # Draw the contour on the image
                cv2.drawContours(original_image, [contour], -1, (0, 255, 0), 2)

    return crack_info, original_image

def visualize_results(original_image, processed_image, edges, final_image):
    """Visualize the processing steps."""
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    titles = ["Original", "Preprocessed", "Edge Detection", "Detected Cracks"]
    images = [
        cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
        processed_image,
        edges,
        cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    ]

    for i, (img, title) in enumerate(zip(images, titles)):
        ax[i].imshow(img, cmap='gray' if i != 3 else None)
        ax[i].set_title(title)
        ax[i].axis("off")

    plt.show()

# Example usage
image_path = "D:/Desktop/cap1/sample1.jpeg"  # Replace with your image path
original_image, processed_image = preprocess_image(image_path)
edges = detect_cracks(processed_image)
segmented_image = segment_crack(edges)
crack_dimensions, final_image = analyze_crack_dimensions(segmented_image, original_image)

# Show results
visualize_results(original_image, processed_image, edges, final_image)

# Print crack dimensions
if crack_dimensions:
    print("\nMajor Cracks Detected (in mm):")
    for idx, crack in enumerate(crack_dimensions):
        print(f"Crack {idx+1}: Length={crack['length_mm']:.2f}cm, Width={crack['width_mm']:.2f}cm")
else:
    print("\nNo cracks detected.")