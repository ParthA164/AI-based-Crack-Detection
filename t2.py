import cv2
import numpy as np
import matplotlib.pyplot as plt

# Constants
MIN_CRACK_AREA = 2000  # Minimum area threshold for cracks
MIN_CRACK_LENGTH = 30  # Minimum crack length in pixels
MIN_CRACK_WIDTH = 5    # Minimum crack width in pixels
SCALE_FACTOR_MM = 0.0334  # Scale factor in mm per pixel (adjust accordingly)

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
    mask = np.zeros_like(binary_image)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_width = np.max(dist_transform)  
    return max_width

def convert_to_mm(pixels):
    """Convert pixels to millimeters."""
    return pixels * SCALE_FACTOR_MM

def analyze_crack_dimensions(image, original_image, x_offset, y_offset):
    """Analyze and measure cracks in the image with offset correction."""
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crack_info = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area >= MIN_CRACK_AREA:
            length_pixels = get_longest_thread_length(contour)
            width_pixels = get_crack_width(contour, image)
            
            if length_pixels >= MIN_CRACK_LENGTH and width_pixels >= MIN_CRACK_WIDTH:
                length_mm = convert_to_mm(length_pixels)
                width_mm = convert_to_mm(width_pixels) - 0.45
                crack_info.append({'length_pixels': length_pixels, 'width_pixels': width_pixels, 'length_mm': length_mm, 'width_mm': width_mm})
                
                # Apply offset correction before drawing contours
                contour_shifted = contour + np.array([x_offset, y_offset])
                cv2.drawContours(original_image, [contour_shifted], -1, (0, 255, 0), 2)

    return crack_info, original_image

def add_black_padding(image, target_size):
    """Add black padding to make all images the same size."""
    h, w = image.shape[:2]
    th, tw = target_size
    top = (th - h) // 2
    bottom = th - h - top
    left = (tw - w) // 2
    right = tw - w - left
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

def visualize_results(original_image, processed_image, edges, final_image):
    """Visualize the processing steps with uniform dimensions."""
    target_size = max(original_image.shape[:2]), max(original_image.shape[:2])
    
    # Convert grayscale images to RGB for proper display
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    images = [
        add_black_padding(original_image, target_size),
        add_black_padding(processed_image, target_size),
        add_black_padding(edges, target_size),
        add_black_padding(final_image, target_size)
    ]
    
    titles = ["Original", "Preprocessed", "Edge Detection", "Detected Cracks"]

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        ax[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[i].set_title(title)
        ax[i].axis("off")

    plt.show()

# Load and preprocess image
image_path = "D:/Desktop/cap1/sample2.jpeg"  # Replace with your image path
original_image, processed_image = preprocess_image(image_path)

# Select area for crack detection
screen_res = 1280, 720  
scale_width = screen_res[0] / original_image.shape[1]
scale_height = screen_res[1] / original_image.shape[0]
scale = min(scale_width, scale_height)
window_width = int(original_image.shape[1] * scale)
window_height = int(original_image.shape[0] * scale)
dim = (window_width, window_height)
resized_image = cv2.resize(original_image, dim, interpolation=cv2.INTER_AREA)

roi = cv2.selectROI("Select Crack Area", resized_image, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Crack Area")

x, y, w, h = int(roi[0] / scale), int(roi[1] / scale), int(roi[2] / scale), int(roi[3] / scale)
selected_region = processed_image[y:y+h, x:x+w]

edges = detect_cracks(selected_region)
segmented_image = segment_crack(edges)
crack_dimensions, final_image = analyze_crack_dimensions(segmented_image, original_image, x, y)

visualize_results(original_image, processed_image, edges, final_image)

if crack_dimensions:
    print("\nMajor Cracks Detected (in mm):")
    for idx, crack in enumerate(crack_dimensions):
        print(f"Crack {idx+1}: Length={crack['length_mm']:.2f}cm, Width={crack['width_mm']:.2f}cm")
else:
    print("\nNo cracks detected.")