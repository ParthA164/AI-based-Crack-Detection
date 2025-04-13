# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Constants
# MIN_CRACK_AREA = 2000
# MIN_CRACK_LENGTH = 30
# MIN_CRACK_WIDTH = 5
# SCALE_FACTOR_MM = 0.03341

# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     gray = clahe.apply(gray)
#     enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
#     return image, enhanced

# def detect_cracks(image):
#     blurred = cv2.GaussianBlur(image, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#     return edges

# def segment_crack(image):
#     thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                                    cv2.THRESH_BINARY_INV, 11, 2)
#     return thresh

# def skeletonize(binary_img):
#     """Use morphological thinning to get the skeleton."""
#     skel = np.zeros(binary_img.shape, np.uint8)
#     element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#     done = False
#     temp = np.zeros_like(binary_img)
#     eroded = np.zeros_like(binary_img)

#     img = binary_img.copy()

#     while not done:
#         eroded = cv2.erode(img, element)
#         temp = cv2.dilate(eroded, element)
#         temp = cv2.subtract(img, temp)
#         skel = cv2.bitwise_or(skel, temp)
#         img = eroded.copy()

#         if cv2.countNonZero(img) == 0:
#             done = True

#     return skel

# def get_crack_length_skeleton(contour, binary_image):
#     mask = np.zeros_like(binary_image)
#     cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
#     skeleton = skeletonize(mask)
#     length = cv2.countNonZero(skeleton)
#     return length

# def get_crack_width_from_skeleton(contour, binary_image):
#     mask = np.zeros_like(binary_image)
#     cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

#     dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
#     skeleton = skeletonize(mask)

#     # Get all distance values along skeleton path (half-widths)
#     width_values = dist_transform[skeleton == 255] * 2  # full width

#     if len(width_values) > 0:
#         return np.mean(width_values)
#     else:
#         return 0

# def convert_to_mm(pixels):
#     return pixels * SCALE_FACTOR_MM

# def analyze_crack_dimensions(image, original_image):
#     contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     crack_info = []
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area >= MIN_CRACK_AREA:
#             length_pixels = get_crack_length_skeleton(contour, image)
#             width_pixels = get_crack_width_from_skeleton(contour, image)

#             if length_pixels >= MIN_CRACK_LENGTH and width_pixels >= MIN_CRACK_WIDTH:
#                 length_mm = convert_to_mm(length_pixels)
#                 width_mm = convert_to_mm(width_pixels) - 0.45
#                 crack_info.append({
#                     'length_pixels': length_pixels,
#                     'width_pixels': width_pixels,
#                     'length_mm': length_mm,
#                     'width_mm': width_mm
#                 })
#                 cv2.drawContours(original_image, [contour], -1, (0, 255, 0), 2)

#     return crack_info, original_image

# def visualize_results(original_image, processed_image, edges, final_image):
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

# # ---- USAGE ----
# image_path = "D:/Desktop/cap1/sample1.jpeg"  # Replace with your image path
# original_image, processed_image = preprocess_image(image_path)
# edges = detect_cracks(processed_image)
# segmented_image = segment_crack(edges)
# crack_dimensions, final_image = analyze_crack_dimensions(segmented_image, original_image)

# visualize_results(original_image, processed_image, edges, final_image)

# # Print results
# if crack_dimensions:
#     print("\nMajor Cracks Detected (in mm):")
#     for idx, crack in enumerate(crack_dimensions):
#         print(f"Crack {idx+1}: Length = {crack['length_mm']:.2f} cm, Width = {crack['width_mm']:.2f} cm")
# else:
#     print("\nNo cracks detected.")


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Constants
MIN_CRACK_AREA = 2000
MIN_CRACK_LENGTH = 30
MIN_CRACK_WIDTH = 5
SCALE_FACTOR_MM = 0.03341
MIN_ASPECT_RATIO = 1.5  # Higher value = more elongated structuresq

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=10)
    return image, enhanced

def detect_cracks(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def segment_crack(image):
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    # Remove small blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned

def skeletonize(binary_img):
    skel = np.zeros(binary_img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary_img.copy()

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel

def get_crack_length_skeleton(contour, binary_image):
    mask = np.zeros_like(binary_image)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    skeleton = skeletonize(mask)
    return cv2.countNonZero(skeleton)

def get_crack_width_from_skeleton(contour, binary_image):
    mask = np.zeros_like(binary_image)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    skeleton = skeletonize(mask)
    width_values = dist_transform[skeleton == 255] * 2
    return np.mean(width_values) if len(width_values) > 0 else 0

def convert_to_mm(pixels):
    return pixels * SCALE_FACTOR_MM

def is_elongated(contour):
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
    return aspect_ratio >= MIN_ASPECT_RATIO

def analyze_crack_dimensions(image, original_image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crack_info = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CRACK_AREA or not is_elongated(contour):
            continue

        length_pixels = get_crack_length_skeleton(contour, image)
        width_pixels = get_crack_width_from_skeleton(contour, image)

        if length_pixels >= MIN_CRACK_LENGTH and width_pixels >= MIN_CRACK_WIDTH:
            length_mm = convert_to_mm(length_pixels)
            width_mm = convert_to_mm(width_pixels) - 0.45
            crack_info.append({
                'length_pixels': length_pixels,
                'width_pixels': width_pixels,
                'length_mm': length_mm,
                'width_mm': width_mm
            })
            cv2.drawContours(original_image, [contour], -1, (0, 255, 0), 2)

    return crack_info, original_image

def visualize_results(original_image, processed_image, edges, final_image):
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

# ---- USAGE ----
image_path = "D:/Desktop/cap1/sample4.jpeg"  # Replace with your image path
original_image, processed_image = preprocess_image(image_path)
edges = detect_cracks(processed_image)
segmented_image = segment_crack(edges)
crack_dimensions, final_image = analyze_crack_dimensions(segmented_image, original_image.copy())

visualize_results(original_image, processed_image, edges, final_image)

# Print results
if crack_dimensions:
    print("\nMajor Cracks Detected (in mm):")
    for idx, crack in enumerate(crack_dimensions):
        print(f"Crack {idx+1}: Length = {crack['length_mm']:.2f} cm, Width = {crack['width_mm']:.2f} cm")
else:
    print("\nNo cracks detected.")