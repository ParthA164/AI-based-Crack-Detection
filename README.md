# AI-Based Crack Detection System

---

### 1. Methodology
This project implements an image processing pipeline to automatically detect cracks in surfaces and measure their dimensions using computer vision techniques:

1. **Image Preprocessing:**
   - Grayscale conversion and CLAHE for contrast enhancement.
   - Brightening and contrast adjustment for improved crack visibility.

2. **Crack Detection:**
   - Edge detection using Gaussian Blur and Canny detector.
   - Adaptive thresholding and morphological operations for crack segmentation.

3. **Crack Analysis:**
   - Skeletonization for precise crack length estimation.
   - Distance transform to estimate average crack width.
   - Filtering based on area, elongation (aspect ratio), length, and width.

4. **Measurement Calibration:**
   - A separate script (`ruler.py`) is used to define real-world pixel-to-mm scale using an image of a ruler.
   - This scale is used to convert detected crack dimensions from pixels to millimeters and centimeters.

---

### 2. Description
The AI-Based Crack Detection System is designed to detect major surface cracks from input images and analyze their dimensions in both pixels and real-world units (mm/cm). It helps in structural health monitoring and can be used in civil engineering or maintenance automation scenarios.

**Key Features:**
- Fully automated pipeline with minimal user input.
- Real-world measurement using a custom scaling tool (`ruler.py`).
- Skeleton-based crack length estimation.
- Distance transform-based crack width estimation.
- Visualization of original, processed, edge-detected, and crack-detected images.

---

### 3. Input / Output

#### **Input:**
- Surface images (`sample1.jpg` to `sample8.jpg`) for crack detection.
- A ruler image (e.g., `sample2.jpeg`) for pixel-to-mm scale calibration.

#### **Output:**
- Annotated images with detected cracks.
- Console output of crack dimensions:
  - Length in centimeters
  - Width in centimeters
- Scale calibration result printed from `ruler.py`

---

### 4. Installation

### Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/ParthA164/AI-based-Crack-Detection.git
   cd AI-based-Crack-Detection
   ```

2. Install dependencies:
   ```
   pip install opencv-python numpy matplotlib
   ```

### 5. Usage

### Step 1: Calibrate Scale 

For accurate dimension measurement, run the ruler calibration script first:

```
python ruler.py
```

You will be prompted to:
1. Load an image containing a ruler
2. Select two points corresponding to a known length
3. Enter the real-world length between these points

This calibrates the scale (mm per pixel) for accurate crack dimensioning.

### Step 2: Detect Cracks

Run the main crack detection script to analyze the input image:

```
python t5.py
```

The script will:
1. Load the target image
2. Process the image to identify cracks
3. Calculate dimensions of detected cracks
4. Display results both visually and in the terminal

## Example Output

### Console Output:
```
Major Cracks Detected (in cm):
Crack 1: Length = 3.47 cm, Width = 0.92 cm
Crack 2: Length = 2.81 cm, Width = 0.86 cm
```

### Visual Output:
The program will display the original image with green contours outlining the detected cracks.

