from PIL import Image, ImageEnhance

def enhance_image(image_path, output_path=None):
    # Load the image
    image = Image.open(image_path)

    # Enhance brightness: +18 on a scale, approximately factor of 1.18
    brightness_enhancer = ImageEnhance.Brightness(image)
    image = brightness_enhancer.enhance(1.22)

    # Enhance contrast: +6 on a scale, approximately factor of 1.06
    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(1.08)

    # Save or return the modified image
    if output_path:
        image.save(output_path)
    return image

# Example usage
if __name__ == "__main__":
    input_path = "D:/Desktop/Cap1/sample4.jpeg"           # Replace with your image path
    output_path = "enhanced_image1.jpg"      # Output file path

    result_image = enhance_image(input_path, output_path)
    result_image.show()  # Display the modified image
