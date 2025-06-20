import cv2
import numpy as np
from PIL import Image, ImageFilter
import argparse

# A lot of this was written by Claude (AI)

DEBUG = False


def _debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def downsample_image(
    input_path,
    output_path,
    target_width=None,
    target_height=None,
    quality_method="lanczos",
    sharpen=True,
):
    """
    Downsample an image while preserving fine details like text and small elements.

    Args:
        input_path (str): Path to the input image
        output_path (str): Path to save the downsampled image
        target_width (int): Target width in pixels
        target_height (int): Target height in pixels
        quality_method (str): Resampling method ('lanczos', 'bicubic', 'antialias')
        sharpen (bool): Whether to apply sharpening after downsampling
    """

    # Open the image
    img = Image.open(input_path)
    original_width, original_height = img.size

    _debug_print(f"Original image size: {original_width} x {original_height}")

    # Calculate target dimensions if not provided
    if target_width is None and target_height is None:
        # Default to half size
        target_width = original_width // 2
        target_height = original_height // 2
    elif target_width is None:
        # Calculate width based on aspect ratio
        aspect_ratio = original_width / original_height
        target_width = int(target_height * aspect_ratio)
    elif target_height is None:
        # Calculate height based on aspect ratio
        aspect_ratio = original_height / original_width
        target_height = int(target_width * aspect_ratio)

    _debug_print(f"Target image size: {target_width} x {target_height}")

    # Choose resampling method
    if quality_method == "lanczos":
        resample_method = Image.Resampling.LANCZOS
    elif quality_method == "bicubic":
        resample_method = Image.Resampling.BICUBIC
    elif quality_method == "antialias":
        resample_method = (
            Image.Resampling.LANCZOS
        )  # ANTIALIAS is deprecated, use LANCZOS
    else:
        resample_method = Image.Resampling.LANCZOS

    # Downsample the image
    downsampled = img.resize((target_width, target_height), resample_method)

    # Apply sharpening to preserve fine details
    if sharpen:
        # Create a subtle sharpening filter
        sharpen_filter = ImageFilter.UnsharpMask(radius=0.5, percent=120, threshold=2)
        downsampled = downsampled.filter(sharpen_filter)

    # Save the result
    downsampled.save(output_path, quality=95, optimize=True)
    _debug_print(f"Downsampled image saved to: {output_path}")

    return downsampled


def advanced_downsample(input_path, output_path, target_width=None, target_height=None):
    """
    Advanced downsampling using OpenCV for better control over the process.
    This method uses area interpolation which is often better for downsampling.
    """

    # Read image with OpenCV
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image from {input_path}")

    original_height, original_width = img.shape[:2]
    _debug_print(f"Original image size: {original_width} x {original_height}")

    # Calculate target dimensions
    if target_width is None and target_height is None:
        target_width = original_width // 2
        target_height = original_height // 2
    elif target_width is None:
        aspect_ratio = original_width / original_height
        target_width = int(target_height * aspect_ratio)
    elif target_height is None:
        aspect_ratio = original_height / original_width
        target_height = int(target_width * aspect_ratio)

    _debug_print(f"Target image size: {target_width} x {target_height}")

    # Use INTER_AREA for downsampling (best for shrinking images)
    downsampled = cv2.resize(
        img, (target_width, target_height), interpolation=cv2.INTER_AREA
    )

    # Apply subtle sharpening using a kernel
    kernel = np.array([[-0.5, -0.5, -0.5], [-0.5, 5.0, -0.5], [-0.5, -0.5, -0.5]])
    sharpened = cv2.filter2D(downsampled, -1, kernel * 0.1)  # Gentle sharpening

    # Blend original downsampled with sharpened version
    result = cv2.addWeighted(downsampled, 0.7, sharpened, 0.3, 0)

    # Save the result
    cv2.imwrite(output_path, result)
    _debug_print(f"Advanced downsampled image saved to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Downsample an image while preserving fine details"
    )
    parser.add_argument("input", help="Input image path")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--width", type=int, help="Target width in pixels")
    parser.add_argument("--height", type=int, help="Target height in pixels")
    parser.add_argument(
        "--method",
        choices=["basic", "advanced"],
        default="advanced",
        help="Downsampling method to use",
    )
    parser.add_argument(
        "--quality",
        choices=["lanczos", "bicubic", "antialias"],
        default="lanczos",
        help="Resampling quality method",
    )
    parser.add_argument(
        "--no-sharpen",
        action="store_true",
        help="Disable sharpening after downsampling",
    )

    args = parser.parse_args()

    try:
        if args.method == "advanced":
            advanced_downsample(args.input, args.output, args.width, args.height)
        else:
            downsample_image(
                args.input,
                args.output,
                args.width,
                args.height,
                args.quality,
                not args.no_sharpen,
            )

        print("Downsampling completed successfully!")

    except Exception as e:
        print(f"Error: {e}")


# Example usage for your specific case
def downsample_logo_example():
    """
    Example function showing how to downsample the AURJ Summer Camp logo
    """
    # Assuming the large image is saved as 'large_logo.png'
    # and you want to create a smaller version similar to the second image

    input_file = "large_logo.png"  # Replace with your actual file path
    output_file = "small_logo.png"

    # Method 1: Basic downsampling with PIL
    print("Method 1: Basic downsampling")
    downsample_image(
        input_file, "basic_" + output_file, target_width=200, target_height=60
    )  # Adjust dimensions as needed

    # Method 2: Advanced downsampling with OpenCV
    print("\nMethod 2: Advanced downsampling")
    advanced_downsample(
        input_file, "advanced_" + output_file, target_width=200, target_height=60
    )


if __name__ == "__main__":
    # Uncomment the line below to run the example
    # downsample_logo_example()

    # Or run the main function for command-line usage
    main()
