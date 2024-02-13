# Image Processing for Noise Reduction, Text Extraction, and Color Swapping

### INSAT IIA4

**Technologies:** Python, OpenCV, NumPy, Matplotlib Pyplot, DFT, Filters, Inverse FFT, Canny Edge Detector, Noise Reduction

## Introduction

This is is a semester project I worked on for the Image Processing subject during my 4th year of Industrial Computing and Automation (IIA4) at INSAT. It focuses on advanced image processing techniques aimed at noise reduction, text extraction, and color swapping. Leveraging a combination of Python programming and OpenCV library, the project explores innovative methods to enhance image quality and extract valuable information from visual data.

## Project Scope

The project was conducted from December 8, 2022, to December 19, 2022.

## Technical Details

- **Discrete Fourier Transform (DFT) with Circular Low Pass Filter**: Leveraged DFT with a strategically positioned 70-pixel circular low pass filter to selectively mask unwanted frequency components, resulting in a remarkable 90% noise reduction.
- **Inverse Fast Fourier Transform (FFT)**: Executed an Inverse FFT on the modified frequency spectrum to seamlessly translate the image back to the spatial domain, enhancing relevant features with minimal artifacts.
- **Visual Analysis and Comparison**: Conducted visual analysis and comparison of input image, Fourier spectrum, frequency domain-filtered image, and final result after inverse FFT, validating the effectiveness of the noise reduction technique.
- **Non-linear Median Filter for Noise Reduction**: Employed a non-linear median filter with a 5x5-size kernel to reduce salt-and-pepper noise, resulting in significantly less blur on edges compared to linear filters.
- **Text Extraction and Edge Detection**: Defined a Region of Interest (ROI) to isolate desired text, then applied Canny Edge Detector to highlight text edges in the binarized image, facilitating clear delineation of text boundaries.
- **Color Swapping Algorithm**: Implemented a color swap algorithm on an image of the German flag by defining precise intervals for each color segment, achieving the desired Lithuanian flag output.

## Getting Started

### Installation

1. Ensure compatibility and proper setup of Python environment.
2. Install required libraries including OpenCV, NumPy, and Matplotlib.
3. Configure system settings as per project requirements.

### Usage

1. Load the target image into the Python environment.
2. Execute the desired image processing algorithms as per project objectives.
3. Observe and analyze the processed images for desired outcomes.

## Contributing

Contributions aimed at further enhancing image processing functionalities are welcomed.

## License

This project is licensed under the [GPL-3.0 License](LICENSE).

## Contacts

For inquiries or collaboration opportunities, please contact:

- Elyes Khechine: elyeskhechine@gmail.com
