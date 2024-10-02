# Implementation-of-filter
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step 1: Load the Image

1. Read the image file using a suitable image processing library like OpenCV or PIL.
2. Convert the image to grayscale if necessary for grayscale filtering techniques.
### Step 2: Choose a Filter

1. Decide on the type of filter you want to apply based on your desired outcome. Some common filters include:

    a. Averaging filter

    b. Gaussian filter

    c. Median filter

    d. Laplacian filter

### Step 3: Create the Filter Kernel

A filter kernel is a small matrix that is applied to each pixel in the image to produce the filtered result.
The size and values of the kernel determine the filter's behavior.
For example, an averaging filter kernel has all elements equal to 1/N, where N is the kernel size.
### Step 4: Apply the Filter

Use the library's functions to apply the filter to the image.
The filtering process typically involves convolving the image with the filter kernel.
### Step 5: Display or Save the Result

Visualize the filtered image using a suitable method (e.g., OpenCV's imshow, Matplotlib).
Save the filtered image to a file if needed.

## Program:
### Developed By   :Subashini S
### Register Number:212222240106


### 1. Smoothing Filters

i) Using Averaging Filter
```
import cv2
import numpy as np

# Load the image
image = cv2.imread("shrine.jpeg")

# Create the averaging kernel
kernel = np.ones((3, 3)) / 9

# Apply the averaging filter
averaging_smoothed = cv2.filter2D(image, -1, kernel)

# Display the result
cv2.imshow("Averaging Smoothed", averaging_smoothed)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
ii) Using Weighted Averaging Filter
```
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image1 = cv2.imread("shrine.jpeg")

# Convert the image to grayscale
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Create the weighted averaging kernel
kernel1 = np.array([[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]]) / 16  # Normalized weights for better visualization

# Apply the weighted averaging filter
image3 = cv2.filter2D(image2, -1, kernel1)

# Create the figure and subplots
plt.figure(figsize=(8, 8))

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image2, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Display the filtered image
plt.subplot(1, 2, 2)
plt.imshow(image3, cmap='gray')
plt.title("Weighted Average Filter Image")
plt.axis("off")

# Show the plot
plt.show()
```
iii) Using Gaussian Filter
```
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image1 = cv2.imread("shrine.jpeg")

# Convert the image to grayscale
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur with a kernel size of 5x5 and sigmaX (standard deviation) of 0 (calculated automatically)
gaussian_blur = cv2.GaussianBlur(image2, (5, 5), 0)

# Create the figure and subplots
plt.figure(figsize=(8, 8))

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image2, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Display the Gaussian blurred image
plt.subplot(1, 2, 2)
plt.imshow(gaussian_blur, cmap='gray')
plt.title("Gaussian Blur")
plt.axis("off")

# Show the plot
plt.show()

```
v) Using Median Filter
```
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image1 = cv2.imread("shrine.jpeg")

# Convert the image to grayscale
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

# Apply median filter with a kernel size of 3x3
median = cv2.medianBlur(image2, 3)

# Create the figure and subplots
plt.figure(figsize=(8, 8))

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image2, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Display the median filtered image
plt.subplot(1, 2, 2)
plt.imshow(median, cmap='gray')
plt.title("Median Filter")
plt.axis("off")

# Show the plot
plt.show()
```

### 2. Sharpening Filters
i) Using Laplacian Linear Kernal
```
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image1 = cv2.imread("red.jpg")

# Convert the image to RGB color space
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

# Create the Laplacian kernel
kernel = np.array([[-1, -1, -1],
                   [-1, 8, -1],
                   [-1, -1, -1]])

# Apply the Laplacian kernel
image3 = cv2.filter2D(image2, -1, kernel)

# Create the figure and subplots
plt.figure(figsize=(10, 8))

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

# Display the Laplacian filtered image
plt.subplot(1, 2, 2)
plt.imshow(image3)
plt.title("Laplacian Kernel")
plt.axis("off")

plt.show()
```
ii) Using Laplacian Operator
```
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image1 = cv2.imread("shrine.jpeg")

# Convert the image to RGB color space
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

# Apply the Laplacian operator
laplacian = cv2.Laplacian(image2, cv2.CV_64F)  # Use CV_64F for better precision

# Convert the Laplacian image back to uint8 for display
laplacian = cv2.convertScaleAbs(laplacian)

# Create the figure and subplots
plt.figure(figsize=(8, 8))

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

# Display the Laplacian filtered image
plt.subplot(1, 2, 2)
plt.imshow(laplacian)
plt.title("Laplacian Operator")
plt.axis("off")

plt.show()





```

## OUTPUT:
### 1. Smoothing Filters
</br>

i) Using Averaging Filter

![Screenshot 2024-10-03 001115](https://github.com/user-attachments/assets/3fe165bc-d3d5-49c1-a218-555fd0e48fdf)


ii)Using Weighted Averaging Filter
![download (2)](https://github.com/user-attachments/assets/af682605-57e1-4cc1-b3d6-2459e3b25d5d)


iii) Using Gaussian Filter
![download](https://github.com/user-attachments/assets/66caeaf0-258b-44ae-af36-56dee0930ca3)


iv) Using Median Filter

![download (1)](https://github.com/user-attachments/assets/ca624d52-e3cf-4d99-b3cf-40e34a219d72)

### 2. Sharpening Filters
i) Using Laplacian Kernal

![download (3)](https://github.com/user-attachments/assets/99cd49f7-4f81-4cad-baf9-2178d229904e)

ii) Using Laplacian Operator
![download (4)](https://github.com/user-attachments/assets/69e695e9-f6b4-4b87-9de1-5476906d199b)

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain is executed successfully.

