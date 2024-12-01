
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Simulated Image (example gradient image with sharp edges)
image = np.zeros((100, 100))
image[:50, :] = 0.3  # Dark region
image[50:, :] = 0.8  # Bright region
image[:, 50] = 1.0  # Sharp edge

# Step 1: Original Image
plt.figure(figsize=(15, 10))
plt.subplot(3, 2, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=1)
plt.title("Step 1: Original Image (Input)")
plt.axis("off")

# Step 2: Grayscale Gradient (Approximating Gradyan Calculation)
gradient_x = np.abs(np.gradient(image, axis=1))
gradient_y = np.abs(np.gradient(image, axis=0))
gradient_magnitude = gradient_x + gradient_y

plt.subplot(3, 2, 2)
plt.imshow(gradient_magnitude, cmap='viridis')
plt.title("Step 2: Gradient Magnitude")
plt.axis("off")

# Step 3: Weights Based on Gradient (Edge Preservation Weights)
sigma = 0.1
weights = np.exp(-gradient_magnitude**2 / (2 * sigma**2))

plt.subplot(3, 2, 3)
plt.imshow(weights, cmap='viridis')
plt.title("Step 3: Edge Preservation Weights")
plt.axis("off")

# Step 4: Simulated Tonal Component (Base Layer) with Weights
base_layer = gaussian_filter(image * weights, sigma=3)

plt.subplot(3, 2, 4)
plt.imshow(base_layer, cmap='gray', vmin=0, vmax=1)
plt.title("Step 4: Base Layer (Tonal Component)")
plt.axis("off")

# Step 5: Detail Layer (High-Frequency Component)
detail_layer = image - base_layer

plt.subplot(3, 2, 5)
plt.imshow(detail_layer, cmap='gray', vmin=-0.5, vmax=0.5)
plt.title("Step 5: Detail Layer (High-Frequency Component)")
plt.axis("off")

# Step 6: Reconstructed Image (Base + Detail)
reconstructed_image = base_layer + detail_layer

plt.subplot(3, 2, 6)
plt.imshow(reconstructed_image, cmap='gray', vmin=0, vmax=1)
plt.title("Step 6: Reconstructed Image")
plt.axis("off")

plt.tight_layout()
plt.show()

