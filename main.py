import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Input Imange
image = np.zeros((100, 100))
image[:50, :] = 0.3  # Dark region
image[50:, :] = 0.8  # Bright region

# Add a sharp edge
image[:, 50] = 1.0

# Apply Weighted Least Squares-inspired edge preservation
# Simulating the "weights" (gradient-based smoothing)
weights = np.abs(np.gradient(image, axis=0)) + np.abs(np.gradient(image, axis=1))
weights = 1 / (1 + weights)  # Higher weights for smooth regions

# Smoothed image (conceptual, using Gaussian as a proxy)
smoothed_image = gaussian_filter(image * weights, sigma=1)

# Visualization
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Original Image
axs[0].imshow(image, cmap='gray', vmin=0, vmax=1)
axs[0].set_title('Original Image (With Edge)')
axs[0].axis('off')

# Weights
axs[1].imshow(weights, cmap='viridis')
axs[1].set_title('Weights (Edge Preservation)')
axs[1].axis('off')

# Smoothed Image
axs[2].imshow(smoothed_image, cmap='gray', vmin=0, vmax=1)
axs[2].set_title('Smoothed Image (WLS Concept)')
axs[2].axis('off')

plt.tight_layout()
plt.show()
