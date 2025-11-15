"""
MRI Transformation Module

Provides basic image transformations for MRI data:
- Add synthetic noise (Gaussian, Salt & Pepper, Speckle)
- Apply filters (Gaussian blur, Median, Sharpen, Edge detection)
- Geometric transformations (Rotate, Scale, Flip)
- Comparison tools (Original vs Transformed)

Note: These are image-level transformations, not real MRI physics simulations.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional
from scipy import ndimage
from skimage import filters, transform, util, exposure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MRITransformer:
    """Apply various transformations to MRI volumes and slices."""
    
    def __init__(self):
        """Initialize the MRI transformer."""
        logger.info("MRI Transformer initialized")
    
    # ==================== NOISE OPERATIONS ====================
    
    def add_gaussian_noise(self, data: np.ndarray, mean: float = 0, 
                          sigma: float = 0.1) -> np.ndarray:
        """
        Add Gaussian (normal) noise to the image.
        
        Args:
            data: Input array (2D slice or 3D volume)
            mean: Mean of the Gaussian distribution
            sigma: Standard deviation of the noise
            
        Returns:
            Noisy array
        """
        noise = np.random.normal(mean, sigma, data.shape)
        noisy_data = data + noise
        
        # Clip to valid range [0, 1]
        noisy_data = np.clip(noisy_data, 0, 1)
        
        logger.info(f"Added Gaussian noise: mean={mean}, sigma={sigma}")
        return noisy_data
    
    def add_salt_pepper_noise(self, data: np.ndarray, 
                             amount: float = 0.05) -> np.ndarray:
        """
        Add salt and pepper noise (random black and white pixels).
        
        Args:
            data: Input array
            amount: Proportion of pixels to affect (0.0 to 1.0)
            
        Returns:
            Noisy array
        """
        noisy_data = data.copy()
        
        # Salt (white pixels)
        num_salt = int(amount * data.size * 0.5)
        coords = [np.random.randint(0, i - 1, num_salt) for i in data.shape]
        noisy_data[tuple(coords)] = 1.0
        
        # Pepper (black pixels)
        num_pepper = int(amount * data.size * 0.5)
        coords = [np.random.randint(0, i - 1, num_pepper) for i in data.shape]
        noisy_data[tuple(coords)] = 0.0
        
        logger.info(f"Added salt & pepper noise: amount={amount}")
        return noisy_data
    
    def add_speckle_noise(self, data: np.ndarray, variance: float = 0.1) -> np.ndarray:
        """
        Add speckle (multiplicative) noise.
        
        Args:
            data: Input array
            variance: Variance of the speckle noise
            
        Returns:
            Noisy array
        """
        noise = np.random.randn(*data.shape) * variance
        noisy_data = data + data * noise
        noisy_data = np.clip(noisy_data, 0, 1)
        
        logger.info(f"Added speckle noise: variance={variance}")
        return noisy_data
    
    # ==================== FILTERING OPERATIONS ====================
    
    def gaussian_blur(self, data: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """
        Apply Gaussian blur filter.
        
        Args:
            data: Input array
            sigma: Standard deviation for Gaussian kernel
            
        Returns:
            Blurred array
        """
        blurred = ndimage.gaussian_filter(data, sigma=sigma)
        logger.info(f"Applied Gaussian blur: sigma={sigma}")
        return blurred
    
    def median_filter(self, data: np.ndarray, size: int = 3) -> np.ndarray:
        """
        Apply median filter (good for salt & pepper noise removal).
        
        Args:
            data: Input array
            size: Size of the filter kernel
            
        Returns:
            Filtered array
        """
        filtered = ndimage.median_filter(data, size=size)
        logger.info(f"Applied median filter: size={size}")
        return filtered
    
    def sharpen(self, data: np.ndarray, alpha: float = 1.5) -> np.ndarray:
        """
        Sharpen the image using unsharp masking.
        
        Args:
            data: Input array
            alpha: Sharpening strength
            
        Returns:
            Sharpened array
        """
        blurred = ndimage.gaussian_filter(data, sigma=1.0)
        sharpened = data + alpha * (data - blurred)
        sharpened = np.clip(sharpened, 0, 1)
        
        logger.info(f"Applied sharpening: alpha={alpha}")
        return sharpened
    
    def edge_detection(self, data: np.ndarray, method: str = 'sobel') -> np.ndarray:
        """
        Detect edges in the image.
        
        Args:
            data: Input array (2D slice only)
            method: Edge detection method ('sobel', 'canny', 'prewitt')
            
        Returns:
            Edge-detected array
        """
        if data.ndim != 2:
            raise ValueError("Edge detection only works on 2D slices")
        
        if method == 'sobel':
            edges = filters.sobel(data)
        elif method == 'canny':
            edges = filters.canny(data, sigma=1.0)
        elif method == 'prewitt':
            edges = filters.prewitt(data)
        else:
            raise ValueError(f"Unknown edge detection method: {method}")
        
        # Normalize to [0, 1]
        if edges.max() > 0:
            edges = edges / edges.max()
        
        logger.info(f"Applied edge detection: method={method}")
        return edges
    
    # ==================== GEOMETRIC TRANSFORMATIONS ====================
    
    def rotate(self, data: np.ndarray, angle: float, 
              axes: Tuple[int, int] = (0, 1)) -> np.ndarray:
        """
        Rotate the image.
        
        Args:
            data: Input array
            angle: Rotation angle in degrees
            axes: Axes defining the rotation plane
            
        Returns:
            Rotated array
        """
        rotated = ndimage.rotate(data, angle, axes=axes, reshape=False, order=1)
        logger.info(f"Rotated by {angle}° on axes {axes}")
        return rotated
    
    def scale(self, data: np.ndarray, factor: float) -> np.ndarray:
        """
        Scale (zoom) the image.
        
        Args:
            data: Input array
            factor: Scaling factor (>1 = zoom in, <1 = zoom out)
            
        Returns:
            Scaled array
        """
        scaled = ndimage.zoom(data, factor, order=1)
        
        # Crop or pad to original size
        if scaled.shape != data.shape:
            # Simple center crop/pad
            scaled = self._resize_to_shape(scaled, data.shape)
        
        logger.info(f"Scaled by factor {factor}")
        return scaled
    
    def flip(self, data: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Flip the image along an axis.
        
        Args:
            data: Input array
            axis: Axis to flip along
            
        Returns:
            Flipped array
        """
        flipped = np.flip(data, axis=axis)
        logger.info(f"Flipped along axis {axis}")
        return flipped
    
    def crop(self, data: np.ndarray, crop_size: Tuple[int, ...]) -> np.ndarray:
        """
        Center crop the image.
        
        Args:
            data: Input array
            crop_size: Desired output size
            
        Returns:
            Cropped array
        """
        starts = [(d - c) // 2 for d, c in zip(data.shape, crop_size)]
        slices = [slice(s, s + c) for s, c in zip(starts, crop_size)]
        cropped = data[tuple(slices)]
        
        logger.info(f"Cropped to size {crop_size}")
        return cropped
    
    # ==================== INTENSITY ADJUSTMENTS ====================
    
    def adjust_brightness(self, data: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """
        Adjust brightness.
        
        Args:
            data: Input array
            factor: Brightness factor (>1 = brighter, <1 = darker)
            
        Returns:
            Adjusted array
        """
        adjusted = data * factor
        adjusted = np.clip(adjusted, 0, 1)
        
        logger.info(f"Adjusted brightness: factor={factor}")
        return adjusted
    
    def adjust_contrast(self, data: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """
        Adjust contrast.
        
        Args:
            data: Input array
            factor: Contrast factor (>1 = more contrast, <1 = less)
            
        Returns:
            Adjusted array
        """
        mean = data.mean()
        adjusted = (data - mean) * factor + mean
        adjusted = np.clip(adjusted, 0, 1)
        
        logger.info(f"Adjusted contrast: factor={factor}")
        return adjusted
    
    def histogram_equalization(self, data: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization for better contrast.
        
        Args:
            data: Input array (2D slice)
            
        Returns:
            Equalized array
        """
        if data.ndim != 2:
            raise ValueError("Histogram equalization works on 2D slices")
        
        equalized = exposure.equalize_hist(data)
        logger.info("Applied histogram equalization")
        return equalized
    
    # ==================== HELPER METHODS ====================
    
    def _resize_to_shape(self, data: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Resize array to target shape by cropping or padding."""
        result = np.zeros(target_shape)
        
        # Calculate slicing for center crop/pad
        slices_src = []
        slices_tgt = []
        
        for d, t in zip(data.shape, target_shape):
            if d > t:
                # Crop
                start = (d - t) // 2
                slices_src.append(slice(start, start + t))
                slices_tgt.append(slice(None))
            else:
                # Pad
                start = (t - d) // 2
                slices_src.append(slice(None))
                slices_tgt.append(slice(start, start + d))
        
        result[tuple(slices_tgt)] = data[tuple(slices_src)]
        return result
    
    # ==================== BATCH OPERATIONS ====================
    
    def apply_pipeline(self, data: np.ndarray, 
                      operations: list) -> Tuple[np.ndarray, list]:
        """
        Apply a pipeline of transformations.
        
        Args:
            data: Input array
            operations: List of (method_name, kwargs) tuples
            
        Returns:
            Tuple of (transformed_data, intermediate_results)
        """
        result = data.copy()
        intermediates = [('original', data.copy())]
        
        for op_name, kwargs in operations:
            method = getattr(self, op_name)
            result = method(result, **kwargs)
            intermediates.append((op_name, result.copy()))
        
        logger.info(f"Applied pipeline with {len(operations)} operations")
        return result, intermediates


def main():
    """Example usage of the transformer."""
    print("MRI Transformer Module - Example Usage\n")
    
    # Create a synthetic test image (2D slice)
    test_slice = np.random.rand(256, 256) * 0.5 + 0.3
    print(f"Test slice shape: {test_slice.shape}")
    
    transformer = MRITransformer()
    
    # Test noise operations
    print("\n1. Adding noise...")
    noisy_gaussian = transformer.add_gaussian_noise(test_slice, sigma=0.1)
    noisy_sp = transformer.add_salt_pepper_noise(test_slice, amount=0.05)
    
    # Test filtering
    print("\n2. Applying filters...")
    blurred = transformer.gaussian_blur(test_slice, sigma=2.0)
    sharpened = transformer.sharpen(test_slice, alpha=1.5)
    
    # Test geometric transformations
    print("\n3. Applying geometric transformations...")
    rotated = transformer.rotate(test_slice, angle=45)
    flipped = transformer.flip(test_slice, axis=0)
    
    # Test intensity adjustments
    print("\n4. Adjusting intensity...")
    brighter = transformer.adjust_brightness(test_slice, factor=1.3)
    higher_contrast = transformer.adjust_contrast(test_slice, factor=1.5)
    
    # Test pipeline
    print("\n5. Applying transformation pipeline...")
    pipeline = [
        ('add_gaussian_noise', {'sigma': 0.05}),
        ('gaussian_blur', {'sigma': 1.0}),
        ('adjust_contrast', {'factor': 1.2})
    ]
    final, intermediates = transformer.apply_pipeline(test_slice, pipeline)
    print(f"Pipeline completed. Generated {len(intermediates)} intermediate results.")
    
    print("\n✓ All transformations completed successfully!")


if __name__ == "__main__":
    main()
