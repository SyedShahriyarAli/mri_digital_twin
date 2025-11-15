"""
MRI Comparison Module

Tools for comparing original vs transformed MRI images.
Provides metrics, difference maps, and side-by-side visualization.
"""

import numpy as np
import logging
from typing import Tuple, Dict
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MRIComparator:
    """Compare original and transformed MRI data."""
    
    def __init__(self):
        """Initialize the comparator."""
        logger.info("MRI Comparator initialized")
    
    def calculate_metrics(self, original: np.ndarray, 
                         transformed: np.ndarray) -> Dict[str, float]:
        """
        Calculate comparison metrics between two images.
        
        Args:
            original: Original image
            transformed: Transformed image
            
        Returns:
            Dictionary of metrics
        """
        if original.shape != transformed.shape:
            raise ValueError("Images must have the same shape")
        
        # For 3D volumes, work with flattened arrays for some metrics
        is_2d = original.ndim == 2
        
        metrics = {}
        
        # Mean Squared Error (MSE)
        metrics['mse'] = mean_squared_error(original, transformed)
        
        # Root Mean Squared Error (RMSE)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Peak Signal-to-Noise Ratio (PSNR)
        # Higher is better (typically 20-50 dB for good quality)
        metrics['psnr'] = peak_signal_noise_ratio(original, transformed, 
                                                   data_range=1.0)
        
        # Structural Similarity Index (SSIM)
        # Range: [-1, 1], where 1 means identical
        if is_2d:
            metrics['ssim'] = ssim(original, transformed, data_range=1.0)
        else:
            # For 3D, calculate mean SSIM across slices
            ssim_values = []
            for i in range(original.shape[2]):
                ssim_val = ssim(original[:, :, i], transformed[:, :, i], 
                              data_range=1.0)
                ssim_values.append(ssim_val)
            metrics['ssim'] = np.mean(ssim_values)
            metrics['ssim_std'] = np.std(ssim_values)
        
        # Mean Absolute Error (MAE)
        metrics['mae'] = np.mean(np.abs(original - transformed))
        
        # Maximum Absolute Difference
        metrics['max_diff'] = np.max(np.abs(original - transformed))
        
        # Normalized Cross-Correlation
        metrics['ncc'] = self._normalized_cross_correlation(original, transformed)
        
        # Percentage of changed pixels (threshold = 0.01)
        changed_pixels = np.sum(np.abs(original - transformed) > 0.01)
        total_pixels = original.size
        metrics['changed_pixels_pct'] = (changed_pixels / total_pixels) * 100
        
        logger.info(f"Calculated {len(metrics)} comparison metrics")
        return metrics
    
    def _normalized_cross_correlation(self, img1: np.ndarray, 
                                     img2: np.ndarray) -> float:
        """Calculate normalized cross-correlation."""
        img1_norm = (img1 - np.mean(img1)) / (np.std(img1) + 1e-8)
        img2_norm = (img2 - np.mean(img2)) / (np.std(img2) + 1e-8)
        ncc = np.mean(img1_norm * img2_norm)
        return float(ncc)
    
    def create_difference_map(self, original: np.ndarray, 
                            transformed: np.ndarray,
                            absolute: bool = True) -> np.ndarray:
        """
        Create a difference map showing changes.
        
        Args:
            original: Original image
            transformed: Transformed image
            absolute: Use absolute difference (True) or signed (False)
            
        Returns:
            Difference map
        """
        diff = transformed - original
        
        if absolute:
            diff = np.abs(diff)
        
        # Normalize to [0, 1] for visualization
        diff_normalized = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        
        logger.info("Created difference map")
        return diff_normalized
    
    def create_error_heatmap(self, original: np.ndarray, 
                           transformed: np.ndarray) -> np.ndarray:
        """
        Create an error heatmap (squared differences).
        
        Args:
            original: Original image
            transformed: Transformed image
            
        Returns:
            Error heatmap
        """
        error = (transformed - original) ** 2
        
        # Normalize for visualization
        error_normalized = (error - error.min()) / (error.max() - error.min() + 1e-8)
        
        logger.info("Created error heatmap")
        return error_normalized
    
    def create_overlay(self, original: np.ndarray, 
                      transformed: np.ndarray,
                      alpha: float = 0.5) -> np.ndarray:
        """
        Create an overlay blend of two images.
        
        Args:
            original: Original image
            transformed: Transformed image
            alpha: Blending factor (0 = original, 1 = transformed)
            
        Returns:
            Blended image
        """
        overlay = alpha * transformed + (1 - alpha) * original
        logger.info(f"Created overlay with alpha={alpha}")
        return overlay
    
    def get_statistics_comparison(self, original: np.ndarray,
                                 transformed: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compare statistical properties of two images.
        
        Args:
            original: Original image
            transformed: Transformed image
            
        Returns:
            Dictionary with statistics for both images
        """
        stats = {
            'original': {
                'min': float(np.min(original)),
                'max': float(np.max(original)),
                'mean': float(np.mean(original)),
                'std': float(np.std(original)),
                'median': float(np.median(original)),
                'q25': float(np.percentile(original, 25)),
                'q75': float(np.percentile(original, 75))
            },
            'transformed': {
                'min': float(np.min(transformed)),
                'max': float(np.max(transformed)),
                'mean': float(np.mean(transformed)),
                'std': float(np.std(transformed)),
                'median': float(np.median(transformed)),
                'q25': float(np.percentile(transformed, 25)),
                'q75': float(np.percentile(transformed, 75))
            }
        }
        
        logger.info("Generated statistics comparison")
        return stats
    
    def assess_quality(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Assess image quality based on metrics.
        
        Args:
            metrics: Dictionary of calculated metrics
            
        Returns:
            Dictionary with quality assessments
        """
        assessment = {}
        
        # PSNR assessment
        psnr = metrics.get('psnr', 0)
        if psnr > 40:
            assessment['psnr'] = 'Excellent (>40 dB)'
        elif psnr > 30:
            assessment['psnr'] = 'Good (30-40 dB)'
        elif psnr > 20:
            assessment['psnr'] = 'Fair (20-30 dB)'
        else:
            assessment['psnr'] = 'Poor (<20 dB)'
        
        # SSIM assessment
        ssim_val = metrics.get('ssim', 0)
        if ssim_val > 0.95:
            assessment['ssim'] = 'Excellent (>0.95)'
        elif ssim_val > 0.85:
            assessment['ssim'] = 'Good (0.85-0.95)'
        elif ssim_val > 0.70:
            assessment['ssim'] = 'Fair (0.70-0.85)'
        else:
            assessment['ssim'] = 'Poor (<0.70)'
        
        # Overall assessment
        if psnr > 35 and ssim_val > 0.90:
            assessment['overall'] = 'Excellent - Minimal degradation'
        elif psnr > 25 and ssim_val > 0.80:
            assessment['overall'] = 'Good - Acceptable quality'
        elif psnr > 20 and ssim_val > 0.70:
            assessment['overall'] = 'Fair - Noticeable degradation'
        else:
            assessment['overall'] = 'Poor - Significant degradation'
        
        logger.info("Generated quality assessment")
        return assessment


def main():
    """Example usage of the comparator."""
    print("MRI Comparator Module - Example Usage\n")
    
    # Create test images
    original = np.random.rand(256, 256) * 0.8 + 0.1
    
    # Create a slightly modified version
    noise = np.random.randn(256, 256) * 0.05
    transformed = np.clip(original + noise, 0, 1)
    
    print(f"Original shape: {original.shape}")
    print(f"Transformed shape: {transformed.shape}")
    
    comparator = MRIComparator()
    
    # Calculate metrics
    print("\n1. Calculating comparison metrics...")
    metrics = comparator.calculate_metrics(original, transformed)
    for key, value in metrics.items():
        print(f"   {key}: {value:.6f}")
    
    # Assess quality
    print("\n2. Quality assessment...")
    assessment = comparator.assess_quality(metrics)
    for key, value in assessment.items():
        print(f"   {key}: {value}")
    
    # Create difference map
    print("\n3. Creating difference map...")
    diff_map = comparator.create_difference_map(original, transformed)
    print(f"   Difference map shape: {diff_map.shape}")
    print(f"   Difference range: [{diff_map.min():.4f}, {diff_map.max():.4f}]")
    
    # Create error heatmap
    print("\n4. Creating error heatmap...")
    error_map = comparator.create_error_heatmap(original, transformed)
    print(f"   Error heatmap shape: {error_map.shape}")
    
    # Get statistics comparison
    print("\n5. Statistical comparison...")
    stats = comparator.get_statistics_comparison(original, transformed)
    print("   Original statistics:")
    for key, value in stats['original'].items():
        print(f"      {key}: {value:.4f}")
    print("   Transformed statistics:")
    for key, value in stats['transformed'].items():
        print(f"      {key}: {value:.4f}")
    
    print("\n✓ All comparison operations completed successfully!")


if __name__ == "__main__":
    main()
