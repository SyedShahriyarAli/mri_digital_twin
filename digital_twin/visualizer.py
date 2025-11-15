"""
MRI Visualizer Module

Handles visualization and slice extraction from MRI volumes.
Generates slice images for display and saves them as PNG files.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MRIVisualizer:
    """Visualizes MRI data by extracting and saving slices."""
    
    def __init__(self, output_dir: str = "static/slices"):
        """
        Initialize the MRI visualizer.
        
        Args:
            output_dir: Directory to save slice images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = 150  # High DPI for quality
        logger.info(f"MRI Visualizer initialized with output directory: {self.output_dir}")
    
    def get_slice(self, volume: np.ndarray, view: str, index: int) -> np.ndarray:
        """
        Extract a single slice from the volume along the specified view.
        
        Args:
            volume: 3D MRI volume
            view: View orientation ('axial', 'sagittal', or 'coronal')
            index: Slice index to extract
            
        Returns:
            2D numpy array containing the slice
        """
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape: {volume.shape}")
        
        view = view.lower()
        
        if view == "axial":
            if index < 0 or index >= volume.shape[2]:
                raise ValueError(f"Axial index {index} out of range [0, {volume.shape[2]-1}]")
            return volume[:, :, index]
        
        elif view == "sagittal":
            if index < 0 or index >= volume.shape[0]:
                raise ValueError(f"Sagittal index {index} out of range [0, {volume.shape[0]-1}]")
            return volume[index, :, :]
        
        elif view == "coronal":
            if index < 0 or index >= volume.shape[1]:
                raise ValueError(f"Coronal index {index} out of range [0, {volume.shape[1]-1}]")
            return volume[:, index, :]
        
        else:
            raise ValueError(f"Unknown view: {view}. Use 'axial', 'sagittal', or 'coronal'")
    
    def get_all_slices(self, volume: np.ndarray, view: str) -> np.ndarray:
        """
        Extract all slices from the volume along the specified view.
        
        Args:
            volume: 3D MRI volume
            view: View orientation ('axial', 'sagittal', or 'coronal')
            
        Returns:
            3D numpy array where first dimension is the slice index
        """
        view = view.lower()
        
        if view == "axial":
            # Transpose to get slices as first dimension
            return np.transpose(volume, (2, 0, 1))
        elif view == "sagittal":
            return np.transpose(volume, (0, 1, 2))
        elif view == "coronal":
            return np.transpose(volume, (1, 0, 2))
        else:
            raise ValueError(f"Unknown view: {view}")
    
    def get_volume_dimensions(self, volume: np.ndarray) -> Dict[str, int]:
        """
        Get the number of slices for each view orientation.
        
        Args:
            volume: 3D MRI volume
            
        Returns:
            Dictionary with slice counts for each view
        """
        return {
            "axial": volume.shape[2],
            "sagittal": volume.shape[0],
            "coronal": volume.shape[1]
        }
    
    def save_slice_as_png(self, slice_data: np.ndarray, filename: str, 
                         cmap: str = "gray", apply_rotation: bool = True) -> str:
        """
        Save a slice as a PNG image file.
        
        Args:
            slice_data: 2D numpy array containing the slice
            filename: Output filename (without extension)
            cmap: Matplotlib colormap to use
            apply_rotation: Whether to rotate the image for proper orientation
            
        Returns:
            Path to the saved file
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for saving images. Install with: pip install matplotlib")
        
        # Log slice statistics for debugging
        logger.debug(f"Slice stats before norm - min: {slice_data.min():.4f}, max: {slice_data.max():.4f}, mean: {slice_data.mean():.4f}")
        
        # Normalize slice data to [0, 1] range for display
        slice_min = slice_data.min()
        slice_max = slice_data.max()
        
        if slice_max - slice_min > 1e-6:  # Avoid division by very small numbers
            # Normalize to 0-1 range
            slice_data = (slice_data - slice_min) / (slice_max - slice_min)
        else:
            # If slice is uniform, set to mid-gray
            logger.warning(f"Slice {filename} has no variation (min={slice_min}, max={slice_max})")
            slice_data = np.ones_like(slice_data) * 0.5
        
        logger.debug(f"Slice stats after norm - min: {slice_data.min():.4f}, max: {slice_data.max():.4f}")
        
        # Rotate for proper orientation (optional)
        if apply_rotation:
            slice_data = np.rot90(slice_data)
        
        # Calculate optimal figure size based on slice dimensions
        # Use slice dimensions directly for 1:1 pixel mapping
        height, width = slice_data.shape
        dpi = self.dpi  # Use instance DPI setting
        
        # Calculate figure size in inches (size / dpi)
        figsize = (width / dpi, height / dpi)
        
        # Create figure with exact dimensions
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])  # Fill entire figure
        
        # Display image with proper normalization and NO interpolation
        # Use 'nearest' for pixel-perfect rendering
        im = ax.imshow(slice_data, cmap=cmap, interpolation='nearest', 
                      vmin=0, vmax=1, aspect='equal')
        ax.axis('off')
        
        # Save the image
        if not filename.endswith('.png'):
            filename = f"{filename}.png"
        
        output_path = self.output_dir / filename
        
        # Save with high DPI and no compression for maximum quality
        plt.savefig(output_path, dpi=dpi, facecolor='black', 
                   edgecolor='none', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        logger.info(f"Saved slice to: {output_path}")
        return str(output_path)
    
    def generate_view_slices(self, volume: np.ndarray, view: str, 
                           scan_name: str, indices: Optional[list] = None) -> Dict[int, str]:
        """
        Generate and save multiple slice images for a specific view.
        
        Args:
            volume: 3D MRI volume
            view: View orientation ('axial', 'sagittal', or 'coronal')
            scan_name: Name of the scan (used in filename)
            indices: List of slice indices to generate (None = all slices)
            
        Returns:
            Dictionary mapping slice indices to file paths
        """
        dimensions = self.get_volume_dimensions(volume)
        max_slices = dimensions[view.lower()]
        
        if indices is None:
            # Generate all slices (may be memory intensive)
            indices = list(range(0, max_slices, max(1, max_slices // 50)))  # Max 50 slices
        
        saved_files = {}
        
        for idx in indices:
            if 0 <= idx < max_slices:
                slice_data = self.get_slice(volume, view, idx)
                filename = f"{scan_name}_{view}_{idx:03d}.png"
                filepath = self.save_slice_as_png(slice_data, filename)
                saved_files[idx] = filepath
        
        logger.info(f"Generated {len(saved_files)} slices for {view} view")
        return saved_files
    
    def generate_all_views(self, volume: np.ndarray, scan_name: str, 
                          middle_only: bool = True) -> Dict[str, Dict]:
        """
        Generate slice images for all three views.
        
        Args:
            volume: 3D MRI volume
            scan_name: Name of the scan (used in filename)
            middle_only: If True, generate only middle slices; otherwise generate multiple slices
            
        Returns:
            Dictionary with view names as keys and slice info as values
        """
        views = ["axial", "sagittal", "coronal"]
        dimensions = self.get_volume_dimensions(volume)
        results = {}
        
        for view in views:
            if middle_only:
                # Generate only the middle slice
                middle_idx = dimensions[view] // 2
                indices = [middle_idx]
            else:
                # Generate evenly spaced slices
                max_slices = dimensions[view]
                num_samples = min(20, max_slices)  # Generate up to 20 slices
                indices = [int(i * max_slices / num_samples) for i in range(num_samples)]
            
            saved_files = self.generate_view_slices(volume, view, scan_name, indices)
            
            results[view] = {
                "files": saved_files,
                "total_slices": dimensions[view],
                "middle_slice": dimensions[view] // 2
            }
        
        return results
    
    def create_slice_array(self, slice_data: np.ndarray, apply_rotation: bool = True) -> np.ndarray:
        """
        Prepare slice data as a numpy array for display (convert to 0-255 uint8).
        
        Args:
            slice_data: 2D numpy array containing the slice
            apply_rotation: Whether to rotate the image for proper orientation
            
        Returns:
            2D numpy array with values in [0, 255] range
        """
        # Ensure data is in [0, 1] range
        if slice_data.max() > 1.0 or slice_data.min() < 0.0:
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
        
        # Rotate for proper orientation (optional)
        if apply_rotation:
            slice_data = np.rot90(slice_data)
        
        # Convert to 0-255 range
        slice_array = (slice_data * 255).astype(np.uint8)
        
        return slice_array
    
    def clear_output_directory(self):
        """Remove all files from the output directory."""
        if self.output_dir.exists():
            for file in self.output_dir.glob("*.png"):
                file.unlink()
            logger.info(f"Cleared output directory: {self.output_dir}")


def main():
    """Example usage of the MRI visualizer."""
    try:
        from mri_loader import MRILoader
        
        # Load a sample MRI scan
        loader = MRILoader()
        scans = loader.list_available_scans()
        
        if not scans['nifti'] and not scans['numpy']:
            print("No scans available for visualization.")
            return
        
        # Load first available scan
        if scans['nifti']:
            volume, _ = loader.load_and_prepare(scans['nifti'][0])
            scan_name = scans['nifti'][0]
        else:
            volume, _ = loader.load_and_prepare(scans['numpy'][0], format='numpy')
            scan_name = scans['numpy'][0]
        
        # Create visualizer
        visualizer = MRIVisualizer()
        
        # Get volume dimensions
        dims = visualizer.get_volume_dimensions(volume)
        print(f"Volume dimensions:")
        for view, count in dims.items():
            print(f"  {view}: {count} slices")
        
        # Generate middle slices for all views
        print("\nGenerating visualization slices...")
        results = visualizer.generate_all_views(volume, scan_name, middle_only=True)
        
        print("\nGenerated slices:")
        for view, info in results.items():
            print(f"  {view}: {len(info['files'])} files, middle at index {info['middle_slice']}")
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
