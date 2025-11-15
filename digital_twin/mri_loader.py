"""
MRI Data Loader Module

Handles loading and preprocessing of MRI data from various formats.
Supports .nii.gz (NIfTI) and .npy (NumPy) file formats.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MRILoader:
    """Loads and preprocesses MRI data from multiple file formats."""
    
    def __init__(self, data_dir: str = "mri_data/preprocessed"):
        """
        Initialize the MRI loader.
        
        Args:
            data_dir: Base directory containing preprocessed MRI data
        """
        self.data_dir = Path(data_dir)
        self.nifti_dir = self.data_dir / "nifti"
        self.numpy_dir = self.data_dir / "numpy"
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        logger.info(f"MRI Loader initialized with data directory: {self.data_dir}")
    
    def list_available_scans(self) -> Dict[str, list]:
        """
        List all available MRI scans in the preprocessed directory.
        
        Returns:
            Dictionary with 'nifti' and 'numpy' keys containing lists of available files
        """
        scans = {"nifti": [], "numpy": []}
        
        # List NIfTI files
        if self.nifti_dir.exists():
            scans["nifti"] = sorted([
                f.stem.replace(".nii", "") 
                for f in self.nifti_dir.glob("*.nii.gz")
            ])
        
        # List NumPy files
        if self.numpy_dir.exists():
            scans["numpy"] = sorted([
                f.stem for f in self.numpy_dir.glob("*.npy")
            ])
        
        logger.info(f"Found {len(scans['nifti'])} NIfTI and {len(scans['numpy'])} NumPy scans")
        return scans
    
    def load_nifti(self, filename: str) -> np.ndarray:
        """
        Load a NIfTI (.nii.gz) file.
        
        Args:
            filename: Name of the file (with or without extension)
            
        Returns:
            3D numpy array containing the MRI volume
        """
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("nibabel is required to load NIfTI files. Install with: pip install nibabel")
        
        # Handle filename with or without extension
        if not filename.endswith(".nii.gz"):
            filename = f"{filename}.nii.gz"
        
        filepath = self.nifti_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"NIfTI file not found: {filepath}")
        
        logger.info(f"Loading NIfTI file: {filepath}")
        nii_img = nib.load(str(filepath))
        data = nii_img.get_fdata()
        
        logger.info(f"Loaded NIfTI with shape: {data.shape}")
        return data
    
    def load_numpy(self, filename: str) -> np.ndarray:
        """
        Load a NumPy (.npy) file.
        
        Args:
            filename: Name of the file (with or without extension)
            
        Returns:
            3D numpy array containing the MRI volume
        """
        # Handle filename with or without extension
        if not filename.endswith(".npy"):
            filename = f"{filename}.npy"
        
        filepath = self.numpy_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"NumPy file not found: {filepath}")
        
        logger.info(f"Loading NumPy file: {filepath}")
        data = np.load(str(filepath))
        
        logger.info(f"Loaded NumPy array with shape: {data.shape}")
        return data
    
    def load_mri(self, filename: str, format: str = "auto") -> np.ndarray:
        """
        Load MRI data from file, automatically detecting or using specified format.
        
        Args:
            filename: Name of the file (with or without extension)
            format: File format ('nifti', 'numpy', or 'auto')
            
        Returns:
            3D numpy array containing the MRI volume
        """
        if format == "auto":
            # Try to detect format from filename
            if ".nii" in filename or filename in self.list_available_scans()["nifti"]:
                format = "nifti"
            elif ".npy" in filename or filename in self.list_available_scans()["numpy"]:
                format = "numpy"
            else:
                # Try nifti first, then numpy
                try:
                    return self.load_nifti(filename)
                except FileNotFoundError:
                    return self.load_numpy(filename)
        
        if format == "nifti":
            return self.load_nifti(filename)
        elif format == "numpy":
            return self.load_numpy(filename)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'nifti', 'numpy', or 'auto'")
    
    def normalize_volume(self, volume: np.ndarray, method: str = "minmax") -> np.ndarray:
        """
        Normalize MRI volume data to [0, 1] range.
        
        Args:
            volume: Input MRI volume
            method: Normalization method ('minmax' or 'percentile')
            
        Returns:
            Normalized volume
        """
        if method == "minmax":
            # Simple min-max normalization
            vol_min = np.min(volume)
            vol_max = np.max(volume)
            if vol_max - vol_min > 0:
                normalized = (volume - vol_min) / (vol_max - vol_min)
            else:
                normalized = np.zeros_like(volume)
        
        elif method == "percentile":
            # Percentile-based normalization (robust to outliers)
            p2, p98 = np.percentile(volume, [2, 98])
            normalized = np.clip((volume - p2) / (p98 - p2), 0, 1)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        logger.info(f"Normalized volume using {method} method")
        return normalized
    
    def extract_middle_slices(self, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract middle slices from all three anatomical views.
        
        Args:
            volume: 3D MRI volume (assumed to be in RAS orientation)
            
        Returns:
            Dictionary with 'axial', 'sagittal', and 'coronal' keys
        """
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape: {volume.shape}")
        
        # Get middle indices
        mid_x = volume.shape[0] // 2
        mid_y = volume.shape[1] // 2
        mid_z = volume.shape[2] // 2
        
        slices = {
            "axial": volume[:, :, mid_z],      # Horizontal (top view)
            "sagittal": volume[mid_x, :, :],   # Side view
            "coronal": volume[:, mid_y, :]     # Front view
        }
        
        logger.info(f"Extracted middle slices - Axial: {mid_z}, Sagittal: {mid_x}, Coronal: {mid_y}")
        return slices
    
    def load_and_prepare(self, filename: str, format: str = "auto", 
                        normalize: bool = False) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Load MRI data and prepare it for visualization.
        
        Args:
            filename: Name of the file to load
            format: File format ('nifti', 'numpy', or 'auto')
            normalize: Whether to normalize the volume
            
        Returns:
            Tuple of (normalized_volume, middle_slices_dict)
        """
        # Load the volume
        volume = self.load_mri(filename, format)
        
        # Normalize if requested
        if normalize:
            volume = self.normalize_volume(volume, method="percentile")
        
        # Extract middle slices
        slices = self.extract_middle_slices(volume)
        
        return volume, slices


def main():
    """Example usage of the MRI loader."""
    try:
        loader = MRILoader()
        
        # List available scans
        scans = loader.list_available_scans()
        print("Available scans:")
        print(f"  NIfTI: {scans['nifti']}")
        print(f"  NumPy: {scans['numpy']}")
        
        # Try to load the first available scan
        if scans['nifti']:
            volume, slices = loader.load_and_prepare(scans['nifti'][0])
            print(f"\nLoaded volume shape: {volume.shape}")
            print(f"Slice shapes:")
            for view, slice_data in slices.items():
                print(f"  {view}: {slice_data.shape}")
        
        elif scans['numpy']:
            volume, slices = loader.load_and_prepare(scans['numpy'][0], format='numpy')
            print(f"\nLoaded volume shape: {volume.shape}")
            print(f"Slice shapes:")
            for view, slice_data in slices.items():
                print(f"  {view}: {slice_data.shape}")
        
        else:
            print("No scans found in the preprocessed directory.")
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
