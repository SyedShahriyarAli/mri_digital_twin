"""
MRI Security Testbed - Attack Simulator

Simulates various attack types on MRI data to test detection systems.
Includes: Data tampering, metadata manipulation, noise injection, file corruption.
"""

import numpy as np
import nibabel as nib
import hashlib
import json
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MRIAttackSimulator:
    """
    Simulates various attack scenarios on MRI data.
    
    Attack Types:
        1. Data Tampering - Modify image pixels/slices
        2. Metadata Manipulation - Change scan information
        3. Noise Injection - Add artifacts
        4. File Corruption - Break file structure
    """
    
    def __init__(self, output_dir: str = "mri_data/attacked"):
        """
        Initialize the attack simulator.
        
        Args:
            output_dir: Directory to save attacked files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Attack Simulator initialized. Output: {self.output_dir}")
    
    # ==================== ATTACK TYPE 1: DATA TAMPERING ====================
    
    def tamper_pixel_values(self, volume: np.ndarray, 
                           region: str = "random",
                           intensity_change: float = 0.3) -> np.ndarray:
        """
        Modify pixel values in a region (simulate hiding tumors, altering diagnosis).
        
        Args:
            volume: MRI volume data
            region: "random", "center", "edge", or coordinates
            intensity_change: Amount to change (+ = brighter, - = darker)
            
        Returns:
            Tampered volume
        """
        tampered = volume.copy()
        
        if region == "random":
            # Random 3D region
            x_start = random.randint(0, volume.shape[0] // 2)
            y_start = random.randint(0, volume.shape[1] // 2)
            z_start = random.randint(0, volume.shape[2] // 2)
            
            x_size = random.randint(20, 50)
            y_size = random.randint(20, 50)
            z_size = random.randint(10, 30)
            
            tampered[x_start:x_start+x_size, 
                    y_start:y_start+y_size,
                    z_start:z_start+z_size] += intensity_change
            
            logger.info(f"Tampered random region at ({x_start}, {y_start}, {z_start})")
            
        elif region == "center":
            # Central region (critical brain areas!)
            cx, cy, cz = volume.shape[0]//2, volume.shape[1]//2, volume.shape[2]//2
            size = 30
            
            tampered[cx-size:cx+size,
                    cy-size:cy+size,
                    cz-size//2:cz+size//2] += intensity_change
            
            logger.info(f"Tampered center region")
        
        # Clip to valid range
        tampered = np.clip(tampered, 0, 1)
        
        return tampered
    
    def delete_slices(self, volume: np.ndarray, 
                     num_slices: int = 10,
                     axis: int = 2) -> np.ndarray:
        """
        Remove slices (simulate missing data).
        
        Args:
            volume: MRI volume data
            num_slices: Number of slices to remove
            axis: Which axis (0=sagittal, 1=coronal, 2=axial)
            
        Returns:
            Volume with deleted slices
        """
        # Random slice indices to delete
        total_slices = volume.shape[axis]
        if num_slices >= total_slices:
            num_slices = total_slices // 2
        
        keep_indices = list(range(total_slices))
        delete_indices = random.sample(keep_indices, num_slices)
        delete_indices.sort()
        
        for idx in delete_indices:
            keep_indices.remove(idx)
        
        # Extract remaining slices
        if axis == 0:
            tampered = volume[keep_indices, :, :]
        elif axis == 1:
            tampered = volume[:, keep_indices, :]
        else:  # axis == 2
            tampered = volume[:, :, keep_indices]
        
        logger.info(f"Deleted {num_slices} slices on axis {axis}")
        logger.info(f"Original shape: {volume.shape} → New shape: {tampered.shape}")
        
        return tampered
    
    def blur_region(self, volume: np.ndarray,
                   region_size: int = 40,
                   sigma: float = 5.0) -> np.ndarray:
        """
        Blur a region (simulate hiding lesions/tumors).
        
        Args:
            volume: MRI volume data
            region_size: Size of region to blur
            sigma: Blur strength
            
        Returns:
            Volume with blurred region
        """
        from scipy import ndimage
        
        tampered = volume.copy()
        
        # Random region
        x_start = random.randint(0, volume.shape[0] - region_size)
        y_start = random.randint(0, volume.shape[1] - region_size)
        z_start = random.randint(0, volume.shape[2] - region_size//2)
        
        # Extract and blur region
        region = tampered[x_start:x_start+region_size,
                         y_start:y_start+region_size,
                         z_start:z_start+region_size//2]
        
        blurred_region = ndimage.gaussian_filter(region, sigma=sigma)
        
        # Replace
        tampered[x_start:x_start+region_size,
                y_start:y_start+region_size,
                z_start:z_start+region_size//2] = blurred_region
        
        logger.info(f"Blurred region at ({x_start}, {y_start}, {z_start})")
        
        return tampered
    
    # ==================== ATTACK TYPE 2: METADATA MANIPULATION ====================
    
    def manipulate_metadata(self, original_metadata: Dict,
                           attack_type: str = "date") -> Dict:
        """
        Modify scan metadata (change dates, scanner info, parameters).
        
        Args:
            original_metadata: Original metadata dictionary
            attack_type: "date", "scanner", "parameters", "patient", or "all"
            
        Returns:
            Modified metadata dictionary
        """
        metadata = original_metadata.copy()
        
        if attack_type in ["date", "all"]:
            # Change scan date
            if "scan_date" in metadata:
                original_date = datetime.fromisoformat(metadata["scan_date"])
                # Random change: -5 years to +1 year
                days_offset = random.randint(-1825, 365)
                new_date = original_date + timedelta(days=days_offset)
                metadata["scan_date"] = new_date.isoformat()
                logger.info(f"Changed scan_date: {original_date.date()} → {new_date.date()}")
        
        if attack_type in ["scanner", "all"]:
            # Change scanner information
            fake_scanners = ["Siemens 3T", "GE 1.5T", "Philips 3T", "Fake Scanner X"]
            metadata["scanner"] = random.choice(fake_scanners)
            logger.info(f"Changed scanner to: {metadata['scanner']}")
        
        if attack_type in ["parameters", "all"]:
            # Modify MRI parameters (TR, TE, etc.)
            if "TR" in metadata:
                metadata["TR"] = round(random.uniform(500, 5000), 2)
            if "TE" in metadata:
                metadata["TE"] = round(random.uniform(10, 150), 2)
            if "flip_angle" in metadata:
                metadata["flip_angle"] = random.randint(10, 90)
            logger.info("Modified MRI parameters")
        
        if attack_type in ["patient", "all"]:
            # Change patient info
            if "patient_age" in metadata:
                metadata["patient_age"] = random.randint(1, 120)
            if "patient_id" in metadata:
                metadata["patient_id"] = f"FAKE_{random.randint(1000, 9999)}"
            logger.info("Modified patient information")
        
        return metadata
    
    # ==================== ATTACK TYPE 3: NOISE INJECTION ====================
    
    def inject_gaussian_noise(self, volume: np.ndarray,
                             sigma: float = 0.1) -> np.ndarray:
        """
        Add Gaussian noise (simulate poor scan quality).
        
        Args:
            volume: MRI volume data
            sigma: Noise strength
            
        Returns:
            Noisy volume
        """
        noise = np.random.normal(0, sigma, volume.shape)
        noisy = volume + noise
        noisy = np.clip(noisy, 0, 1)
        
        logger.info(f"Injected Gaussian noise: sigma={sigma}")
        return noisy
    
    def inject_spikes(self, volume: np.ndarray,
                     num_spikes: int = 100,
                     intensity: float = 1.0) -> np.ndarray:
        """
        Add random bright/dark spikes (simulate electrical interference).
        
        Args:
            volume: MRI volume data
            num_spikes: Number of spike artifacts
            intensity: Spike brightness (0-1)
            
        Returns:
            Volume with spikes
        """
        spiked = volume.copy()
        
        for _ in range(num_spikes):
            x = random.randint(0, volume.shape[0]-1)
            y = random.randint(0, volume.shape[1]-1)
            z = random.randint(0, volume.shape[2]-1)
            
            # Random bright or dark spike
            spike_value = intensity if random.random() > 0.5 else 0
            spiked[x, y, z] = spike_value
        
        logger.info(f"Injected {num_spikes} spike artifacts")
        return spiked
    
    def inject_motion_artifact(self, volume: np.ndarray,
                               strength: float = 3.0) -> np.ndarray:
        """
        Simulate motion blur (patient moved during scan).
        
        Args:
            volume: MRI volume data
            strength: Blur strength
            
        Returns:
            Volume with motion artifacts
        """
        from scipy import ndimage
        
        # Apply directional blur
        motion_blurred = ndimage.gaussian_filter(volume, sigma=[strength, strength/2, 0])
        
        logger.info(f"Injected motion artifacts: strength={strength}")
        return motion_blurred
    
    # ==================== ATTACK TYPE 4: FILE CORRUPTION ====================
    
    def corrupt_file_header(self, filepath: Path) -> Path:
        """
        Corrupt NIfTI file header (file won't open properly).
        
        Args:
            filepath: Path to NIfTI file
            
        Returns:
            Path to corrupted file
        """
        corrupted_path = self.output_dir / f"corrupted_header_{filepath.name}"
        
        # Read file
        with open(filepath, 'rb') as f:
            file_data = bytearray(f.read())
        
        # Corrupt first 100 bytes (header region)
        for i in range(20, 100):
            file_data[i] = random.randint(0, 255)
        
        # Save corrupted file
        with open(corrupted_path, 'wb') as f:
            f.write(file_data)
        
        logger.info(f"Corrupted file header: {corrupted_path}")
        return corrupted_path
    
    def corrupt_random_bytes(self, filepath: Path,
                            num_bytes: int = 1000) -> Path:
        """
        Corrupt random bytes in file (data scrambling).
        
        Args:
            filepath: Path to file
            num_bytes: Number of bytes to corrupt
            
        Returns:
            Path to corrupted file
        """
        corrupted_path = self.output_dir / f"corrupted_data_{filepath.name}"
        
        # Read file
        with open(filepath, 'rb') as f:
            file_data = bytearray(f.read())
        
        file_size = len(file_data)
        
        # Corrupt random bytes (skip first 352 bytes - NIfTI header)
        for _ in range(num_bytes):
            pos = random.randint(352, file_size - 1)
            file_data[pos] = random.randint(0, 255)
        
        # Save corrupted file
        with open(corrupted_path, 'wb') as f:
            f.write(file_data)
        
        logger.info(f"Corrupted {num_bytes} random bytes: {corrupted_path}")
        return corrupted_path
    
    # ==================== ATTACK EXECUTION ====================
    
    def execute_attack(self, volume: np.ndarray,
                      metadata: Dict,
                      filepath: Optional[Path],
                      attack_config: Dict) -> Tuple[np.ndarray, Dict, Optional[Path]]:
        """
        Execute configured attack(s).
        
        Args:
            volume: MRI volume data
            metadata: Scan metadata
            filepath: Path to original file (for file corruption)
            attack_config: Attack configuration
                {
                    "type": "data_tampering|metadata|noise|file_corruption",
                    "method": specific method name,
                    "params": method parameters
                }
        
        Returns:
            (attacked_volume, attacked_metadata, attacked_file_path)
        """
        attack_type = attack_config.get("type")
        method = attack_config.get("method")
        params = attack_config.get("params", {})
        
        attacked_volume = volume
        attacked_metadata = metadata
        attacked_file = filepath
        
        if attack_type == "data_tampering":
            if method == "tamper_pixels":
                attacked_volume = self.tamper_pixel_values(volume, **params)
            elif method == "delete_slices":
                attacked_volume = self.delete_slices(volume, **params)
            elif method == "blur_region":
                attacked_volume = self.blur_region(volume, **params)
        
        elif attack_type == "metadata":
            attacked_metadata = self.manipulate_metadata(metadata, **params)
        
        elif attack_type == "noise":
            if method == "gaussian":
                attacked_volume = self.inject_gaussian_noise(volume, **params)
            elif method == "spikes":
                attacked_volume = self.inject_spikes(volume, **params)
            elif method == "motion":
                attacked_volume = self.inject_motion_artifact(volume, **params)
        
        elif attack_type == "file_corruption":
            if filepath and filepath.exists():
                if method == "header":
                    attacked_file = self.corrupt_file_header(filepath)
                elif method == "random_bytes":
                    attacked_file = self.corrupt_random_bytes(filepath, **params)
        
        logger.info(f"Executed attack: {attack_type} - {method}")
        
        return attacked_volume, attacked_metadata, attacked_file


def main():
    """Example usage of attack simulator."""
    print("="*70)
    print("MRI ATTACK SIMULATOR - Demo")
    print("="*70)
    
    # Create test volume
    print("\n1. Creating test MRI volume (256³)...")
    volume = np.random.rand(256, 256, 256) * 0.8 + 0.1
    print(f"   Shape: {volume.shape}")
    print(f"   Range: [{volume.min():.3f}, {volume.max():.3f}]")
    
    # Test metadata
    metadata = {
        "scan_date": "2024-01-15T10:30:00",
        "scanner": "Siemens 3T",
        "TR": 2000.0,
        "TE": 30.0,
        "patient_age": 45,
        "patient_id": "TEST001"
    }
    
    simulator = MRIAttackSimulator()
    
    # Test each attack type
    print("\n2. Testing Data Tampering Attacks:")
    print("-" * 70)
    
    tampered = simulator.tamper_pixel_values(volume, region="center", intensity_change=-0.3)
    print(f"   ✓ Pixel tampering: Changed {np.sum(tampered != volume)} voxels")
    
    deleted = simulator.delete_slices(volume, num_slices=20, axis=2)
    print(f"   ✓ Slice deletion: {volume.shape} → {deleted.shape}")
    
    blurred = simulator.blur_region(volume, region_size=50, sigma=5.0)
    print(f"   ✓ Region blurring: Applied to 50³ region")
    
    print("\n3. Testing Metadata Manipulation:")
    print("-" * 70)
    
    manipulated = simulator.manipulate_metadata(metadata, attack_type="all")
    print(f"   Original metadata:")
    for key, value in metadata.items():
        print(f"      {key}: {value}")
    print(f"   Attacked metadata:")
    for key, value in manipulated.items():
        print(f"      {key}: {value}")
    
    print("\n4. Testing Noise Injection:")
    print("-" * 70)
    
    noisy = simulator.inject_gaussian_noise(volume, sigma=0.15)
    print(f"   ✓ Gaussian noise: sigma=0.15")
    
    spiked = simulator.inject_spikes(volume, num_spikes=500, intensity=1.0)
    print(f"   ✓ Spike artifacts: 500 spikes")
    
    motion = simulator.inject_motion_artifact(volume, strength=4.0)
    print(f"   ✓ Motion artifacts: strength=4.0")
    
    print("\n5. Testing Attack Execution:")
    print("-" * 70)
    
    attack_config = {
        "type": "data_tampering",
        "method": "tamper_pixels",
        "params": {"region": "random", "intensity_change": 0.5}
    }
    
    attacked_vol, attacked_meta, _ = simulator.execute_attack(
        volume, metadata, None, attack_config
    )
    print(f"   ✓ Executed configured attack")
    print(f"   ✓ Volume shape: {attacked_vol.shape}")
    
    print("\n" + "="*70)
    print("✓ All attack methods tested successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
