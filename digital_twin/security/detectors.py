"""
MRI Security Testbed - Detection System

Implements three detection methods to catch attacks on MRI data:
1. Hash Verification (file integrity)
2. Anomaly Detection (image differences)
3. Metadata Validation (logical checks)
"""

import numpy as np
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MRISecurityDetector:
    """
    Detect attacks on MRI data using multiple methods.
    
    Detection Methods:
        1. Hash Verification - Check file integrity
        2. Anomaly Detection - Compare image differences
        3. Metadata Validation - Validate logical consistency
    """
    
    def __init__(self):
        """Initialize the security detector."""
        logger.info("MRI Security Detector initialized")
    
    # ==================== METHOD 1: HASH VERIFICATION ====================
    
    def calculate_file_hash(self, filepath: Path, algorithm: str = "sha256") -> str:
        """
        Calculate cryptographic hash of file.
        
        Args:
            filepath: Path to file
            algorithm: Hash algorithm ('sha256', 'md5', 'sha512')
            
        Returns:
            Hex digest of hash
        """
        if algorithm == "sha256":
            hasher = hashlib.sha256()
        elif algorithm == "md5":
            hasher = hashlib.md5()
        elif algorithm == "sha512":
            hasher = hashlib.sha512()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        with open(filepath, 'rb') as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        
        file_hash = hasher.hexdigest()
        logger.info(f"Calculated {algorithm} hash: {file_hash[:16]}...")
        
        return file_hash
    
    def calculate_volume_hash(self, volume: np.ndarray, algorithm: str = "sha256") -> str:
        """
        Calculate hash of volume data (in-memory).
        
        Args:
            volume: MRI volume data
            algorithm: Hash algorithm
            
        Returns:
            Hex digest of hash
        """
        if algorithm == "sha256":
            hasher = hashlib.sha256()
        elif algorithm == "md5":
            hasher = hashlib.md5()
        elif algorithm == "sha512":
            hasher = hashlib.sha512()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Hash the volume data
        hasher.update(volume.tobytes())
        
        volume_hash = hasher.hexdigest()
        logger.info(f"Calculated volume {algorithm} hash: {volume_hash[:16]}...")
        
        return volume_hash
    
    def verify_file_integrity(self, filepath: Path, 
                             expected_hash: str,
                             algorithm: str = "sha256") -> Dict:
        """
        Verify file hasn't been tampered with.
        
        Args:
            filepath: Path to file to check
            expected_hash: Known good hash
            algorithm: Hash algorithm used
            
        Returns:
            Detection result dictionary
        """
        current_hash = self.calculate_file_hash(filepath, algorithm)
        
        is_intact = (current_hash == expected_hash)
        
        result = {
            "method": "hash_verification",
            "algorithm": algorithm,
            "expected_hash": expected_hash,
            "current_hash": current_hash,
            "is_intact": is_intact,
            "attack_detected": not is_intact,
            "confidence": 1.0 if not is_intact else 0.0,
            "details": "File tampered!" if not is_intact else "File integrity verified"
        }
        
        if not is_intact:
            logger.warning(f"⚠️  ATTACK DETECTED: Hash mismatch!")
        else:
            logger.info("✓ File integrity verified")
        
        return result
    
    # ==================== METHOD 2: ANOMALY DETECTION ====================
    
    def detect_shape_anomaly(self, original: np.ndarray,
                            current: np.ndarray,
                            tolerance: int = 5) -> Dict:
        """
        Detect if volume shape changed (slice deletion).
        
        Args:
            original: Original volume
            current: Current volume to check
            tolerance: Allowed difference in slices
            
        Returns:
            Detection result
        """
        shape_diff = np.array(original.shape) - np.array(current.shape)
        shape_changed = np.any(np.abs(shape_diff) > tolerance)
        
        result = {
            "method": "shape_anomaly_detection",
            "original_shape": original.shape,
            "current_shape": current.shape,
            "difference": shape_diff.tolist(),
            "attack_detected": shape_changed,
            "confidence": 1.0 if shape_changed else 0.0,
            "details": f"Shape changed by {shape_diff}" if shape_changed else "Shape intact"
        }
        
        if shape_changed:
            logger.warning(f"⚠️  ATTACK DETECTED: Shape anomaly!")
            logger.warning(f"   Original: {original.shape} → Current: {current.shape}")
        
        return result
    
    def detect_intensity_anomaly(self, original: np.ndarray,
                                current: np.ndarray,
                                threshold: float = 0.1) -> Dict:
        """
        Detect significant intensity changes (pixel tampering).
        
        Args:
            original: Original volume
            current: Current volume to check
            threshold: Max allowed mean absolute difference
            
        Returns:
            Detection result
        """
        if original.shape != current.shape:
            return {
                "method": "intensity_anomaly_detection",
                "attack_detected": True,
                "confidence": 1.0,
                "details": "Cannot compare - shapes differ"
            }
        
        # Calculate difference metrics
        abs_diff = np.abs(original - current)
        mean_diff = np.mean(abs_diff)
        max_diff = np.max(abs_diff)
        changed_voxels_pct = (np.sum(abs_diff > 0.01) / original.size) * 100
        
        # Detect anomaly
        attack_detected = mean_diff > threshold
        
        result = {
            "method": "intensity_anomaly_detection",
            "mean_absolute_diff": float(mean_diff),
            "max_absolute_diff": float(max_diff),
            "changed_voxels_pct": float(changed_voxels_pct),
            "threshold": threshold,
            "attack_detected": attack_detected,
            "confidence": min(mean_diff / threshold, 1.0) if attack_detected else 0.0,
            "details": f"Mean diff: {mean_diff:.6f} (threshold: {threshold})"
        }
        
        if attack_detected:
            logger.warning(f"⚠️  ATTACK DETECTED: Intensity anomaly!")
            logger.warning(f"   Mean diff: {mean_diff:.6f}, Changed: {changed_voxels_pct:.2f}%")
        
        return result
    
    def detect_noise_anomaly(self, volume: np.ndarray,
                           expected_snr: Optional[float] = None,
                           min_snr: float = 20.0) -> Dict:
        """
        Detect abnormal noise levels (noise injection).
        
        Args:
            volume: Volume to check
            expected_snr: Expected signal-to-noise ratio (if known)
            min_snr: Minimum acceptable SNR
            
        Returns:
            Detection result
        """
        # Calculate SNR (simplified)
        signal = np.mean(volume[volume > 0.1])
        noise = np.std(volume[volume > 0.1])
        snr = signal / noise if noise > 0 else float('inf')
        snr_db = 20 * np.log10(snr) if snr > 0 else 0
        
        # Detect anomaly
        if expected_snr is not None:
            attack_detected = abs(snr_db - expected_snr) > 5.0
            threshold_used = expected_snr
        else:
            attack_detected = snr_db < min_snr
            threshold_used = min_snr
        
        result = {
            "method": "noise_anomaly_detection",
            "calculated_snr_db": float(snr_db),
            "expected_snr_db": expected_snr,
            "min_snr_threshold": min_snr,
            "attack_detected": attack_detected,
            "confidence": 0.7 if attack_detected else 0.0,  # Medium confidence
            "details": f"SNR: {snr_db:.2f} dB (threshold: {threshold_used:.2f} dB)"
        }
        
        if attack_detected:
            logger.warning(f"⚠️  ATTACK DETECTED: Noise anomaly!")
            logger.warning(f"   SNR: {snr_db:.2f} dB")
        
        return result
    
    def detect_statistical_anomaly(self, original: np.ndarray,
                                  current: np.ndarray) -> Dict:
        """
        Detect statistical property changes.
        
        Args:
            original: Original volume
            current: Current volume to check
            
        Returns:
            Detection result
        """
        if original.shape != current.shape:
            return {
                "method": "statistical_anomaly_detection",
                "attack_detected": True,
                "confidence": 1.0,
                "details": "Cannot compare - shapes differ"
            }
        
        # Calculate statistics
        orig_stats = {
            'mean': np.mean(original),
            'std': np.std(original),
            'min': np.min(original),
            'max': np.max(original),
            'median': np.median(original)
        }
        
        curr_stats = {
            'mean': np.mean(current),
            'std': np.std(current),
            'min': np.min(current),
            'max': np.max(current),
            'median': np.median(current)
        }
        
        # Calculate relative changes
        mean_change = abs(curr_stats['mean'] - orig_stats['mean']) / orig_stats['mean']
        std_change = abs(curr_stats['std'] - orig_stats['std']) / orig_stats['std']
        
        # Detect anomaly (>10% change in mean or std)
        attack_detected = mean_change > 0.1 or std_change > 0.1
        
        result = {
            "method": "statistical_anomaly_detection",
            "original_stats": {k: float(v) for k, v in orig_stats.items()},
            "current_stats": {k: float(v) for k, v in curr_stats.items()},
            "mean_change_pct": float(mean_change * 100),
            "std_change_pct": float(std_change * 100),
            "attack_detected": attack_detected,
            "confidence": max(mean_change, std_change) if attack_detected else 0.0,
            "details": f"Mean changed {mean_change*100:.2f}%, Std changed {std_change*100:.2f}%"
        }
        
        if attack_detected:
            logger.warning(f"⚠️  ATTACK DETECTED: Statistical anomaly!")
        
        return result
    
    # ==================== METHOD 3: METADATA VALIDATION ====================
    
    def validate_metadata(self, metadata: Dict) -> Dict:
        """
        Validate metadata for logical consistency and suspicious values.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            Validation result with detected issues
        """
        issues = []
        warnings = []
        
        current_date = datetime.now()
        
        # Validate scan date
        if "scan_date" in metadata:
            try:
                scan_date = datetime.fromisoformat(metadata["scan_date"])
                
                # Future date?
                if scan_date > current_date:
                    issues.append({
                        "field": "scan_date",
                        "value": metadata["scan_date"],
                        "issue": "Date is in the future!",
                        "severity": "critical"
                    })
                
                # Too old? (>50 years)
                if (current_date - scan_date).days > 365 * 50:
                    warnings.append({
                        "field": "scan_date",
                        "value": metadata["scan_date"],
                        "issue": "Date is very old (>50 years)",
                        "severity": "warning"
                    })
            
            except ValueError:
                issues.append({
                    "field": "scan_date",
                    "value": metadata["scan_date"],
                    "issue": "Invalid date format",
                    "severity": "critical"
                })
        
        # Validate MRI parameters
        if "TR" in metadata:
            tr = float(metadata["TR"])
            if tr < 0:
                issues.append({
                    "field": "TR",
                    "value": tr,
                    "issue": "TR cannot be negative",
                    "severity": "critical"
                })
            elif tr < 100 or tr > 10000:
                warnings.append({
                    "field": "TR",
                    "value": tr,
                    "issue": "TR value unusual (typical: 500-5000 ms)",
                    "severity": "warning"
                })
        
        if "TE" in metadata:
            te = float(metadata["TE"])
            if te < 0:
                issues.append({
                    "field": "TE",
                    "value": te,
                    "issue": "TE cannot be negative",
                    "severity": "critical"
                })
            elif te > 200:
                warnings.append({
                    "field": "TE",
                    "value": te,
                    "issue": "TE value very high (typical: 10-100 ms)",
                    "severity": "warning"
                })
        
        if "flip_angle" in metadata:
            angle = float(metadata["flip_angle"])
            if angle < 0 or angle > 180:
                issues.append({
                    "field": "flip_angle",
                    "value": angle,
                    "issue": "Flip angle must be 0-180 degrees",
                    "severity": "critical"
                })
        
        # Validate patient information
        if "patient_age" in metadata:
            age = int(metadata["patient_age"])
            if age < 0:
                issues.append({
                    "field": "patient_age",
                    "value": age,
                    "issue": "Age cannot be negative",
                    "severity": "critical"
                })
            elif age > 120:
                issues.append({
                    "field": "patient_age",
                    "value": age,
                    "issue": "Age unrealistic (>120 years)",
                    "severity": "critical"
                })
        
        # Validate scanner information
        if "scanner" in metadata:
            scanner = metadata["scanner"]
            known_scanners = ["Siemens", "GE", "Philips", "Toshiba", "Hitachi"]
            if not any(brand in scanner for brand in known_scanners):
                warnings.append({
                    "field": "scanner",
                    "value": scanner,
                    "issue": "Unknown scanner brand",
                    "severity": "warning"
                })
        
        # Build result
        attack_detected = len(issues) > 0
        
        result = {
            "method": "metadata_validation",
            "attack_detected": attack_detected,
            "critical_issues": issues,
            "warnings": warnings,
            "num_issues": len(issues),
            "num_warnings": len(warnings),
            "confidence": 0.9 if attack_detected else 0.0,
            "details": f"Found {len(issues)} critical issues, {len(warnings)} warnings"
        }
        
        if attack_detected:
            logger.warning(f"⚠️  ATTACK DETECTED: Metadata validation failed!")
            for issue in issues:
                logger.warning(f"   {issue['field']}: {issue['issue']}")
        
        return result
    
    # ==================== COMBINED DETECTION ====================
    
    def run_full_detection(self, 
                          original_volume: Optional[np.ndarray],
                          current_volume: np.ndarray,
                          original_metadata: Optional[Dict],
                          current_metadata: Dict,
                          original_hash: Optional[str] = None,
                          current_hash: Optional[str] = None) -> Dict:
        """
        Run all detection methods and aggregate results.
        
        Args:
            original_volume: Original volume (if available)
            current_volume: Current volume to check
            original_metadata: Original metadata (if available)
            current_metadata: Current metadata
            original_hash: Original file hash (if available)
            current_hash: Current file hash
            
        Returns:
            Comprehensive detection report
        """
        results = []
        
        # 1. Hash verification (if hashes provided)
        if original_hash and current_hash:
            hash_result = {
                "method": "hash_verification",
                "attack_detected": original_hash != current_hash,
                "confidence": 1.0 if original_hash != current_hash else 0.0,
                "details": "Hash mismatch" if original_hash != current_hash else "Hash match"
            }
            results.append(hash_result)
        
        # 2. Anomaly detection (if original volume provided)
        if original_volume is not None:
            results.append(self.detect_shape_anomaly(original_volume, current_volume))
            results.append(self.detect_intensity_anomaly(original_volume, current_volume))
            results.append(self.detect_statistical_anomaly(original_volume, current_volume))
        
        # 3. Noise detection
        results.append(self.detect_noise_anomaly(current_volume))
        
        # 4. Metadata validation
        results.append(self.validate_metadata(current_metadata))
        
        # Aggregate results
        num_detections = sum(1 for r in results if r["attack_detected"])
        max_confidence = max(r.get("confidence", 0) for r in results)
        
        overall_attack_detected = num_detections > 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_attack_detected": overall_attack_detected,
            "num_detections": num_detections,
            "max_confidence": max_confidence,
            "detection_methods_run": len(results),
            "individual_results": results,
            "summary": f"{num_detections}/{len(results)} methods detected attacks" if overall_attack_detected else "No attacks detected"
        }
        
        if overall_attack_detected:
            logger.warning(f"🚨 SECURITY ALERT: {num_detections} detection(s)!")
        else:
            logger.info("✓ Security check passed")
        
        return report


def main():
    """Example usage of security detector."""
    print("="*70)
    print("MRI SECURITY DETECTOR - Demo")
    print("="*70)
    
    # Create test volumes
    print("\n1. Creating test volumes...")
    original = np.random.rand(256, 256, 256) * 0.8 + 0.1
    
    # Create attacked version (tampered)
    attacked = original.copy()
    attacked[100:150, 100:150, 100:130] += 0.3
    attacked = np.clip(attacked, 0, 1)
    
    print(f"   Original shape: {original.shape}")
    print(f"   Attacked shape: {attacked.shape}")
    
    # Test metadata
    original_metadata = {
        "scan_date": "2024-01-15T10:30:00",
        "scanner": "Siemens 3T",
        "TR": 2000.0,
        "TE": 30.0,
        "patient_age": 45
    }
    
    attacked_metadata = {
        "scan_date": "2025-06-30T10:30:00",  # Future date!
        "scanner": "Fake Scanner X",
        "TR": -500.0,  # Negative TR!
        "TE": 30.0,
        "patient_age": 250  # Impossible age!
    }
    
    detector = MRISecurityDetector()
    
    # Test hash verification
    print("\n2. Testing Hash Verification:")
    print("-" * 70)
    
    orig_hash = detector.calculate_volume_hash(original)
    attack_hash = detector.calculate_volume_hash(attacked)
    
    print(f"   Original hash: {orig_hash[:32]}...")
    print(f"   Attacked hash: {attack_hash[:32]}...")
    print(f"   Hashes match: {orig_hash == attack_hash}")
    
    # Test anomaly detection
    print("\n3. Testing Anomaly Detection:")
    print("-" * 70)
    
    shape_result = detector.detect_shape_anomaly(original, attacked)
    print(f"   Shape anomaly: {shape_result['attack_detected']}")
    
    intensity_result = detector.detect_intensity_anomaly(original, attacked)
    print(f"   Intensity anomaly: {intensity_result['attack_detected']}")
    print(f"   Mean diff: {intensity_result['mean_absolute_diff']:.6f}")
    
    stats_result = detector.detect_statistical_anomaly(original, attacked)
    print(f"   Statistical anomaly: {stats_result['attack_detected']}")
    
    noise_result = detector.detect_noise_anomaly(attacked)
    print(f"   Noise anomaly: {noise_result['attack_detected']}")
    
    # Test metadata validation
    print("\n4. Testing Metadata Validation:")
    print("-" * 70)
    
    validation_result = detector.validate_metadata(attacked_metadata)
    print(f"   Attack detected: {validation_result['attack_detected']}")
    print(f"   Critical issues: {validation_result['num_issues']}")
    print(f"   Warnings: {validation_result['num_warnings']}")
    
    for issue in validation_result['critical_issues']:
        print(f"      ⚠️  {issue['field']}: {issue['issue']}")
    
    # Test full detection
    print("\n5. Running Full Detection Suite:")
    print("-" * 70)
    
    full_report = detector.run_full_detection(
        original_volume=original,
        current_volume=attacked,
        original_metadata=original_metadata,
        current_metadata=attacked_metadata,
        original_hash=orig_hash,
        current_hash=attack_hash
    )
    
    print(f"   Overall attack detected: {full_report['overall_attack_detected']}")
    print(f"   Detection methods run: {full_report['detection_methods_run']}")
    print(f"   Positive detections: {full_report['num_detections']}")
    print(f"   Max confidence: {full_report['max_confidence']:.2f}")
    print(f"   Summary: {full_report['summary']}")
    
    print("\n" + "="*70)
    print("✓ All detection methods tested successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
