"""
Flask Web Application for MRI Digital Twin Viewer

Provides a web interface for viewing MRI scans with interactive slice navigation
and transformation capabilities.
"""

import os
import json
import logging
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory
import numpy as np

# Import our custom modules
from digital_twin.mri_loader import MRILoader
from digital_twin.visualizer import MRIVisualizer
from digital_twin.transformer import MRITransformer
from digital_twin.comparator import MRIComparator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Initialize MRI loader, visualizer, transformer, and comparator
try:
    mri_loader = MRILoader()
    mri_visualizer = MRIVisualizer()
    mri_transformer = MRITransformer()
    mri_comparator = MRIComparator()
    logger.info("All MRI modules initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MRI modules: {e}")
    mri_loader = None
    mri_visualizer = None
    mri_transformer = None
    mri_comparator = None

# Cache for loaded volumes and transformations
volume_cache = {}
transformation_cache = {}


@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html', active_page='dashboard')


@app.route('/api/scans')
def get_scans():
    """API endpoint to get list of available scans."""
    try:
        if mri_loader is None:
            return jsonify({'error': 'MRI Loader not initialized'}), 500
        
        scans = mri_loader.list_available_scans()
        all_scans = []
        
        for scan_name in scans['nifti']:
            all_scans.append({'name': scan_name, 'format': 'nifti'})
        
        for scan_name in scans['numpy']:
            all_scans.append({'name': scan_name, 'format': 'numpy'})
        
        return jsonify({'scans': all_scans})
    
    except Exception as e:
        logger.error(f"Error getting scans: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/load_scan', methods=['POST'])
def load_scan():
    """Load a scan and return its dimensions."""
    try:
        data = request.get_json()
        scan_name = data.get('scan_name')
        scan_format = data.get('format', 'auto')
        
        if not scan_name:
            return jsonify({'error': 'No scan name provided'}), 400
        
        logger.info(f"Loading scan: {scan_name} (format: {scan_format})")
        
        # Load the volume
        volume, middle_slices = mri_loader.load_and_prepare(scan_name, format=scan_format)
        
        # Cache the volume
        volume_cache[scan_name] = volume
        
        # Get dimensions
        dimensions = mri_visualizer.get_volume_dimensions(volume)
        
        # Calculate middle indices
        middle_indices = {
            'axial': dimensions['axial'] // 2,
            'sagittal': dimensions['sagittal'] // 2,
            'coronal': dimensions['coronal'] // 2
        }
        
        logger.info(f"Loaded scan with dimensions: {dimensions}")
        
        return jsonify({
            'success': True,
            'dimensions': dimensions,
            'middle_indices': middle_indices,
            'shape': list(volume.shape)
        })
    
    except Exception as e:
        logger.error(f"Error loading scan: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_slice', methods=['POST'])
def get_slice():
    """Get a specific slice from a loaded volume."""
    try:
        data = request.get_json()
        scan_name = data.get('scan_name')
        view = data.get('view')
        slice_index = data.get('slice_index')
        
        if not all([scan_name, view, slice_index is not None]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Check if volume is cached
        if scan_name not in volume_cache:
            return jsonify({'error': 'Scan not loaded. Load scan first.'}), 400
        
        volume = volume_cache[scan_name]
        slice_index = int(slice_index)

        # Extract the slice
        slice_data = mri_visualizer.get_slice(volume, view, slice_index)
        
        # Generate filename for this slice
        filename = f"{scan_name}_{view}_{slice_index:03d}.png"
        
        # Save the slice as PNG
        filepath = mri_visualizer.save_slice_as_png(slice_data, filename)
        
        # Return relative path for web access
        relative_path = f"/slices/{filename}"
        
        return jsonify({
            'success': True,
            'image_url': relative_path,
            'slice_index': slice_index,
            'view': view
        })
    
    except ValueError as e:
        return jsonify({'error': f'Invalid slice index: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Error getting slice: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/slices/<filename>')
def serve_slice(filename):
    """Serve slice images from the static/slices directory."""
    return send_from_directory('static/slices', filename)


@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the volume cache and generated slices."""
    try:
        volume_cache.clear()
        transformation_cache.clear()
        mri_visualizer.clear_output_directory()
        logger.info("Cache cleared successfully")
        return jsonify({'success': True, 'message': 'Cache cleared'})
    
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/transform', methods=['POST'])
def apply_transformation():
    """Apply a transformation to a loaded MRI volume."""
    try:
        data = request.get_json()
        scan_name = data.get('scan_name')
        view = data.get('view')
        slice_index = data.get('slice_index')
        transform_type = data.get('transform_type')
        params = data.get('params', {})
        
        if not all([scan_name, view, slice_index is not None, transform_type]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Check if volume is cached
        if scan_name not in volume_cache:
            return jsonify({'error': 'Scan not loaded. Load scan first.'}), 400
        
        volume = volume_cache[scan_name]
        slice_index = int(slice_index)
        
        # Extract the slice
        slice_data = mri_visualizer.get_slice(volume, view, slice_index)
        
        # Apply transformation
        if transform_type == 'noise':
            noise_type = params.get('noise_type', 'gaussian')
            if noise_type == 'gaussian':
                transformed = mri_transformer.add_gaussian_noise(
                    slice_data, 
                    sigma=params.get('sigma', 0.1)
                )
            elif noise_type == 'salt_pepper':
                transformed = mri_transformer.add_salt_pepper_noise(
                    slice_data,
                    amount=params.get('amount', 0.05)
                )
            elif noise_type == 'speckle':
                transformed = mri_transformer.add_speckle_noise(
                    slice_data,
                    variance=params.get('variance', 0.1)
                )
            else:
                return jsonify({'error': f'Unknown noise type: {noise_type}'}), 400
        
        elif transform_type == 'filter':
            filter_type = params.get('filter_type', 'gaussian')
            if filter_type == 'gaussian':
                transformed = mri_transformer.gaussian_blur(
                    slice_data,
                    sigma=params.get('sigma', 2.0)
                )
            elif filter_type == 'median':
                transformed = mri_transformer.median_filter(
                    slice_data,
                    size=params.get('size', 3)
                )
            elif filter_type == 'sharpen':
                transformed = mri_transformer.sharpen(
                    slice_data,
                    alpha=params.get('alpha', 1.5)
                )
            elif filter_type == 'edge':
                transformed = mri_transformer.edge_detection(
                    slice_data,
                    method=params.get('method', 'sobel')
                )
            else:
                return jsonify({'error': f'Unknown filter type: {filter_type}'}), 400
        
        elif transform_type == 'geometric':
            geo_type = params.get('geo_type', 'rotate')
            if geo_type == 'rotate':
                transformed = mri_transformer.rotate(
                    slice_data,
                    angle=params.get('angle', 45)
                )
            elif geo_type == 'flip':
                transformed = mri_transformer.flip(
                    slice_data,
                    axis=int(params.get('axis', 0))
                )
            else:
                return jsonify({'error': f'Unknown geometric type: {geo_type}'}), 400
        
        elif transform_type == 'intensity':
            intensity_type = params.get('intensity_type', 'brightness')
            if intensity_type == 'brightness':
                transformed = mri_transformer.adjust_brightness(
                    slice_data,
                    factor=params.get('factor', 1.2)
                )
            elif intensity_type == 'contrast':
                transformed = mri_transformer.adjust_contrast(
                    slice_data,
                    factor=params.get('factor', 1.5)
                )
            elif intensity_type == 'histogram':
                transformed = mri_transformer.histogram_equalization(slice_data)
            else:
                return jsonify({'error': f'Unknown intensity type: {intensity_type}'}), 400
        
        else:
            return jsonify({'error': f'Unknown transform type: {transform_type}'}), 400
        
        # Calculate comparison metrics
        metrics = mri_comparator.calculate_metrics(slice_data, transformed)
        assessment = mri_comparator.assess_quality(metrics)
        
        # Save original and transformed images
        cache_key = f"{scan_name}_{view}_{slice_index}_{transform_type}"
        transformation_cache[cache_key] = {
            'original': slice_data,
            'transformed': transformed,
            'metrics': metrics
        }
        
        original_filename = f"{cache_key}_original.png"
        transformed_filename = f"{cache_key}_transformed.png"
        diff_filename = f"{cache_key}_diff.png"
        
        mri_visualizer.save_slice_as_png(slice_data, original_filename)
        mri_visualizer.save_slice_as_png(transformed, transformed_filename)
        
        # Create and save difference map
        diff_map = mri_comparator.create_difference_map(slice_data, transformed)
        mri_visualizer.save_slice_as_png(diff_map, diff_filename)
        
        return jsonify({
            'success': True,
            'original_url': f"/slices/{original_filename}",
            'transformed_url': f"/slices/{transformed_filename}",
            'diff_url': f"/slices/{diff_filename}",
            'metrics': metrics,
            'assessment': assessment
        })
    
    except Exception as e:
        logger.error(f"Error applying transformation: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/transform/list', methods=['GET'])
def list_transformations():
    """List available transformation operations."""
    transformations = {
        'noise': {
            'gaussian': {'params': ['sigma'], 'defaults': {'sigma': 0.1}},
            'salt_pepper': {'params': ['amount'], 'defaults': {'amount': 0.05}},
            'speckle': {'params': ['variance'], 'defaults': {'variance': 0.1}}
        },
        'filter': {
            'gaussian': {'params': ['sigma'], 'defaults': {'sigma': 2.0}},
            'median': {'params': ['size'], 'defaults': {'size': 3}},
            'sharpen': {'params': ['alpha'], 'defaults': {'alpha': 1.5}},
            'edge': {'params': ['method'], 'defaults': {'method': 'sobel'}}
        },
        'geometric': {
            'rotate': {'params': ['angle'], 'defaults': {'angle': 45}},
            'flip': {'params': ['axis'], 'defaults': {'axis': 0}}
        },
        'intensity': {
            'brightness': {'params': ['factor'], 'defaults': {'factor': 1.2}},
            'contrast': {'params': ['factor'], 'defaults': {'factor': 1.5}},
            'histogram': {'params': [], 'defaults': {}}
        }
    }
    
    return jsonify({'transformations': transformations})


@app.route('/security')
def security():
    """Render the security testing page."""
    return render_template('security.html', active_page='security')


# Security Testing API Endpoints
@app.route('/api/security/load_scan', methods=['POST'])
def security_load_scan():
    """Load a scan for security testing."""
    try:
        data = request.get_json()
        scan_name = data.get('scan_name')
        scan_format = data.get('format', 'auto')
        
        if not scan_name:
            return jsonify({'error': 'No scan name provided'}), 400
        
        logger.info(f"Loading scan for security testing: {scan_name}")
        
        # Load the volume
        volume, middle_slices = mri_loader.load_and_prepare(scan_name, format=scan_format)
        
        # Cache the original volume
        volume_cache[f'security_original_{scan_name}'] = volume
        
        # Generate metadata (simulated)
        import hashlib
        volume_bytes = volume.tobytes()
        file_hash = hashlib.sha256(volume_bytes).hexdigest()[:16]
        
        metadata = {
            'scan_date': '2024-03-15',
            'scanner': 'Siemens 3T',
            'tr': '2000',
            'te': '30',
            'hash': file_hash
        }
        
        dimensions = volume.shape
        
        return jsonify({
            'success': True,
            'dimensions': dimensions,
            'middle_slice': dimensions[0] // 2,
            'metadata': metadata
        })
        
    except Exception as e:
        logger.error(f"Error loading scan for security testing: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/security/inject_attack', methods=['POST'])
def security_inject_attack():
    """Inject an attack into the scan."""
    try:
        data = request.get_json()
        scan_name = data.get('scan_name')
        attack_type = data.get('attack_type')
        intensity = data.get('intensity', 'medium')
        
        if not scan_name or not attack_type:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        logger.info(f"Injecting attack: {attack_type} on {scan_name} with {intensity} intensity")
        
        # Get original volume
        original_volume = volume_cache.get(f'security_original_{scan_name}')
        if original_volume is None:
            return jsonify({'error': 'Scan not loaded'}), 400
        
        # Apply attack based on type
        attacked_volume = original_volume.copy()
        modified_metadata = {
            'scan_date': '2024-03-15',
            'scanner': 'Siemens 3T',
            'tr': '2000',
            'te': '30',
            'hash': 'modified'
        }
        
        # Attack simulation
        if 'pixel_modification' in attack_type:
            # Modify pixel values
            factor = 0.7 if intensity == 'low' else 0.5 if intensity == 'medium' else 0.3
            attacked_volume = attacked_volume * factor
            
        elif 'region_deletion' in attack_type:
            # Delete a region
            size = 20 if intensity == 'low' else 40 if intensity == 'medium' else 60
            mid = attacked_volume.shape[0] // 2
            attacked_volume[mid-size:mid+size, mid-size:mid+size, mid-size:mid+size] = 0
            
        elif 'region_blur' in attack_type:
            # Blur a region (simplified)
            from scipy.ndimage import gaussian_filter
            sigma = 1 if intensity == 'low' else 2 if intensity == 'medium' else 3
            attacked_volume = gaussian_filter(attacked_volume, sigma=sigma)
            
        elif 'date_modification' in attack_type:
            # Change metadata
            modified_metadata['scan_date'] = '2020-01-01'
            
        elif 'scanner_modification' in attack_type:
            # Change scanner type
            modified_metadata['scanner'] = 'GE 1.5T'
            
        elif 'parameter_modification' in attack_type:
            # Change parameters
            modified_metadata['tr'] = '1500'
            modified_metadata['te'] = '50'
            
        elif 'gaussian_noise' in attack_type:
            # Add Gaussian noise
            from digital_twin.transformer import MRITransformer
            transformer = MRITransformer()
            std = 0.01 if intensity == 'low' else 0.05 if intensity == 'medium' else 0.1
            attacked_volume = transformer.add_gaussian_noise(attacked_volume, sigma=std)
            
        elif 'salt_pepper' in attack_type:
            # Add salt and pepper noise
            from digital_twin.transformer import MRITransformer
            transformer = MRITransformer()
            amount = 0.01 if intensity == 'low' else 0.05 if intensity == 'medium' else 0.1
            attacked_volume = transformer.add_salt_pepper_noise(attacked_volume,  amount)
            
        elif 'motion_artifact' in attack_type:
            # Simulate motion artifact with blur
            from scipy.ndimage import gaussian_filter
            sigma = 1 if intensity == 'low' else 2 if intensity == 'medium' else 3
            attacked_volume = gaussian_filter(attacked_volume, sigma=sigma)
        
        # Recalculate hash
        import hashlib
        attacked_bytes = attacked_volume.tobytes()
        modified_metadata['hash'] = hashlib.sha256(attacked_bytes).hexdigest()[:16]
        
        # Cache the attacked volume
        volume_cache[f'security_attacked_{scan_name}'] = attacked_volume
        
        return jsonify({
            'success': True,
            'attack_type': attack_type,
            'intensity': intensity,
            'metadata': modified_metadata
        })
        
    except Exception as e:
        logger.error(f"Error injecting attack: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/security/get_original_slice/<scan_name>/<view>/<int:slice_index>')
def get_original_slice(scan_name, view, slice_index):
    """Get a slice from the original volume."""
    try:
        original_volume = volume_cache.get(f'security_original_{scan_name}')
        
        if original_volume is None:
            return jsonify({'error': 'Original scan not found'}), 404
        
        # Extract slice based on view
        slice_data = mri_visualizer.get_slice(original_volume, view, slice_index)
        
        # Save as temporary image
        from pathlib import Path
        temp_dir = Path('static/temp_slices')
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f'security_original_{scan_name}_{view}_{slice_index}.png'

        visualizer = MRIVisualizer(output_dir='static/temp_slices')

        visualizer.save_slice_as_png(slice_data, filename)
        
        return send_from_directory('static/temp_slices', f'security_original_{scan_name}_{view}_{slice_index}.png')
        
    except Exception as e:
        logger.error(f"Error getting original slice: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/security/get_tampered_slice/<scan_name>/<view>/<int:slice_index>')
def get_tampered_slice(scan_name, view, slice_index):
    """Get a slice from the attacked volume."""
    try:
        attacked_volume = volume_cache.get(f'security_attacked_{scan_name}')
        
        if attacked_volume is None:
            # If no attack applied yet, return original
            return get_original_slice(scan_name, view, slice_index)
        
        # Extract slice based on view
        slice_data = mri_visualizer.get_slice(attacked_volume, view, slice_index)
        
        # Save as temporary image
        from pathlib import Path
        temp_dir = Path('static/temp_slices')
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f'security_tampered_{scan_name}_{view}_{slice_index}.png'

        visualizer = MRIVisualizer(output_dir='static/temp_slices')

        visualizer.save_slice_as_png(slice_data, filename)
        
        return send_from_directory('static/temp_slices', f'security_tampered_{scan_name}_{view}_{slice_index}.png')
        
    except Exception as e:
        logger.error(f"Error getting tampered slice: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/security/run_detection', methods=['POST'])
def security_run_detection():
    """Run security detection on the scan."""
    try:
        data = request.get_json()
        scan_name = data.get('scan_name')
        
        if not scan_name:
            return jsonify({'error': 'No scan name provided'}), 400
        
        logger.info(f"Running security detection on: {scan_name}")
        
        # Get volumes
        original_volume = volume_cache.get(f'security_original_{scan_name}')
        attacked_volume = volume_cache.get(f'security_attacked_{scan_name}')
        
        if original_volume is None or attacked_volume is None:
            return jsonify({'error': 'Scan not loaded or not attacked'}), 400
        
        # Run detection checks
        results = {}
        
        # 1. Hash Verification
        import hashlib
        original_hash = hashlib.sha256(original_volume.tobytes()).hexdigest()[:16]
        attacked_hash = hashlib.sha256(attacked_volume.tobytes()).hexdigest()[:16]
        
        hash_match = original_hash == attacked_hash
        results['hash_verification'] = {
            'passed': hash_match,
            'status': 'Verified' if hash_match else 'Hash Mismatch Detected',
            'message': 'File integrity verified' if hash_match else 'File has been tampered with - hash values do not match'
        }
        
        # 2. Anomaly Detection
        diff = np.abs(original_volume - attacked_volume)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        anomaly_threshold = 0.01
        anomaly_detected = mean_diff > anomaly_threshold
        
        results['anomaly_detection'] = {
            'passed': not anomaly_detected,
            'status': 'Normal' if not anomaly_detected else 'Anomaly Detected',
            'message': f'No significant differences detected' if not anomaly_detected else f'Significant image differences detected (mean: {mean_diff:.4f}, max: {max_diff:.4f})'
        }
        
        # 3. Metadata Validation (simulated)
        # In real scenario, check metadata consistency
        results['metadata_validation'] = {
            'passed': False,
            'status': 'Metadata Modified',
            'message': 'Metadata inconsistencies detected - scan parameters have been altered'
        }
        
        # Overall assessment
        all_passed = all(r['passed'] for r in results.values())
        
        assessment = {
            'status': 'secure' if all_passed else 'compromised',
            'message': 'All security checks passed - scan is authentic' if all_passed else 'Security breach detected - scan has been tampered with'
        }
        
        return jsonify({
            'success': True,
            'results': results,
            'assessment': assessment
        })
        
    except Exception as e:
        logger.error(f"Error running detection: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/security/reset_scan', methods=['POST'])
def security_reset_scan():
    """Reset scan to original state."""
    try:
        data = request.get_json()
        scan_name = data.get('scan_name')
        
        if not scan_name:
            return jsonify({'error': 'No scan name provided'}), 400
        
        logger.info(f"Resetting scan: {scan_name}")
        
        # Remove attacked volume from cache
        attacked_key = f'security_attacked_{scan_name}'
        if attacked_key in volume_cache:
            del volume_cache[attacked_key]
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Error resetting scan: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500



@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


def main():
    """Run the Flask application."""
    # Check if required directories exist
    required_dirs = ['templates', 'static', 'static/slices', 'static/css']
    for directory in required_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting MRI Digital Twin Viewer on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)


if __name__ == '__main__':
    main()
