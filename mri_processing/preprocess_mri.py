import os
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.transform import resize
import matplotlib.pyplot as plt
import json
from datetime import datetime

class MRIPreprocessor:
    def __init__(self, input_dir="./mri_data/raw", output_dir="mri_data/preprocessed"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_shape = (256, 256, 256)
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "nifti"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "numpy"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    def load_nifti(self, filepath):
        """Load NIfTI file and return image data"""
        try:
            img = nib.load(filepath)
            data = img.get_fdata()
            return data, img.affine, img.header
        except Exception as e:
            print(f"Error loading {filepath}: {str(e)}")
            return None, None, None
    
    def normalize_image(self, img_data):
        """Normalize image intensity to [0, 1] range"""
        mask = img_data > np.percentile(img_data, 1)
        
        img_min = img_data[mask].min()
        img_max = img_data[mask].max()
        
        normalized = np.clip((img_data - img_min) / (img_max - img_min + 1e-8), 0, 1)
        return normalized
    
    def resize_image(self, img_data, target_shape):
        """Resize image to target shape"""
        if img_data.shape == target_shape:
            return img_data
        
        print(f"  Resizing from {img_data.shape} to {target_shape}...")
        resized = resize(img_data, target_shape, mode='constant', 
                        anti_aliasing=True, preserve_range=True)
        return resized
    
    def denoise_image(self, img_data, sigma=1.0):
        """Apply Gaussian filtering for denoising"""
        denoised = ndimage.gaussian_filter(img_data, sigma=sigma)
        return denoised
    
    def save_preprocessed(self, img_data, filename, affine, header):
        """Save preprocessed image in both NIfTI and NumPy formats"""
        base_name = os.path.splitext(os.path.splitext(filename)[0])[0]
        
        # Save as NIfTI
        nifti_path = os.path.join(self.output_dir, "nifti", f"{base_name}_preprocessed.nii.gz")
        nifti_img = nib.Nifti1Image(img_data, affine, header)
        nib.save(nifti_img, nifti_path)
        
        # Save as NumPy
        numpy_path = os.path.join(self.output_dir, "numpy", f"{base_name}_preprocessed.npy")
        np.save(numpy_path, img_data)
        
        return nifti_path, numpy_path
    
    def visualize_slices(self, img_data, filename):
        """Create visualization of middle slices"""
        base_name = os.path.splitext(os.path.splitext(filename)[0])[0]
        
        # Get middle slices
        mid_x = img_data.shape[0] // 2
        mid_y = img_data.shape[1] // 2
        mid_z = img_data.shape[2] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_data[mid_x, :, :], cmap='gray')
        axes[0].set_title('Sagittal View')
        axes[0].axis('off')
        
        axes[1].imshow(img_data[:, mid_y, :], cmap='gray')
        axes[1].set_title('Coronal View')
        axes[1].axis('off')
        
        axes[2].imshow(img_data[:, :, mid_z], cmap='gray')
        axes[2].set_title('Axial View')
        axes[2].axis('off')
        
        plt.suptitle(f'Preprocessed MRI: {base_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        viz_path = os.path.join(self.output_dir, "visualizations", f"{base_name}_slices.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return viz_path
    
    def preprocess_file(self, filename):
        """Complete preprocessing pipeline for a single file"""
        print(f"\nProcessing: {filename}")
        filepath = os.path.join(self.input_dir, filename)
        
        # Load
        img_data, affine, header = self.load_nifti(filepath)
        if img_data is None:
            return None
        
        print(f"  Original shape: {img_data.shape}")
        print(f"  Original range: [{img_data.min():.2f}, {img_data.max():.2f}]")
        
        # Preprocessing steps
        print("  Denoising...")
        img_data = self.denoise_image(img_data)
        
        print("  Normalizing...")
        img_data = self.normalize_image(img_data)
        
        print("  Resizing...")
        img_data = self.resize_image(img_data, self.target_shape)
        
        print(f"  Final shape: {img_data.shape}")
        print(f"  Final range: [{img_data.min():.2f}, {img_data.max():.2f}]")
        
        # Save
        nifti_path, numpy_path = self.save_preprocessed(img_data, filename, affine, header)
        print(f"  Saved NIfTI: {nifti_path}")
        print(f"  Saved NumPy: {numpy_path}")
        
        # Visualize
        viz_path = self.visualize_slices(img_data, filename)
        print(f"  Saved visualization: {viz_path}")
        
        return {
            'filename': filename,
            'nifti_path': nifti_path,
            'numpy_path': numpy_path,
            'visualization': viz_path,
            'shape': img_data.shape,
            'processed_time': datetime.now().isoformat()
        }
    
    def preprocess_all(self):
        """Preprocess all files in input directory"""
        files = [f for f in os.listdir(self.input_dir) if f.endswith('.nii.gz')]
        
        if not files:
            print("No NIfTI files found in input directory!")
            return []
        
        print(f"Found {len(files)} files to process\n")
        
        results = []
        for filename in files:
            result = self.preprocess_file(filename)
            if result:
                results.append(result)
        
        print("\n" + "="*60)
        print(f"Preprocessing complete: {len(results)}/{len(files)} files")
        print("="*60)
        
        summary_path = os.path.join(self.output_dir, "preprocessing_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nSummary saved to: {summary_path}")
        
        return results

def main():
    preprocessor = MRIPreprocessor()
    results = preprocessor.preprocess_all()
    
    if results:
        print(f"\nSuccessfully preprocessed {len(results)} MRI scans!")
        print(f"Check the 'mri_data/preprocessed' directory for outputs")
    else:
        print("\n No files were successfully preprocessed")

if __name__ == "__main__":
    main()