import os
import json
import random
from datetime import datetime, timedelta
from faker import Faker

class MetadataGenerator:
    def __init__(self, output_dir="./mri_data/preprocessed"):
        self.output_dir = output_dir
        self.fake = Faker()
        
        # MRI parameters
        self.scan_types = ['T1-weighted', 'T2-weighted', 'FLAIR', 'T1-FLAIR', 'PD-weighted']
        self.coil_types = [
            '8-channel head coil',
            '16-channel head coil', 
            '32-channel head coil',
            '64-channel head/neck coil',
            '20-channel head/neck coil'
        ]
        self.scanner_models = [
            'Siemens Magnetom Prisma 3T',
            'GE Discovery MR750 3T',
            'Philips Ingenia 3T',
            'Siemens Skyra 3T',
            'GE Signa HDxt 1.5T'
        ]
        self.field_strengths = ['1.5T', '3T']
        self.manufacturers = ['Siemens', 'GE Healthcare', 'Philips']
        
    def generate_patient_id(self, index):
        """Generate anonymized patient ID"""
        return f"PATIENT_{str(index).zfill(4)}"
    
    def generate_scan_metadata(self, filename, patient_index):
        """Generate complete metadata for a single scan"""
        
        base_name = os.path.splitext(os.path.splitext(filename)[0])[0]
        
        days_ago = random.randint(1, 730)
        acquisition_time = datetime.now() - timedelta(days=days_ago)
        
        metadata = {
            # Patient Information
            "patient_id": self.generate_patient_id(patient_index),
            "age": random.randint(18, 85),
            "sex": random.choice(['M', 'F']),
            
            # Scan Information
            "scan_id": f"SCAN_{base_name}",
            "scan_type": random.choice(self.scan_types),
            "acquisition_date": acquisition_time.strftime("%Y-%m-%d"),
            "acquisition_time": acquisition_time.strftime("%H:%M:%S"),
            "acquisition_datetime": acquisition_time.isoformat(),
            
            # Equipment Information
            "manufacturer": random.choice(self.manufacturers),
            "scanner_model": random.choice(self.scanner_models),
            "field_strength": random.choice(self.field_strengths),
            "coil_type": random.choice(self.coil_types),
            "software_version": f"VE{random.randint(10, 15)}{chr(random.randint(65, 90))}",
            
            # Acquisition Parameters
            "repetition_time_ms": round(random.uniform(1500, 3000), 2),
            "echo_time_ms": round(random.uniform(2.5, 30), 2),
            "flip_angle_degrees": random.choice([8, 9, 12, 15, 20]),
            "slice_thickness_mm": round(random.uniform(1.0, 5.0), 2),
            "matrix_size": random.choice(["256x256", "512x512", "384x384"]),
            "voxel_size_mm": f"{round(random.uniform(0.8, 1.2), 2)}x{round(random.uniform(0.8, 1.2), 2)}x{round(random.uniform(0.8, 1.2), 2)}",
            
            # Study Information
            "study_description": random.choice([
                "Brain MRI - Structural",
                "Brain MRI - Research Protocol",
                "Neurological Assessment",
                "Brain Volumetric Study"
            ]),
            "series_number": random.randint(1, 20),
            "institution_name": self.fake.company(),
            
            # Quality Metrics
            "snr_estimate": round(random.uniform(15, 45), 2),
            "motion_artifact": random.choice(["None", "Minimal", "Mild"]),
            "image_quality": random.choice(["Excellent", "Good", "Acceptable"]),
            
            # Processing Information
            "preprocessing_date": datetime.now().isoformat(),
            "preprocessed": True,
            "normalized": True,
            "denoised": True,
            "target_dimensions": "256x256x256",
            
            # File References
            "original_filename": filename,
            "preprocessed_nifti": f"nifti/{base_name}_preprocessed.nii.gz",
            "preprocessed_numpy": f"numpy/{base_name}_preprocessed.npy",
            "visualization": f"visualizations/{base_name}_slices.png"
        }
        
        return metadata
    
    def generate_dataset_metadata(self):
        """Generate metadata for all preprocessed files"""
        
        # Find all preprocessed files
        nifti_dir = os.path.join(self.output_dir, "nifti")
        if not os.path.exists(nifti_dir):
            print(f"Error: Preprocessed directory not found: {nifti_dir}")
            print("Please run preprocessing first!")
            return None
        
        preprocessed_files = [f for f in os.listdir(nifti_dir) if f.endswith('_preprocessed.nii.gz')]
        
        if not preprocessed_files:
            print("No preprocessed files found!")
            return None
        
        print("="*60)
        print("MRI Metadata Generator")
        print("="*60)
        print(f"Found {len(preprocessed_files)} preprocessed files\n")
        
        all_metadata = []
        for idx, filename in enumerate(preprocessed_files, start=1):
            original_filename = filename.replace('_preprocessed', '')
            
            print(f"Generating metadata for: {original_filename}")
            metadata = self.generate_scan_metadata(original_filename, idx)
            all_metadata.append(metadata)
        
        metadata_dir = os.path.join(self.output_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
        
        for metadata in all_metadata:
            patient_id = metadata['patient_id']
            metadata_file = os.path.join(metadata_dir, f"{patient_id}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"  Saved: {metadata_file}")
        
        # Save combined metadata
        combined_file = os.path.join(self.output_dir, "dataset_metadata.json")
        with open(combined_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        # Create CSV version for easy viewing
        csv_file = os.path.join(self.output_dir, "dataset_metadata.csv")
        self.create_csv_metadata(all_metadata, csv_file)
        
        # Create summary statistics
        summary = self.generate_summary(all_metadata)
        summary_file = os.path.join(self.output_dir, "metadata_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Metadata generation complete!")
        
        return all_metadata
    
    def create_csv_metadata(self, metadata_list, output_file):
        """Create CSV version of metadata"""
        import csv
        
        if not metadata_list:
            return
        
        keys = metadata_list[0].keys()
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metadata_list)
    
    def generate_summary(self, metadata_list):
        """Generate summary statistics"""
        summary = {
            "total_scans": len(metadata_list),
            "scan_type_distribution": {},
            "coil_type_distribution": {},
            "field_strength_distribution": {},
            "manufacturer_distribution": {},
            "age_range": {
                "min": min(m['age'] for m in metadata_list),
                "max": max(m['age'] for m in metadata_list),
                "mean": round(sum(m['age'] for m in metadata_list) / len(metadata_list), 2)
            },
            "sex_distribution": {
                "M": sum(1 for m in metadata_list if m['sex'] == 'M'),
                "F": sum(1 for m in metadata_list if m['sex'] == 'F')
            },
            "quality_distribution": {}
        }
        
        for metadata in metadata_list:
            # Scan types
            scan_type = metadata['scan_type']
            summary['scan_type_distribution'][scan_type] = \
                summary['scan_type_distribution'].get(scan_type, 0) + 1
            
            # Coil types
            coil = metadata['coil_type']
            summary['coil_type_distribution'][coil] = \
                summary['coil_type_distribution'].get(coil, 0) + 1
            
            # Field strength
            field = metadata['field_strength']
            summary['field_strength_distribution'][field] = \
                summary['field_strength_distribution'].get(field, 0) + 1
            
            # Manufacturer
            mfg = metadata['manufacturer']
            summary['manufacturer_distribution'][mfg] = \
                summary['manufacturer_distribution'].get(mfg, 0) + 1
            
            # Quality
            quality = metadata['image_quality']
            summary['quality_distribution'][quality] = \
                summary['quality_distribution'].get(quality, 0) + 1
        
        return summary

def main():
    generator = MetadataGenerator()
    metadata = generator.generate_dataset_metadata()
    
    if metadata:
        print(f"\nSuccessfully generated metadata for {len(metadata)} scans!")
    else:
        print("\nFailed to generate metadata")

if __name__ == "__main__":
    main()