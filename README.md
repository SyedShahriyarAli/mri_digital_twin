# MRI Digital Twin - Interactive Viewer & Security Testbed

A comprehensive MRI processing, visualization, and security testing platform for medical imaging research and cybersecurity training.

## Project Overview

This project implements a full-featured MRI Digital Twin system that:
- Downloads and preprocesses MRI scans from the IXI public dataset
- Provides an interactive web-based viewer with multi-planar reconstruction
- Simulates MRI transformations (noise, filtering, rotation, scaling)
- Implements a security testbed for attack simulation and detection
- Generates realistic acquisition metadata for research purposes

## Key Features

### 🖼️ Interactive MRI Viewer
- **Multi-format support**: Load .nii.gz (NIfTI) and .npy (NumPy) files
- **Three-plane viewing**: Axial, Sagittal, and Coronal views
- **Interactive slicing**: Real-time slice navigation with responsive sliders
- **Grid dashboard**: Browse all scans with search and filter capabilities
- **Modal viewer**: Full-screen viewing experience

### 🔧 MRI Transformations
- **Noise Addition**: Gaussian, Salt & Pepper, Speckle noise
- **Filtering**: Gaussian Blur, Median Filter, Edge Enhancement
- **Geometric Operations**: Rotation and Scaling
- **Real-time Preview**: Side-by-side comparison of original vs transformed
- **Parameter Control**: Adjustable intensity for all transformations

### 🔒 Security Testbed
- **Attack Simulation**:
  - Data Tampering (pixel modification, slice deletion, region blurring)
  - Metadata Manipulation (date, scanner, parameters)
  - Noise Injection (Gaussian, spikes, motion artifacts)
  - File Corruption (header corruption, random byte changes)
- **Detection Mechanisms**:
  - Hash Verification (SHA-256)
  - Anomaly Detection (statistical comparison)
  - Metadata Validation (integrity checks)
- **Visual Feedback**: Real-time attack monitoring and detection alerts

### 📊 Data Processing Pipeline
1. **Dataset Management**
   - Automated download from IXI dataset
   - Smart caching (skip existing files)

2. **Image Preprocessing**
   - Format conversion (NIfTI ↔ NumPy)
   - Gaussian denoising
   - Intensity normalization (0-1 range)
   - Standardized resizing (256×256×256)
   - High-quality interpolation

3. **Metadata Generation**
   - Anonymized patient IDs
   - Realistic scan parameters (TR, TE, flip angle)
   - Equipment specifications
   - Quality metrics (SNR, artifacts)
   - JSON and CSV export

## Installation

### Prerequisites
- Python 3.8+
- pip or uv package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/SyedShahriyarDev/mri_digital_twin
cd mri_digital_twin
```

2. **Install dependencies**
```bash
uv sync
```

Or manually:
```bash
pip install flask nibabel numpy scipy scikit-image matplotlib faker pillow
```

3. **Prepare data directory**
```bash
mkdir -p mri_data/raw mri_data/preprocessed
mkdir -p static/slices
```

## Quick Start

### 1. Download and Preprocess MRI Data

**Run complete pipeline:**
```bash
python main_pipeline.py --all
```

This will:
1. Download 5 sample MRI scans (~500MB-1GB)
2. Preprocess all images (denoising, normalization, resizing to 256×256×256)
3. Generate metadata
4. Create visualizations

**Run individual steps:**
```bash
python main_pipeline.py --download     # Download only
python main_pipeline.py --preprocess   # Preprocess only
python main_pipeline.py --metadata     # Generate metadata only
```

### 2. Launch Web Application

```bash
python app.py
```

Then open your browser to: **http://localhost:5000**

### 3. Navigate the Application

**MRI Scans** (`/`)
- Browse all available MRI scans
- Filter by format (NIfTI/NumPy)
- Search by filename
- Click "View" to open interactive viewer

**Security Testing** (`/security`)
- Load an MRI scan
- Select attack type (tampering, metadata, noise, corruption)
- Launch simulated attack
- Run detection mechanisms
- View integrity reports and alerts

## Project Structure

```
mri_digital_twin/
├── app.py                          # Flask web application
├── main_pipeline.py                # Data processing pipeline
├── digital_twin/
│   ├── __init__.py
│   ├── mri_loader.py               # Load MRI data (NIfTI/NumPy)
│   ├── visualizer.py               # Generate slice images
│   ├── transformations.py          # Image transformation operations
│   ├── attack_simulator.py         # Security attack implementations
│   └── detector.py                 # Attack detection mechanisms
├── mri_processing/
│   ├── downloader.py               # Download IXI dataset
│   ├── preprocessor.py             # Image preprocessing
│   ├── metadata_generator.py       # Generate scan metadata
│   └── visualizer.py               # Create visualization images
├── mri_data/
│   ├── raw/                        # Downloaded MRI scans
│   └── preprocessed/               # Processed scans (.nii.gz, .npy)
├── templates/
│   ├── base.html                   # Master layout template
│   ├── index.html                  # Dashboard page
│   ├── transformations.html        # Transformations interface
│   └── security_testing.html       # Security testbed interface
├── static/
│   ├── css/
│   │   └── style.css               # Application styling
│   └── slices/                     # Generated slice images
└── README.md
```

## Output Files

### Preprocessed Images
- **Location**: `mri_data/preprocessed/`
- **NIfTI format** (`.nii.gz`): Standard neuroimaging format, preserves spatial information
- **NumPy format** (`.npy`): Python-friendly arrays for ML/analysis (256×256×256)

### Slice Images
- **Location**: `static/slices/`
- **Format**: PNG images (optimized for web display)
- **Naming**: `{scan_name}_{view}_{slice}.png`
- **Views**: axial, sagittal, coronal

### Metadata Files

**Individual Metadata** (`metadata/PATIENT_XXXX_metadata.json`):
```json
{
  "patient_id": "PATIENT_0001",
  "age": 45,
  "sex": "F",
  "scan_type": "T1-weighted",
  "acquisition_date": "2024-03-15",
  "scanner_model": "Siemens Magnetom Prisma 3T",
  "field_strength": "3T",
  "coil_type": "32-channel head coil",
  "repetition_time_ms": 2450.5,
  "echo_time_ms": 3.2,
  "flip_angle_degrees": 9.0,
  "voxel_size_mm": [1.0, 1.0, 1.0],
  "matrix_size": [256, 256, 256],
  "snr": 42.5,
  "has_motion_artifacts": false
}
```

**Combined Metadata** (`dataset_metadata.json`): Array of all patient metadata

**CSV Format** (`dataset_metadata.csv`): Spreadsheet-compatible format

**Summary Statistics** (`metadata_summary.json`):
```json
{
  "total_scans": 5,
  "scan_type_distribution": {"T1-weighted": 5},
  "age_range": {"min": 18, "max": 85, "mean": 52.4},
  "scanner_models": ["Siemens Magnetom Prisma 3T", "GE Discovery MR750"]
}
```

## Technical Details

### Preprocessing Pipeline

1. **Loading**: Read NIfTI files using nibabel
2. **Denoising**: Gaussian filter (σ=1.0)
3. **Normalization**: 
   - Remove background (1st percentile threshold)
   - Scale to [0, 1] range
4. **Resizing**: Zoom interpolation to 256×256×256 (optimized for quality)
5. **Saving**: Dual format output with preserved affine transforms

### Visualization Engine

- **Real-time slicing**: Extract any slice from 3D volume
- **Multi-planar reconstruction**: Axial, Sagittal, Coronal views
- **Image optimization**: PNG compression with matplotlib for web display
- **Dynamic loading**: On-demand slice generation

### Transformation Types

**Noise Operations:**
- Gaussian (σ adjustable)
- Salt & Pepper (density adjustable)
- Speckle (variance adjustable)

**Filtering Operations:**
- Gaussian Blur (smoothing)
- Median Filter (noise reduction)
- Edge Enhancement (Laplacian)

**Geometric Operations:**
- Rotation (any angle)
- Scaling (up/down)

### Security Framework

**Attack Types:**
1. **Data Tampering**: Direct pixel/slice manipulation
2. **Metadata Manipulation**: Change scan parameters
3. **Noise Injection**: Add artifacts to simulate interference
4. **File Corruption**: Break file integrity

**Detection Methods:**
1. **Hash Verification**: SHA-256 comparison
2. **Anomaly Detection**: Statistical analysis (mean, std, histogram)
3. **Metadata Validation**: Integrity checks on scan parameters

## Dataset Information

### IXI Dataset
- **Source**: Imperial College London, Guy's and St Thomas' NHS Foundation Trust, Hammersmith Hospital
- **Content**: Brain MRI scans from healthy subjects (600+ subjects)
- **Modalities**: T1, T2, PD-weighted, MRA, DTI
- **Website**: https://brain-development.org/ixi-dataset/

This project uses T1-weighted images for demonstration purposes.