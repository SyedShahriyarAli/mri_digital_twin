import argparse
import sys

from mri_processing.download_mri import IXIDownloader
from mri_processing.preprocess_mri import MRIPreprocessor
from mri_processing.generate_metadata import MetadataGenerator

class MRIPipeline:
    def __init__(self):
        self.downloader = IXIDownloader()
        self.preprocessor = MRIPreprocessor()
        self.metadata_generator = MetadataGenerator()
    
    def run_download(self):
        print("\n STEP 1: DOWNLOADING MRI DATASET")
        
        success = self.downloader.download_dataset()
        return success
    
    def run_preprocessing(self):
        print("\n STEP 2: PREPROCESSING MRI IMAGES")
        
        results = self.preprocessor.preprocess_all()
        return len(results) > 0
    
    def run_metadata_generation(self):
        print("\n STEP 3: GENERATING METADATA")
        
        metadata = self.metadata_generator.generate_dataset_metadata()
        return metadata is not None
    
    def run_complete_pipeline(self):
        """Run the complete pipeline"""

        print("Starting STEP 1 of 3...")
        if not self.run_download():
            print("\nDownload failed! Aborting pipeline.")
            return False
        
        #\n STEP 2: Preprocess
        print("\nStarting STEP 2 of 3...")
        if not self.run_preprocessing():
            print("\nPreprocessing failed! Aborting pipeline.")
            return False
        
        #\n STEP 3: Metadata
        print("\nStarting STEP 3 of 3...")
        if not self.run_metadata_generation():
            print("\nMetadata generation failed!")
            return False
        
        return True
    
    def print_help(self):
        """Print helpful information"""
        print("\nUSAGE:")
        print("   python main_pipeline.py --all         # Run complete pipeline")
        print("   python main_pipeline.py --download    # Only download")
        print("   python main_pipeline.py --preprocess  # Only preprocess")
        print("   python main_pipeline.py --metadata    # Only metadata")
        print("   python main_pipeline.py --help        # Show this help")
        print("\nREQUIREMENTS:")
        print("   pip install requests tqdm nibabel numpy scipy")
        print("   pip install scikit-image matplotlib faker")
        print("\nOUTPUT:")
        print("   All files are saved in the 'mri_data/' directory")

def main():
    parser = argparse.ArgumentParser(
        description='MRI Processing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline (download + preprocess + metadata)')
    parser.add_argument('--download', action='store_true',
                       help='Only download dataset')
    parser.add_argument('--preprocess', action='store_true',
                       help='Only run preprocessing')
    parser.add_argument('--metadata', action='store_true',
                       help='Only generate metadata')
    parser.add_argument('--help-extended', action='store_true',
                       help='Show extended help information')
    
    args = parser.parse_args()
    
    pipeline = MRIPipeline()
    
    if args.help_extended:
        pipeline.print_help()
        return
    
    if not any([args.all, args.download, args.preprocess, args.metadata]):
        print("No arguments provided. Running complete pipeline...")
        args.all = True
    
    try:
        if args.all:
            success = pipeline.run_complete_pipeline()
            sys.exit(0 if success else 1)
        
        if args.download:
            success = pipeline.run_download()
            if not success:
                sys.exit(1)
        
        if args.preprocess:
            success = pipeline.run_preprocessing()
            if not success:
                sys.exit(1)
        
        if args.metadata:
            success = pipeline.run_metadata_generation()
            if not success:
                sys.exit(1)
        
        print("\nRequested operations completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()