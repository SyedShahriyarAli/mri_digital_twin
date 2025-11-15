import os
import requests
from tqdm import tqdm
import tarfile

class IXIDownloader:
    def __init__(self, output_dir="./mri_data/raw"):
        self.output_dir = output_dir

        self.base_url = "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/"
        self.archive_filename = "IXI-T1.tar"
        self.archive_url = self.base_url + self.archive_filename
        self.archive_filepath = os.path.join(self.output_dir, self.archive_filename)
        
        self.sample_files = [
            "IXI002-Guys-0828-T1.nii.gz",
            "IXI012-HH-1211-T1.nii.gz",
            "IXI013-HH-1212-T1.nii.gz",
            "IXI015-HH-1258-T1.nii.gz",
            "IXI025-Guys-0852-T1.nii.gz"
        ]
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _download_archive(self):
        """Download the entire .tar archive with progress bar"""
        url = self.archive_url
        filepath = self.archive_filepath
        
        if os.path.exists(filepath):
            print(f"{self.archive_filename} already exists, skipping download.")
            return True
        
        try:
            print(f"Downloading full archive: {self.archive_filename}...")
            response = requests.get(url, stream=True, timeout=300) 
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=self.archive_filename
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            
            print(f"\nSuccessfully downloaded {self.archive_filename}")
            return True
            
        except Exception as e:
            print(f"Error downloading {self.archive_filename}: {str(e)}")
            if os.path.exists(filepath):
                os.remove(filepath)
            return False

    def _extract_sample_files(self):
        """Extract only the sample files from the downloaded archive."""
        print(f"\nExtracting selected files from {self.archive_filename}...")
        
        extract_root = self.output_dir
        
        try:
            with tarfile.open(self.archive_filepath, 'r') as tar:
                members = tar.getnames()
                
                files_to_extract = [f for f in self.sample_files if f in members]
                
                if not files_to_extract:
                    print("Error: None of the sample files were found in the archive.")
                    return False
                
                for member_name in tqdm(files_to_extract, desc="Extracting files"):
                    member = tar.getmember(member_name)
                    tar.extract(member, path=extract_root)
            
            print("Extraction complete.")
            return True
        
        except Exception as e:
            print(f"Error during extraction: {str(e)}")
            return False

    def download_dataset(self, extract_all=False):
        if not self._download_archive():
            return False

        if extract_all:
            print(f"\nExtracting ALL files from {self.archive_filename}...")
            try:
                with tarfile.open(self.archive_filepath, 'r') as tar:
                    tar.extractall(path=self.output_dir)
                print("Full dataset extraction complete.")
                return True
            except Exception as e:
                print(f"Error during full extraction: {str(e)}")
                return False
        else:
            return self._extract_sample_files()

def main():
    downloader = IXIDownloader()

    success = downloader.download_dataset(extract_all=False) # Change to True for full download

    if success:
        print("Process complete!")
    else:
        print("\nProcess failed. Please check the errors above.")

if __name__ == "__main__":
    main()