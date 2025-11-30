"""
Download a small dataset of sample images for jigsaw puzzle solving
"""
import urllib.request
import os
from pathlib import Path

def download_sample_images():
    dataset_dir = Path('dataset')
    dataset_dir.mkdir(exist_ok=True)
    
    num_images = 5

    images = [
        (f'https://picsum.photos/512/512?random={i}', f'image_{i}.jpg')
        for i in range(1, num_images + 1)
    ]
    
    downloaded_files = []
    
    for url, filename in images:
        filepath = dataset_dir / filename
        
        try:
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            downloaded_files.append(str(filepath))
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
    
    return downloaded_files

if __name__ == "__main__":
    download_sample_images()