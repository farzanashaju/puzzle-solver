"""
Download a small dataset of sample images for jigsaw puzzle solving
"""
import urllib.request
import os
from pathlib import Path

def download_sample_images():
    """Download sample images from public sources"""
    
    # Create dataset directory
    dataset_dir = Path('dataset')
    dataset_dir.mkdir(exist_ok=True)
    
    # Sample images from Unsplash (small resolution, free to use)
    # Using Unsplash's source API for random small images
    images = [
        ('https://picsum.photos/512/512?random=1', 'image_1.jpg'),
        ('https://picsum.photos/512/512?random=2', 'image_2.jpg'),
        ('https://picsum.photos/512/512?random=3', 'image_3.jpg'),
        ('https://picsum.photos/512/512?random=4', 'image_4.jpg'),
        ('https://picsum.photos/512/512?random=5', 'image_5.jpg'),
    ]
    
    print("Downloading sample images...")
    print("=" * 50)
    
    downloaded_files = []
    
    for url, filename in images:
        filepath = dataset_dir / filename
        
        try:
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"✓ Saved to {filepath}")
            downloaded_files.append(str(filepath))
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
    
    print("=" * 50)
    print(f"Downloaded {len(downloaded_files)} images successfully!")
    print(f"Images saved in: {dataset_dir.absolute()}")
    
    return downloaded_files

if __name__ == "__main__":
    download_sample_images()
