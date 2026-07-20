import os
import urllib.request
from src.utils import logger

def download_model_files(dest_dir="models/roberta_goemotions_model"):
    """
    Downloads the trained model files from the GitHub repository.
    """
    # Using the /raw/ path ensures we get the actual file (even for Git LFS files)
    base_url = "https://github.com/Mahesh-Awchar/MentalHealthEmotionDetection/raw/main/models/roberta_goemotions_model/"
    
    files_to_download = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt"
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    full_dest_dir = os.path.join(project_root, dest_dir)
    
    os.makedirs(full_dest_dir, exist_ok=True)
    
    logger.info(f"Downloading model files from GitHub to {full_dest_dir}...")
    
    for filename in files_to_download:
        file_url = base_url + filename
        dest_path = os.path.join(full_dest_dir, filename)
        
        try:
            logger.info(f"Downloading {filename}...")
            urllib.request.urlretrieve(file_url, dest_path)
            logger.info(f"Successfully downloaded {filename}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            
    # Try downloading model weights (safetensors first, fallback to bin)
    model_files = ["model.safetensors", "pytorch_model.bin"]
    model_downloaded = False
    for m_file in model_files:
        try:
            logger.info(f"Attempting to download {m_file}...")
            urllib.request.urlretrieve(base_url + m_file, os.path.join(full_dest_dir, m_file))
            logger.info(f"Successfully downloaded {m_file}")
            model_downloaded = True
            break
        except Exception:
            logger.warning(f"{m_file} not found or failed to download.")
            
    if not model_downloaded:
        logger.error("Failed to download model weights. Please check your internet connection or run the training script.")
    else:
        logger.info("All model files downloaded successfully! You can now run the app or predict mode.")