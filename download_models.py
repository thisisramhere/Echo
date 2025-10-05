import os
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import gdown
import shutil

# --- Model Definitions ---
# This dictionary now only contains models for Whisper and OpenVoice. Wav2Lip has been removed.
MODELS = {
    "whisper": {
        "type": "requests",
        "url": "https://openaipublic.blob.core.windows.net/openai-whisper/models/ed3a5b64c0b3b3a6f1da2e49673d4ee6296c3b5f274e76632434e86e1db126c0/base.pt",
        "path": os.path.join(os.path.expanduser("~"), ".cache", "whisper", "base.pt")
    },
    "openvoice_base_config": {
        "type": "hf",
        "repo_id": "myshell-ai/OpenVoice",
        "filename": "checkpoints/base_speakers/EN/config.json",
        "path": os.path.join("checkpoints", "base_speakers", "EN", "config.json")
    },
    "openvoice_base_tts": {
        "type": "hf",
        "repo_id": "myshell-ai/OpenVoice",
        "filename": "checkpoints/base_speakers/EN/en_base_speaker_tts.pth",
        "path": os.path.join("checkpoints", "base_speakers", "EN", "en_base_speaker_tts.pth")
    },
    "openvoice_base_se": {
        "type": "hf",
        "repo_id": "myshell-ai/OpenVoice",
        "filename": "checkpoints/base_speakers/EN/en_default_se.pth",
        "path": os.path.join("checkpoints", "base_speakers", "EN", "en_default_se.pth")
    },
    "openvoice_converter_config": {
        "type": "hf",
        "repo_id": "myshell-ai/OpenVoice",
        "filename": "checkpoints/converter/config.json",
        "path": os.path.join("checkpoints", "converter", "config.json")
    },
    "openvoice_converter": {
        "type": "hf",
        "repo_id": "myshell-ai/OpenVoice",
        "filename": "checkpoints/converter/converter.pth",
        "path": os.path.join("checkpoints", "converter", "converter.pth")
    }
}

# --- Download Utilities ---

def download_file_requests(url, filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        print(f"✔️ File already exists: {filepath}")
        return
    try:
        print(f"Downloading {os.path.basename(filepath)}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(filepath, 'wb') as f, tqdm(
            desc=os.path.basename(filepath), total=total_size, unit='iB',
            unit_scale=True, unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        print(f"✅ Download complete: {filepath}")
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")

def download_file_hf(repo_id, filename, local_path):
    directory = os.path.dirname(local_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if os.path.exists(local_path):
        print(f"✔️ File already exists: {local_path}")
        return
    try:
        print(f"Downloading {os.path.basename(local_path)} from Hugging Face...")
        temp_path = hf_hub_download(repo_id=repo_id, filename=filename)
        shutil.move(temp_path, local_path)
        print(f"✅ Download complete: {local_path}")
    except Exception as e:
        print(f"❌ Error downloading from Hugging Face {repo_id}/{filename}: {e}")

def download_file_gdown(file_id, filepath, filename=None):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        print(f"✔️ File already exists: {filepath}")
        return
    try:
        print(f"Downloading {os.path.basename(filepath)} from Google Drive...")
        if filename: # If it's a file in a folder
             gdown.download_folder(id=file_id, quiet=False, use_cookies=False)
             # gdown downloads folders to the current dir, find the file and move it
             downloaded_file_path = os.path.join(filename) # Assumes it lands in root
             if os.path.exists(downloaded_file_path):
                 shutil.move(downloaded_file_path, filepath)
                 print(f"✅ Download complete: {filepath}")
             else: # Fallback for nested structures
                 found = False
                 for root, dirs, files in os.walk("."):
                     if filename in files:
                         shutil.move(os.path.join(root, filename), filepath)
                         found = True
                         break
                 if not found:
                     print(f"❌ Could not find {filename} after GDrive folder download.")

        else: # If it's a direct file ID
            gdown.download(id=file_id, output=filepath, quiet=False)
            print(f"✅ Download complete: {filepath}")

    except Exception as e:
        print(f"❌ Error downloading from Google Drive ({file_id}): {e}")


# --- Main Script ---
if __name__ == "__main__":
    print("--- Pre-downloading all required AI models for E.C.H.O. ---")
    
    for model_name, model_info in MODELS.items():
        if model_info["type"] == "requests":
            download_file_requests(model_info["url"], model_info["path"])
        elif model_info["type"] == "hf":
            download_file_hf(model_info["repo_id"], model_info["filename"], model_info["path"])
        elif model_info["type"] == "gdown":
            download_file_gdown(model_info["id"], model_info["path"], model_info.get("filename"))

    print("\n--- All models checked. You can now run api.py ---")

