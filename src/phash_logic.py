from PIL import Image
import imagehash

def compute_phash(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        return imagehash.phash(img)
    except Exception as e:
        print(f"[pHash] Eroare la {image_path}: {e}")
        return None

def hamming_distance(hash1, hash2):
    return hash1 - hash2

def are_similar_phash(hash1, hash2, threshold=10):
    return hamming_distance(hash1, hash2) <= threshold
