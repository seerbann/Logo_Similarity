from PIL import Image
import imagehash

#clustering
def cluster_logos_phash(hashes_dict, threshold=10):
    """
    Grupează logouri pe baza similarității pHash.

    :param hashes_dict: dict {filename: phash}
    :param threshold: pragul maxim de distanță Hamming pentru a considera două imagini similare
    :return: listă de grupuri (fiecare grup e o listă de fișiere)
    """
    visited = set()
    clusters = []

    keys = list(hashes_dict.keys())
    for i in range(len(keys)):
        if keys[i] in visited:
            continue

        current_group = [keys[i]]
        visited.add(keys[i])

        for j in range(i + 1, len(keys)):
            if keys[j] in visited:
                continue

            dist = hamming_distance(hashes_dict[keys[i]], hashes_dict[keys[j]])
            if dist <= threshold:
                current_group.append(keys[j])
                visited.add(keys[j])

        clusters.append(current_group)

    return clusters


# hash computing

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
