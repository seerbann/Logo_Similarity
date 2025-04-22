import cv2
import itertools
from tqdm import tqdm 
from skimage.metrics import structural_similarity as ssim
from multiprocessing import Pool, cpu_count

def compute_ssim_pair(pair):
    key1, key2, path1, path2 = pair
    try:
        #print(f"[SSIM] Comparing {path1} with {path2}")
        img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            return (key1, key2, 0.0)

        img1 = cv2.resize(img1, (300, 300))
        img2 = cv2.resize(img2, (300, 300))

        score, _ = ssim(img1, img2, full=True)
        return (key1, key2, score)
    except Exception as e:
        print(f"[SSIM ERROR] {path1} vs {path2} -> {e}")
        return (key1, key2, 0.0)

def cluster_logos_ssim(images_dict, threshold=0.75):
    """
    Grupare imagini pe baza scorului SSIM cu multiprocessing.

    :param images_dict: dict {filename: image_path}
    :param threshold: pragul minim SSIM pentru a considera două imagini similare
    :return: listă de grupuri (fiecare grup e o listă de fișiere)
    """
    print("[SSIM] Generating image pairs...")
    keys = list(images_dict.keys())
    pairs = [
        (key1, key2, images_dict[key1], images_dict[key2])
        for i, key1 in enumerate(keys)
        for j, key2 in enumerate(keys)
        if j > i
    ]

    print(f"[SSIM] Total pairs to compare: {len(pairs)}")
    
    # Rulează comparațiile în paralel
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(compute_ssim_pair, pairs), total=len(pairs), desc="Comparații SSIM"))

    # Creează graful de similarități
    similarity_graph = {key: set() for key in keys}
    for key1, key2, score in results:
        if score >= threshold:
            similarity_graph[key1].add(key2)
            similarity_graph[key2].add(key1)

    # Creează grupurile (component connected)
    visited = set()
    clusters = []

    for key in keys:
        if key in visited:
            continue
        group = []
        stack = [key]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            group.append(current)
            stack.extend(similarity_graph[current] - visited)
        clusters.append(group)

    return clusters
