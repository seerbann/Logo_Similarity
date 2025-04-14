import os
import cv2
import numpy as np
from PIL import Image
import cairosvg
from cluster_logos_orb import cluster_logos
LOGO_DIR = 'Logos'
PROCESSED_DIR = 'Logos_raster'  # unde salvăm PNG-urile procesate
os.makedirs(PROCESSED_DIR, exist_ok=True)

def convert_svg_to_png(svg_path, output_path):
    try:
        cairosvg.svg2png(url=svg_path, write_to=output_path)
        return output_path
    except Exception as e:
        print(f"Eroare la conversie SVG: {svg_path} -> {e}")
        return None

def compute_orb_descriptors(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return descriptors
    except Exception as e:
        print(f"Eroare ORB la {image_path}: {e}")
        return None

def compare_descriptors(desc1, desc2, ratio_threshold=0.75):
    """
    Compara doi vectori de descriptor ORB si returneaza un scor de similaritate.
    
    - desc1, desc2: descriptorii ORB (np.ndarray)
    - ratio_threshold: prag pentru good match (Lowes ratio test)
    
    Returneaza:
    - numar de match-uri bune
    - scor de similaritate (0.0 - 1.0, optional)
    """
    if desc1 is None or desc2 is None:
        return 0, 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    score = len(good_matches)
    normalized_score = score / max(len(desc1), len(desc2))  # scor intre 0.0 si 1.0
    return score, normalized_score

# === Procesam toate logourile ===
descriptors_dict = {}

for filename in os.listdir(LOGO_DIR):
    filepath = os.path.join(LOGO_DIR, filename)

    # Convertim SVG -> PNG dacă e cazul
    if filename.lower().endswith('.svg'):
        png_filename = os.path.splitext(filename)[0] + '.png'
        png_path = os.path.join(PROCESSED_DIR, png_filename)
        if not os.path.exists(png_path):
            convert_svg_to_png(filepath, png_path)
        img_path = png_path
    else:
        img_path = filepath

    # Calculăm descriptorii ORB
    descriptors = compute_orb_descriptors(img_path)
    if descriptors is not None:
        descriptors_dict[filename] = descriptors

print(f"Descriptorii ORB au fost calculați pentru {len(descriptors_dict)} logouri.")

clusters = cluster_logos(descriptors_dict, similarity_threshold=0.15)
for idx, group in enumerate(clusters, 1):
    print(f"Grup {idx}:")
    for logo in group:
        print(f"   - {logo}")