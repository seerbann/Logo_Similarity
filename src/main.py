import os
from cluster_logos_orb import cluster_logos, compute_orb_descriptors
from cluster_logos_sift import cluster_logos_sift, compute_sift_descriptors
from cluster_logos_phash import cluster_logos_phash, compute_phash
from cluster_logos_orb_phash import cluster_logos_orb_phash
from cluster_logos_ssim import cluster_logos_ssim
from utils import convert_svg_to_png, convert_to_png_with_pil,remove_icc_profile

LOGO_DIR = '../Logos'
PROCESSED_DIR = '../Logos_raster'
os.makedirs(PROCESSED_DIR, exist_ok=True)

descriptors_orb = {}
descriptors_sift = {}
hashes_phash = {}
images_for_ssim = {}

VALID_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.svg', '.ico', '.bmp', '.tif', '.tiff'}

for filename in os.listdir(LOGO_DIR):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in VALID_IMAGE_EXTENSIONS:
        print(f"[SKIP] Format neacceptat: {filename}")
        continue

    filepath = os.path.join(LOGO_DIR, filename)
    png_filename = os.path.splitext(filename)[0] + '.png'
    png_path = os.path.join(PROCESSED_DIR, png_filename)

    # Conversie după tip
    if ext == '.svg':
        if not os.path.exists(png_path):
            convert_svg_to_png(filepath, png_path)
    elif ext != '.png':
        if not os.path.exists(png_path):
            convert_to_png_with_pil(filepath, png_path)
    else:
        png_path = filepath

    remove_icc_profile(png_path)
    # ORB
    orb_desc = compute_orb_descriptors(png_path)
    if orb_desc is not None:
        descriptors_orb[filename] = orb_desc

    # SIFT
    sift_desc = compute_sift_descriptors(png_path)
    if sift_desc is not None:
        descriptors_sift[filename] = sift_desc

    # pHash
    phash = compute_phash(png_path)
    if phash is not None:
        hashes_phash[filename] = phash

    # SSIM
    images_for_ssim[filename] = png_path


print(f"[INFO - ORB] Descriptorii ORB calculați pentru {len(descriptors_orb)} logouri.")
print(f"[INFO - SIFT] Descriptorii SIFT calculați pentru {len(descriptors_sift)} logouri.")
print(f"[INFO - pHash] Hash-urile pHash calculate pentru {len(hashes_phash)} logouri.")
print(f"[INFO - SSIM] Imagini pregătite pentru SSIM: {len(images_for_ssim)}")

# Clustering ORB
clusters = cluster_logos(descriptors_orb, similarity_threshold=0.15)
for idx, group in enumerate(clusters, 1):
    print(f"[ORB] Grup {idx}:")
    for logo in group:
        print(f"   - {logo}")

# Clustering SIFT
clusters_sift = cluster_logos_sift(descriptors_sift, similarity_threshold=0.15)
for idx, group in enumerate(clusters_sift, 1):
    print(f"[SIFT] Grup {idx}:")
    for logo in group:
        print(f"   - {logo}")

# Clustering pHash
clusters_phash = cluster_logos_phash(hashes_phash, threshold=10)
for idx, group in enumerate(clusters_phash, 1):
    print(f"[pHash] Grup {idx}:")
    for logo in group:
        print(f"   - {logo}")

# Clustering ORB + pHash
clusters_orb_phash = cluster_logos_orb_phash(descriptors_orb, hashes_phash, orb_threshold=0.15, phash_threshold=10)
for idx, group in enumerate(clusters_orb_phash, 1):
    print(f"[ORB + pHash] Grup {idx}:")
    for logo in group:
        print(f"   - {logo}")

# Clustering SSIM - very slow
'''
clusters_ssim = cluster_logos_ssim(images_for_ssim, threshold=0.80)
for idx, group in enumerate(clusters_ssim, 1):
    print(f"[SSIM] Grup {idx}:")
    for logo in group:
        print(f"   - {logo}")
'''