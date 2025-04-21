import os
import time
import argparse
from tqdm import tqdm 
from cluster_logos_orb import cluster_logos, compute_orb_descriptors
from cluster_logos_sift import cluster_logos_sift, compute_sift_descriptors
from cluster_logos_phash import cluster_logos_phash, compute_phash
from cluster_logos_orb_phash import cluster_logos_orb_phash
from cluster_logos_ssim import cluster_logos_ssim
from utils import convert_svg_to_png, convert_to_png_with_pil, remove_icc_profile
from cluster_logos_dl import compute_resnet_descriptor, compute_efficientnet_descriptor, cluster_dl_features

# Argument parsing
parser = argparse.ArgumentParser(description="Logo clustering using different descriptors.")
parser.add_argument('--orb', action='store_true', help='Use ORB descriptor')
parser.add_argument('--sift', action='store_true', help='Use SIFT descriptor')
parser.add_argument('--phash', action='store_true', help='Use pHash descriptor')
parser.add_argument('--ssim', action='store_true', help='Use SSIM metric')
parser.add_argument('--resnet', action='store_true', help='Use ResNet descriptor')
parser.add_argument('--effnet', action='store_true', help='Use EfficientNet descriptor')
parser.add_argument('--orbphash', action='store_true', help='Use combined ORB + pHash clustering')
parser.add_argument('--all', action='store_true', help='Run all methods')
args = parser.parse_args()

# Paths and setup
LOGO_DIR = '../Logos'
PROCESSED_DIR = '../Logos_raster'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Storage for features
descriptors_orb = {}
descriptors_sift = {}
hashes_phash = {}
images_for_ssim = {}
resnet_features = {}
effnet_features = {}

VALID_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.svg', '.ico', '.bmp', '.tif', '.tiff'}

# Process images
start_total = time.time()
for filename in os.listdir(LOGO_DIR):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in VALID_IMAGE_EXTENSIONS:
        print(f"[SKIP] Format neacceptat: {filename}")
        continue

    filepath = os.path.join(LOGO_DIR, filename)
    png_filename = os.path.splitext(filename)[0] + '.png'
    png_path = os.path.join(PROCESSED_DIR, png_filename)

    # Convert to PNG if needed
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
    if args.orb or args.all:
        desc = compute_orb_descriptors(png_path)
        if desc is not None:
            descriptors_orb[filename] = desc

    # SIFT
    if args.sift or args.all:
        desc = compute_sift_descriptors(png_path)
        if desc is not None:
            descriptors_sift[filename] = desc

    # pHash
    if args.phash or args.all:
        h = compute_phash(png_path)
        if h is not None:
            hashes_phash[filename] = h

    # SSIM
    if args.ssim or args.all:
        images_for_ssim[filename] = png_path

    # ResNet
    if args.resnet or args.all:
        desc = compute_resnet_descriptor(png_path)
        if desc is not None:
            resnet_features[filename] = desc

    # EfficientNet
    if args.effnet or args.all:
        desc = compute_efficientnet_descriptor(png_path)
        if desc is not None:
            effnet_features[filename] = desc

print(f"[INFO] Procesare imagini finalizată în {time.time() - start_total:.2f} secunde.")

# ORB Clustering
if args.orb or args.all:
    start = time.time()
    clusters = cluster_logos(descriptors_orb, similarity_threshold=0.15)
    print(f"[ORB] Clustering terminat în {time.time() - start:.2f} secunde.")
    for idx, group in enumerate(clusters, 1):
        print(f"[ORB] Grup {idx}:\n" + "\n".join([f"   - {logo}" for logo in group]))

# SIFT Clustering
if args.sift or args.all:
    start = time.time()
    clusters = cluster_logos_sift(descriptors_sift, similarity_threshold=0.15)
    print(f"[SIFT] Clustering terminat în {time.time() - start:.2f} secunde.")
    for idx, group in enumerate(clusters, 1):
        print(f"[SIFT] Grup {idx}:\n" + "\n".join([f"   - {logo}" for logo in group]))

# pHash Clustering
if args.phash or args.all:
    start = time.time()
    clusters = cluster_logos_phash(hashes_phash, threshold=10)
    print(f"[pHash] Clustering terminat în {time.time() - start:.2f} secunde.")
    for idx, group in enumerate(clusters, 1):
        print(f"[pHash] Grup {idx}:\n" + "\n".join([f"   - {logo}" for logo in group]))

# ORB + pHash Clustering
if args.orbphash or args.all:
    start = time.time()
    clusters = cluster_logos_orb_phash(descriptors_orb, hashes_phash, orb_threshold=0.15, phash_threshold=10)
    print(f"[ORB + pHash] Clustering terminat în {time.time() - start:.2f} secunde.")
    for idx, group in enumerate(clusters, 1):
        print(f"[ORB + pHash] Grup {idx}:\n" + "\n".join([f"   - {logo}" for logo in group]))

#SSIM clustering
if args.ssim or args.all:
    start = time.time()
    clusters = []
    for group in tqdm(cluster_logos_ssim(images_for_ssim, threshold=0.80), desc="Clustering SSIM", unit="grup"):
        clusters.append(group)
    print(f"[SSIM] Clustering terminat în {time.time() - start:.2f} secunde.")
    for idx, group in enumerate(clusters, 1):
        print(f"[SSIM] Grup {idx}:\n" + "\n".join([f"   - {logo}" for logo in group]))

# ResNet Clustering
if args.resnet or args.all:
    start = time.time()
    clusters = cluster_dl_features(resnet_features, eps=0.5, min_samples=2)
    print(f"[ResNet] Clustering terminat în {time.time() - start:.2f} secunde.")
    for idx, group in enumerate(clusters, 1):
        print(f"[ResNet] Grup {idx}:\n" + "\n".join([f"   - {logo}" for logo in group]))

# EfficientNet Clustering
if args.effnet or args.all:
    start = time.time()
    clusters = cluster_dl_features(effnet_features, eps=0.5, min_samples=2)
    print(f"[EffNet] Clustering terminat în {time.time() - start:.2f} secunde.")
    for idx, group in enumerate(clusters, 1):
        print(f"[EffNet] Grup {idx}:\n" + "\n".join([f"   - {logo}" for logo in group]))
