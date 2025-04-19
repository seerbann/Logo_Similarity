import os
import argparse
from cluster_logos_orb import cluster_logos, compute_orb_descriptors
from cluster_logos_sift import cluster_logos_sift, compute_sift_descriptors
from cluster_logos_phash import cluster_logos_phash, compute_phash
from cluster_logos_orb_phash import cluster_logos_orb_phash
from cluster_logos_ssim import cluster_logos_ssim
from utils import convert_svg_to_png, convert_to_png_with_pil,remove_icc_profile
from cluster_logos_dl import compute_resnet_descriptor, compute_efficientnet_descriptor, cluster_dl_features

#argument parsing
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

# global vars
LOGO_DIR = '../Logos'
PROCESSED_DIR = '../Logos_raster'
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 'classic' feature descriptors
descriptors_orb = {}
descriptors_sift = {}
hashes_phash = {}
images_for_ssim = {}

# DeepLearning pre-trained feature descriptors
resnet_features = {}
effnet_features = {}

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
    if args.orb or args.all:
        orb_desc = compute_orb_descriptors(png_path)
        if orb_desc is not None:
            descriptors_orb[filename] = orb_desc

    # SIFT
    if args.sift or args.all:
        sift_desc = compute_sift_descriptors(png_path)
        if sift_desc is not None:
            descriptors_sift[filename] = sift_desc

    # pHash
    if args.phash or args.all:
        phash = compute_phash(png_path)
        if phash is not None:
            hashes_phash[filename] = phash

    # SSIM
    if args.ssim or args.all:
        images_for_ssim[filename] = png_path

    # ResNet
    if args.resnet or args.all:
        resnet_desc = compute_resnet_descriptor(png_path)
        if resnet_desc is not None:
            resnet_features[filename] = resnet_desc

    # EfficientNet
    if args.effnet or args.all:
        effnet_desc = compute_efficientnet_descriptor(png_path)
        if effnet_desc is not None:
            effnet_features[filename] = effnet_desc


print(f"[INFO - ORB] Descriptorii ORB calculați pentru {len(descriptors_orb)} logouri.")
print(f"[INFO - SIFT] Descriptorii SIFT calculați pentru {len(descriptors_sift)} logouri.")
print(f"[INFO - pHash] Hash-urile pHash calculate pentru {len(hashes_phash)} logouri.")
print(f"[INFO - SSIM] Imagini pregătite pentru SSIM: {len(images_for_ssim)}")

# Clustering ORB
if args.orb or args.all:
    clusters = cluster_logos(descriptors_orb, similarity_threshold=0.15)
    for idx, group in enumerate(clusters, 1):
        print(f"[ORB] Grup {idx}:")
        for logo in group:
            print(f"   - {logo}")

# Clustering SIFT
if args.sift or args.all:
    clusters_sift = cluster_logos_sift(descriptors_sift, similarity_threshold=0.15)
    for idx, group in enumerate(clusters_sift, 1):
        print(f"[SIFT] Grup {idx}:")
        for logo in group:
            print(f"   - {logo}")

# Clustering pHash
if args.phash or args.all:
    clusters_phash = cluster_logos_phash(hashes_phash, threshold=10)
    for idx, group in enumerate(clusters_phash, 1):
        print(f"[pHash] Grup {idx}:")
        for logo in group:
            print(f"   - {logo}")

# Clustering ORB + pHash
if args.orbphash or args.all:
    clusters_orb_phash = cluster_logos_orb_phash(descriptors_orb, hashes_phash, orb_threshold=0.15, phash_threshold=10)
    for idx, group in enumerate(clusters_orb_phash, 1):
        print(f"[ORB + pHash] Grup {idx}:")
        for logo in group:
            print(f"   - {logo}")

# Clustering SSIM - very slow
if args.ssim or args.all:
    clusters_ssim = cluster_logos_ssim(images_for_ssim, threshold=0.80)
    for idx, group in enumerate(clusters_ssim, 1):
        print(f"[SSIM] Grup {idx}:")
        for logo in group:
            print(f"   - {logo}")


# Clustering ResNet
if args.resnet or args.all:
    clusters_resnet = cluster_dl_features(resnet_features, eps=0.5, min_samples=2)
    for idx, group in enumerate(clusters_resnet, 1):
        print(f"[ResNet] Grup {idx}:")
        for logo in group:
            print(f"   - {logo}")

# Clustering EfficientNet
if args.effnet or args.all:
    clusters_effnet = cluster_dl_features(effnet_features, eps=0.5, min_samples=2)
    for idx, group in enumerate(clusters_effnet, 1):
        print(f"[EffNet] Grup {idx}:")
        for logo in group:
            print(f"   - {logo}")
