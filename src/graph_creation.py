import os
import time
import argparse
from tqdm import tqdm 
from cluster_logos_orb import cluster_logos, compute_orb_descriptors
from cluster_logos_sift import cluster_logos_sift, compute_sift_descriptors
from cluster_logos_phash import cluster_logos_phash, compute_phash
from cluster_logos_orb_phash import cluster_logos_orb_phash
from cluster_logos_ssim import cluster_logos_ssim
from utils import convert_svg_to_png, convert_to_png_with_pil, remove_icc_profile,create_graph
from cluster_logos_dl import compute_resnet_descriptor, compute_efficientnet_descriptor, cluster_dl_features

# Argumente din linie de comandă
parser = argparse.ArgumentParser(description="Logo clustering using different descriptors.")
parser.add_argument('--orb', action='store_true')
parser.add_argument('--sift', action='store_true')
parser.add_argument('--phash', action='store_true')
parser.add_argument('--ssim', action='store_true')
parser.add_argument('--resnet', action='store_true')
parser.add_argument('--effnet', action='store_true')
parser.add_argument('--orbphash', action='store_true')
parser.add_argument('--all', action='store_true')
args = parser.parse_args()

LOGO_DIR = '../Logos_10000'
PROCESSED_DIR = '../Logos_raster'
os.makedirs(PROCESSED_DIR, exist_ok=True)

timings = {}
descriptors_orb = {}
descriptors_sift = {}
hashes_phash = {}
images_for_ssim = {}
resnet_features = {}
effnet_features = {}

VALID_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.svg', '.ico', '.bmp', '.tif', '.tiff', '.webp'}

start_total = time.time()
for filename in os.listdir(LOGO_DIR):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in VALID_IMAGE_EXTENSIONS:
        print(f"[SKIP] Format neacceptat: {filename}")
        continue

    filepath = os.path.join(LOGO_DIR, filename)
    png_filename = os.path.splitext(filename)[0] + '.png'
    png_path = os.path.join(PROCESSED_DIR, png_filename)

    # Convert to PNG
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
        t1 = time.time()
        desc = compute_orb_descriptors(png_path)
        timings["ORB - descriptor"] += time.time() - t1 if "ORB - descriptor" in timings else time.time() - t1
        if desc is not None:
            descriptors_orb[filename] = desc

    # SIFT
    if args.sift or args.all:
        t1 = time.time()
        desc = compute_sift_descriptors(png_path)
        timings["SIFT - descriptor"] += time.time() - t1 if "SIFT - descriptor" in timings else time.time() - t1
        if desc is not None:
            descriptors_sift[filename] = desc

    # pHash
    if args.phash or args.all:
        t1 = time.time()
        h = compute_phash(png_path)
        timings["Phash - descriptor"] += time.time() - t1 if "Phash - descriptor" in timings else time.time() - t1
        if h is not None:
            hashes_phash[filename] = h

    # SSIM
    if args.ssim or args.all:
        images_for_ssim[filename] = png_path

    # ResNet
    if args.resnet or args.all:
        t1 = time.time()
        desc = compute_resnet_descriptor(png_path)
        timings["Resnet - descriptor"] += time.time() - t1 if "Resnet - descriptor" in timings else time.time() - t1
        if desc is not None:
            resnet_features[filename] = desc

    # EfficientNet
    if args.effnet or args.all:
        t1 = time.time()
        desc = compute_efficientnet_descriptor(png_path)
        timings["Effnet - descriptor"] += time.time() - t1 if "Effnet - descriptor" in timings else time.time() - t1
        if desc is not None:
            effnet_features[filename] = desc

print(f"[INFO] Procesare imagini finalizată în {time.time() - start_total:.2f} secunde.")

# Clustering pentru fiecare metodă
def time_clustering(name, func, *args, **kwargs):
    t_start = time.time()
    clusters = func(*args, **kwargs)
    timings[f"{name} - cluster"] = time.time() - t_start
    for idx, group in enumerate(clusters, 1):
        print(f"[{name}] Grup {idx}:\n" + "\n".join([f"   - {logo}" for logo in group]))
    return clusters

if args.orb or args.all:
    clusters = time_clustering("ORB", cluster_logos, descriptors_orb, similarity_threshold=0.15)

if args.sift or args.all:
    clusters = time_clustering("SIFT", cluster_logos_sift, descriptors_sift, similarity_threshold=0.15)

if args.phash or args.all:
    clusters = time_clustering("Phash", cluster_logos_phash, hashes_phash, threshold=10)

if args.orbphash or args.all:
    clusters = time_clustering("ORB + pHash", cluster_logos_orb_phash, descriptors_orb, hashes_phash, orb_threshold=0.15, phash_threshold=10)

if args.ssim or args.all:
    t_start = time.time()
    clusters = []
    for group in tqdm(cluster_logos_ssim(images_for_ssim, threshold=0.80), desc="Clustering SSIM", unit="grup"):
        clusters.append(group)
    timings["SSIM - cluster"] = time.time() - t_start
    for idx, group in enumerate(clusters, 1):
        print(f"[SSIM] Grup {idx}:\n" + "\n".join([f"   - {logo}" for logo in group]))

if args.resnet or args.all:
    clusters = time_clustering("Resnet", cluster_dl_features, resnet_features, eps=0.5, min_samples=2)

if args.effnet or args.all:
    clusters = time_clustering("Effnet", cluster_dl_features, effnet_features, eps=0.5, min_samples=2)

# Afișează graficul
create_graph(timings)
