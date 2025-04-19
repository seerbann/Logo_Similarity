import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0,ResNet50_Weights,EfficientNet_B0_Weights
from sklearn.cluster import DBSCAN
import numpy as np
from PIL import Image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === TRANSFORMARE IMAGINI ===
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # standard ImageNet
        std=[0.229, 0.224, 0.225]
    )
])

# === MODELE ===
resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # scoatem FC
resnet.eval().to(device)

efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
efficientnet = torch.nn.Sequential(*list(efficientnet.children())[:-1])
efficientnet.eval().to(device)

# === DESCRIPTORI ===

def compute_resnet_descriptor(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = resnet(img_tensor).squeeze().cpu().numpy()
        return features
    except Exception as e:
        print(f"[ResNet] Eroare la {image_path}: {e}")
        return None

def compute_efficientnet_descriptor(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = efficientnet(img_tensor).squeeze().flatten().cpu().numpy()
        return features
    except Exception as e:
        print(f"[EffNet] Eroare la {image_path}: {e}")
        return None

# === CLUSTERING using DBSCAN just for comparing ===

def cluster_dl_features(features_dict, eps=0.5, min_samples=2):
    """
    Clustering pe baza descriptorilor extrași.
    """
    filenames = list(features_dict.keys())
    feature_vectors = np.array([features_dict[f] for f in filenames if features_dict[f] is not None])

    if len(feature_vectors) == 0:
        print("[DL] Niciun descriptor valid pentru clustering.")
        return []

    model = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = model.fit_predict(feature_vectors)

    clusters = {}
    for filename, label in zip(filenames, labels):
        if label == -1:
            continue  # -1 = outlier
        clusters.setdefault(label, []).append(filename)

    return list(clusters.values())
