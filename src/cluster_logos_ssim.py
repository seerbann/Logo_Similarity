import cv2
from skimage.metrics import structural_similarity as ssim

def compute_ssim(image_path1, image_path2, resize_to=(300, 300)):
    try:
        print(f"[SSIM] Comparing {image_path1} with {image_path2}")

        img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            return 0.0

        img1 = cv2.resize(img1, resize_to)
        img2 = cv2.resize(img2, resize_to)

        score, _ = ssim(img1, img2, full=True)
        return score
    except Exception as e:
        print(f"[SSIM ERROR] {image_path1} vs {image_path2} -> {e}")
        return 0.0



def cluster_logos_ssim(images_dict, threshold=0.75):
    """
    Grupare imagini pe baza scorului SSIM.

    :param images_dict: dict {filename: image_path}
    :param threshold: pragul minim SSIM pentru a considera două imagini similare
    :return: listă de grupuri (fiecare grup e o listă de fișiere)
    """
    visited = set()
    clusters = []
    keys = list(images_dict.keys())

    for i in range(len(keys)):
        if keys[i] in visited:
            continue

        current_group = [keys[i]]
        visited.add(keys[i])

        for j in range(i + 1, len(keys)):
            if keys[j] in visited:
                continue

            score = compute_ssim(images_dict[keys[i]], images_dict[keys[j]])
            if score >= threshold:
                current_group.append(keys[j])
                visited.add(keys[j])

        clusters.append(current_group)

    return clusters
