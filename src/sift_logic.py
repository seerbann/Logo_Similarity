import cv2

def compute_sift_descriptors(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return descriptors
    except Exception as e:
        print(f"[SIFT] Eroare la {image_path}: {e}")
        return None
