import cv2

def compute_orb_descriptors(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)
        return descriptors
    except Exception as e:
        print(f"[ORB] Eroare la {image_path}: {e}")
        return None
