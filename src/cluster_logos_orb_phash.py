import cv2
from cluster_logos_phash import hamming_distance

def compare_descriptors(desc1, desc2, ratio_threshold=0.75):
    if desc1 is None or desc2 is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = []

    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    return len(good_matches) / max(len(desc1), len(desc2))

def cluster_logos_orb_phash(descriptors_dict, phash_dict, orb_threshold=0.15, phash_threshold=10):
    logos = list(descriptors_dict.keys())
    visited = set()
    clusters = []

    for i, logo1 in enumerate(logos):
        if logo1 in visited:
            continue

        cluster = [logo1]
        visited.add(logo1)

        for j in range(i + 1, len(logos)):
            logo2 = logos[j]
            if logo2 in visited:
                continue

            orb_score = compare_descriptors(descriptors_dict[logo1], descriptors_dict[logo2])
            phash_dist = hamming_distance(phash_dict.get(logo1), phash_dict.get(logo2))

            if orb_score >= orb_threshold and phash_dist <= phash_threshold:
                cluster.append(logo2)
                visited.add(logo2)

        clusters.append(cluster)

    return clusters
