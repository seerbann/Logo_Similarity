import cv2

def compare_descriptors_sift(desc1, desc2, ratio_threshold=0.75):
    if desc1 is None or desc2 is None:
        return 0, 0.0

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    score = len(good_matches)
    normalized_score = score / max(len(desc1), len(desc2))
    return score, normalized_score

def cluster_logos_sift(descriptors_dict, similarity_threshold=0.15):
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
            _, sim_score = compare_descriptors_sift(descriptors_dict[logo1], descriptors_dict[logo2])
            if sim_score >= similarity_threshold:
                cluster.append(logo2)
                visited.add(logo2)

        clusters.append(cluster)

    return clusters
