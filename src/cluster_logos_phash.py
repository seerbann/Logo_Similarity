from phash_logic import hamming_distance

def cluster_logos_phash(hashes_dict, threshold=10):
    """
    Grupează logouri pe baza similarității pHash.

    :param hashes_dict: dict {filename: phash}
    :param threshold: pragul maxim de distanță Hamming pentru a considera două imagini similare
    :return: listă de grupuri (fiecare grup e o listă de fișiere)
    """
    visited = set()
    clusters = []

    keys = list(hashes_dict.keys())
    for i in range(len(keys)):
        if keys[i] in visited:
            continue

        current_group = [keys[i]]
        visited.add(keys[i])

        for j in range(i + 1, len(keys)):
            if keys[j] in visited:
                continue

            dist = hamming_distance(hashes_dict[keys[i]], hashes_dict[keys[j]])
            if dist <= threshold:
                current_group.append(keys[j])
                visited.add(keys[j])

        clusters.append(current_group)

    return clusters
