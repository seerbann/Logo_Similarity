from PIL import Image
import cairosvg
import os
import matplotlib.pyplot as plt
import numpy as np

def convert_svg_to_png(svg_path, output_path):
    try:
        cairosvg.svg2png(url=svg_path, write_to=output_path)
        return output_path
    except Exception as e:
        print(f"Eroare la conversie SVG: {svg_path} -> {e}")
        return None

def convert_to_png_with_pil(input_path, output_path):
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGBA")
            img.save(output_path, format="PNG")
            return output_path
    except Exception as e:
        print(f"Eroare la conversie PIL: {input_path} -> {e}")
        return None

def remove_icc_profile(image_path, save_path=None):
    try:
        img = Image.open(image_path)
        data = list(img.getdata())
        img_no_profile = Image.new(img.mode, img.size)
        img_no_profile.putdata(data)

        if save_path is None:
            save_path = image_path

        img_no_profile.save(save_path)
    except Exception as e:
        print(f"[Clean ICC] Eroare la {image_path}: {e}")


def create_graph(timings, save_path="../timp_algoritmi.png"):

    methods = []
    descriptor_times = []
    cluster_times = []
    total_times = []

    for key in timings:
        if "descriptor" in key:
            method = key.split(" - ")[0]
            descriptor_time = timings[key]
            cluster_time = timings.get(f"{method} - cluster", 0)
            total_time = descriptor_time + cluster_time

            methods.append(method)
            descriptor_times.append(descriptor_time)
            cluster_times.append(cluster_time)
            total_times.append(total_time)

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, descriptor_times, width, label='Descriptor')
    rects2 = ax.bar(x, cluster_times, width, label='Clustering')
    rects3 = ax.bar(x + width, total_times, width, label='Total', color='gray')

    def autolabel(rects, offset=0):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}s',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3 + offset),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3, offset=5)

    ax.set_ylabel('Timp (secunde)')
    ax.set_title('Timp de procesare și clusterizare per metodă')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] Grafic salvat ca '{save_path}'")
