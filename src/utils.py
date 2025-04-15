from PIL import Image
import cairosvg
import os

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
