import sys
import json
import os
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np

def generate_normal_map(image, strength=2.0):
    gray = image.convert('L')
    gray_np = np.array(gray, dtype='float32')
    dx = np.gradient(gray_np, axis=1)
    dy = np.gradient(gray_np, axis=0)
    dz = np.ones_like(dx) * (255.0 / strength)

    normal = np.stack((dx, dy, dz), axis=-1)
    normal = (normal - normal.min()) / (normal.max() - normal.min()) * 255.0
    normal_img = Image.fromarray(normal.astype('uint8'))
    return normal_img

def generate_bump_map(image, strength=1.5):
    enhancer = ImageEnhance.Contrast(image.convert('L'))
    bump = enhancer.enhance(strength)
    return bump

def generate_metallic_map(image, strength=1.0):
    inverted = ImageOps.invert(image.convert('L'))
    metallic = ImageEnhance.Brightness(inverted).enhance(strength)
    return metallic

def generate_occlusion_map(image, strength=1.0):
    blurred = image.convert('L').filter(ImageFilter.GaussianBlur(radius=2))
    occlusion = ImageEnhance.Contrast(blurred).enhance(strength)
    return occlusion

def generate_roughness_map(image, strength=1.0):
    gray = image.convert('L')
    roughness = ImageOps.autocontrast(gray.filter(ImageFilter.EDGE_ENHANCE))
    roughness = ImageEnhance.Sharpness(roughness).enhance(strength)
    return roughness

def generate_emission_map(image, strength=2.0):
    gray = image.convert('L')
    bright = ImageEnhance.Brightness(gray).enhance(strength)
    emission = ImageOps.colorize(bright, black="black", white="cyan")
    return emission

def save_output(img, name, output_dir):
    output_path = os.path.join(output_dir, f"{name}.png")
    img.save(output_path)
    return output_path

def main():
    if len(sys.argv) < 4:
        print(json.dumps({"error": "Invalid arguments"}))
        sys.exit(1)

    image_path = sys.argv[1]
    selected_maps = json.loads(sys.argv[2])
    settings = json.loads(sys.argv[3])

    image = Image.open(image_path).convert("RGB")
    output_dir = os.path.join(os.path.dirname(image_path), "generated_maps")
    os.makedirs(output_dir, exist_ok=True)

    result_paths = {}

    if selected_maps.get("normal"):
        strength = settings.get("normal_strength", 2.0)
        normal_map = generate_normal_map(image, strength)
        result_paths["normal"] = save_output(normal_map, "normal_map", output_dir)

    if selected_maps.get("bump"):
        strength = settings.get("bump_strength", 1.5)
        bump_map = generate_bump_map(image, strength)
        result_paths["bump"] = save_output(bump_map, "bump_map", output_dir)

    if selected_maps.get("metallic"):
        strength = settings.get("metallic_strength", 1.0)
        metallic_map = generate_metallic_map(image, strength)
        result_paths["metallic"] = save_output(metallic_map, "metallic_map", output_dir)

    if selected_maps.get("occlusion"):
        strength = settings.get("occlusion_strength", 1.0)
        occlusion_map = generate_occlusion_map(image, strength)
        result_paths["occlusion"] = save_output(occlusion_map, "occlusion_map", output_dir)

    if selected_maps.get("roughness"):
        strength = settings.get("roughness_strength", 1.0)
        roughness_map = generate_roughness_map(image, strength)
        result_paths["roughness"] = save_output(roughness_map, "roughness_map", output_dir)

    if selected_maps.get("emission"):
        strength = settings.get("emission_strength", 2.0)
        emission_map = generate_emission_map(image, strength)
        result_paths["emission"] = save_output(emission_map, "emission_map", output_dir)

    print(json.dumps({"success": True, "outputs": result_paths}))

if __name__ == "__main__":
    main()
