import sys
import os
import json
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

def generate_map(image, map_type, strength=1.0):
    image = image.convert('RGB')
    arr = np.array(image).astype(np.float32) / 255.0

    if map_type == 'normal':
        gray = image.convert('L').filter(ImageFilter.FIND_EDGES)
        return ImageEnhance.Contrast(gray).enhance(strength)

    elif map_type == 'bump':
        gray = image.convert('L')
        return ImageEnhance.Contrast(gray).enhance(strength)

    elif map_type == 'metallic':
        channel = Image.fromarray((arr[:, :, 0] * 255).astype('uint8'))
        return ImageEnhance.Contrast(channel).enhance(strength)

    elif map_type == 'occlusion':
        gray = image.convert('L')
        return ImageEnhance.Brightness(gray).enhance(1 / (strength + 0.01))

    elif map_type == 'roughness':
        inverted = ImageOps.invert(image.convert('L'))
        return ImageEnhance.Contrast(inverted).enhance(strength)

    elif map_type == 'height':
        edges = image.convert('L').filter(ImageFilter.CONTOUR)
        return ImageEnhance.Contrast(edges).enhance(strength)

    elif map_type == 'specular':
        enhancer = ImageEnhance.Brightness(image.convert('L'))
        return enhancer.enhance(strength)

    elif map_type == 'emissive':
        arr *= strength
        arr[arr > 1] = 1
        return Image.fromarray((arr * 255).astype('uint8'))

    else:
        raise ValueError(f"Unsupported map type: {map_type}")

def main():
    image_path = sys.argv[1]
    selected_maps = json.loads(sys.argv[2])
    settings = json.loads(sys.argv[3])
    
    image = Image.open(image_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(os.path.dirname(image_path), f"{basename}_maps")
    os.makedirs(output_dir, exist_ok=True)

    result = {}

    for map_type, selected in selected_maps.items():
        if selected:
            strength = float(settings.get(map_type, 1.0))
            try:
                map_img = generate_map(image, map_type, strength)
                filename = f"{basename}_{map_type}.png"
                filepath = os.path.join(output_dir, filename)
                map_img.save(filepath)
                result[map_type] = filepath
            except Exception as e:
                result[map_type] = f"Error: {str(e)}"

    print(json.dumps(result))

if __name__ == "__main__":
    main()
