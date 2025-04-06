import sys
import os
import json
import numpy as np
import cv2
from PIL import Image
import argparse
from pathlib import Path
from channel_packing import pack_maps_into_channels

def process_image(image_path, selected_maps, settings=None):
    """
    Process the input image and generate the selected texture maps.
    
    Args:
        image_path (str): Path to the input image
        selected_maps (list): List of maps to generate
        settings (dict, optional): Settings for map generation
        
    Returns:
        dict: Paths to the generated maps
    """
    try:
        # Default settings if none provided
        if settings is None:
            settings = {
                'normalIntensity': 1.0,
                'heightIntensity': 1.0,
                'useGrayscale': True,
                'packMaps': False
            }
        
        # Create output directory based on input filename
        input_path = Path(image_path)
        output_dir = input_path.parent / f"{input_path.stem}_maps"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image with PIL to preserve alpha channel
        pil_image = Image.open(image_path)
        has_alpha = pil_image.mode == 'RGBA'
        
        # Convert to numpy array for OpenCV processing
        np_image = np.array(pil_image)
        
        # Process image based on selected maps
        output_paths = {}
        generated_maps = {}
        
        for map_type in selected_maps:
            if map_type == 'normal':
                output_paths[map_type] = generate_normal_map(np_image, output_dir, input_path.stem, has_alpha, settings)
            elif map_type == 'height':
                output_paths[map_type] = generate_height_map(np_image, output_dir, input_path.stem, has_alpha, settings)
            elif map_type == 'bump':
                output_paths[map_type] = generate_bump_map(np_image, output_dir, input_path.stem, has_alpha, settings)
            elif map_type == 'metallic':
                output_paths[map_type] = generate_metallic_map(np_image, output_dir, input_path.stem, has_alpha, settings)
                # Store the actual map data for potential packing
                if settings.get('packMaps', False):
                    if len(np_image.shape) > 2:
                        gray = cv2.cvtColor(np_image[:,:,:3], cv2.COLOR_RGB2GRAY)
                    else:
                        gray = np_image
                    h, w = gray.shape[:2]
                    metallic = np.zeros((h, w), dtype=np.uint8)
                    # Use HSV for metallic detection
                    if len(np_image.shape) > 2 and np_image.shape[2] >= 3:
                        hsv = cv2.cvtColor(np_image[:,:,:3], cv2.COLOR_RGB2HSV)
                        saturation = hsv[:,:,1]
                        value = hsv[:,:,2]
                        potential_metallic = (saturation > 100) & (value > 150)
                        metallic[potential_metallic] = 200
                        metallic = cv2.GaussianBlur(metallic, (5, 5), 0)
                    generated_maps['metallic'] = metallic
            elif map_type == 'roughness':
                output_paths[map_type] = generate_roughness_map(np_image, output_dir, input_path.stem, has_alpha, settings)
                # Store for potential packing
                if settings.get('packMaps', False):
                    if len(np_image.shape) > 2:
                        gray = cv2.cvtColor(np_image[:,:,:3], cv2.COLOR_RGB2GRAY)
                    else:
                        gray = np_image
                    blur = cv2.GaussianBlur(gray, (3, 3), 0)
                    kernel_size = 5
                    h, w = gray.shape[:2]
                    roughness = np.ones((h, w), dtype=np.uint8) * 128
                    for y in range(0, h, kernel_size):
                        for x in range(0, w, kernel_size):
                            block = blur[y:min(y+kernel_size, h), x:min(x+kernel_size, w)]
                            if block.size > 0:
                                local_stddev = np.std(block)
                                local_roughness = min(255, int(local_stddev * 2))
                                roughness[y:min(y+kernel_size, h), x:min(x+kernel_size, w)] = local_roughness
                    edges = cv2.Canny(blur, 50, 150)
                    roughness[edges > 0] = 255
                    roughness = cv2.GaussianBlur(roughness, (5, 5), 0)
                    generated_maps['roughness'] = roughness
            elif map_type == 'specular':
                output_paths[map_type] = generate_specular_map(np_image, output_dir, input_path.stem, has_alpha, settings)
            elif map_type == 'ao':
                output_paths[map_type] = generate_ao_map(np_image, output_dir, input_path.stem, has_alpha, settings)
                # Store for potential packing
                if settings.get('packMaps', False):
                    if len(np_image.shape) > 2:
                        gray = cv2.cvtColor(np_image[:,:,:3], cv2.COLOR_RGB2GRAY)
                    else:
                        gray = np_image
                    blur = cv2.GaussianBlur(gray, (3, 3), 0)
                    height_map = cv2.bitwise_not(blur)
                    edges = cv2.Canny(blur, 50, 150)
                    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
                    h, w = gray.shape[:2]
                    ao = np.ones((h, w), dtype=np.uint8) * 220
                    ao[dilated_edges > 0] = 180
                    ao = cv2.addWeighted(ao, 0.7, height_map, 0.3, 0)
                    ao = cv2.GaussianBlur(ao, (5, 5), 0)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    ao = clahe.apply(ao)
                    generated_maps['ao'] = ao
            elif map_type == 'displacement':
                output_paths[map_type] = generate_displacement_map(np_image, output_dir, input_path.stem, has_alpha, settings)
            elif map_type == 'curvature':
                output_paths[map_type] = generate_curvature_map(np_image, output_dir, input_path.stem, has_alpha, settings)
            elif map_type == 'id':
                output_paths[map_type] = generate_id_map(np_image, output_dir, input_path.stem, has_alpha, settings)
            elif map_type == 'alpha':
                output_paths[map_type] = generate_alpha_mask(np_image, output_dir, input_path.stem, has_alpha, settings)
                if settings.get('packMaps', False):
                    if has_alpha:
                        generated_maps['alpha'] = np_image[:,:,3]
                    else:
                        if len(np_image.shape) > 2:
                            gray = cv2.cvtColor(np_image[:,:,:3], cv2.COLOR_RGB2GRAY)
                        else:
                            gray = np_image
                        _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
                        kernel = np.ones((3, 3), np.uint8)
                        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
                        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
                        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
                        generated_maps['alpha'] = alpha
            elif map_type == 'opacity':
                output_paths[map_type] = generate_opacity_map(np_image, output_dir, input_path.stem, has_alpha, settings)
            elif map_type == 'emissive':
                output_paths[map_type] = generate_emissive_map(np_image, output_dir, input_path.stem, has_alpha, settings)
            elif map_type == 'smoothness':
                output_paths[map_type] = generate_smoothness_map(np_image, output_dir, input_path.stem, has_alpha, settings)
                if settings.get('packMaps', False):
                    if len(np_image.shape) > 2:
                        gray = cv2.cvtColor(np_image[:,:,:3], cv2.COLOR_RGB2GRAY)
                    else:
                        gray = np_image
                    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
                    kernel_size = 5
                    h, w = gray.shape[:2]
                    smoothness = np.ones((h, w), dtype=np.uint8) * 128
                    for y in range(0, h, kernel_size):
                        for x in range(0, w, kernel_size):
                            block = filtered[y:min(y+kernel_size, h), x:min(x+kernel_size, w)]
                            if block.size > 0:
                                local_stddev = np.std(block)
                                local_smoothness = max(0, 255 - min(255, int(local_stddev * 2)))
                                smoothness[y:min(y+kernel_size, h), x:min(x+kernel_size, w)] = local_smoothness
                    smoothness = cv2.GaussianBlur(smoothness, (5, 5), 0)
                    generated_maps['smoothness'] = smoothness
        
        # Generate packed map if requested
        if settings.get('packMaps', False) and len(generated_maps) > 0:
            # Check if we have enough maps to pack
            if ('ao' in generated_maps or 'metallic' in generated_maps or 
                'roughness' in generated_maps or 'smoothness' in generated_maps):
                packed_path = pack_maps_into_channels(generated_maps, output_dir, input_path.stem, has_alpha, settings)
                output_paths['packed'] = packed_path
        
        return {
            'success': True,
            'output_dir': str(output_dir),
            'maps': output_paths
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

# Placeholder functions for map generation
# These will be implemented with actual image processing algorithms later

def generate_normal_map(image, output_dir, filename, has_alpha, settings):
    """Generate a normal map from the input image.
    
    Uses Sobel operators to detect edges and create a normal map.
    """
    output_path = output_dir / f"{filename}_normal.png"
    
    # Convert to grayscale if it's a color image
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Get the intensity factor from settings
    intensity = settings.get('normalIntensity', 1.0)
    
    # Use Sobel operators to get gradients
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    
    # Scale gradients by intensity
    sobelx = sobelx * intensity
    sobely = sobely * intensity
    
    # Create normal map
    normal_map = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    
    # Normalize gradients to -1 to 1 range
    sobelx = cv2.normalize(sobelx, None, -1, 1, cv2.NORM_MINMAX)
    sobely = cv2.normalize(sobely, None, -1, 1, cv2.NORM_MINMAX)
    
    # Set RGB channels (OpenGL format)
    # R = x direction (from -1 to 1, mapped to 0 to 255)
    # G = y direction (from -1 to 1, mapped to 0 to 255)
    # B = z direction (always positive, pointing outward)
    normal_map[:,:,0] = np.uint8((sobelx + 1) * 127.5)  # R: x-direction
    normal_map[:,:,1] = np.uint8((sobely + 1) * 127.5)  # G: y-direction
    normal_map[:,:,2] = np.uint8(255)  # B: z-direction (always pointing outward)
    
    # Add alpha channel if the original image has one
    if has_alpha:
        alpha = image[:,:,3]
        normal_map_with_alpha = np.zeros((gray.shape[0], gray.shape[1], 4), dtype=np.uint8)
        normal_map_with_alpha[:,:,:3] = normal_map
        normal_map_with_alpha[:,:,3] = alpha
        normal_map = normal_map_with_alpha
    
    # Save the image
    Image.fromarray(normal_map).save(output_path)
    return str(output_path)

def generate_height_map(image, output_dir, filename, has_alpha, settings):
    """Generate a height map from the input image.
    
    Converts the image to grayscale and applies intensity adjustment.
    """
    output_path = output_dir / f"{filename}_height.png"
    
    # Convert to grayscale if it's a color image
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Get the intensity factor from settings
    intensity = settings.get('heightIntensity', 1.0)
    
    # Apply intensity adjustment
    adjusted = np.clip(gray.astype(np.float32) * intensity, 0, 255).astype(np.uint8)
    
    # Apply contrast enhancement if needed
    if intensity > 1.0:
        # Normalize histogram to enhance contrast
        adjusted = cv2.equalizeHist(adjusted)
    
    # Add alpha channel if the original image has one
    if has_alpha:
        height_map = np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.uint8)
        height_map[:,:,0] = adjusted
        height_map[:,:,1] = image[:,:,3]
        
        # Save the image with alpha
        img = Image.new('LA', (gray.shape[1], gray.shape[0]))
        img.putdata(list(zip(adjusted.flatten(), image[:,:,3].flatten())))
        img.save(output_path)
    else:
        # Save the image
        Image.fromarray(adjusted).save(output_path)
    
    return str(output_path)

def generate_bump_map(image, output_dir, filename, has_alpha, settings):
    """Generate a bump map from the input image.
    
    Uses edge detection and blurring to create a bump map.
    """
    output_path = output_dir / f"{filename}_bump.png"
    
    # Convert to grayscale if it's a color image
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Laplacian for edge detection
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    
    # Convert back to uint8
    bump = np.uint8(np.absolute(laplacian))
    
    # Invert the image for better bump mapping
    bump = 255 - bump
    
    # Apply contrast enhancement
    bump = cv2.equalizeHist(bump)
    
    # Add alpha channel if the original image has one
    if has_alpha:
        bump_map = np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.uint8)
        bump_map[:,:,0] = bump
        bump_map[:,:,1] = image[:,:,3]
        
        # Save the image with alpha
        img = Image.new('LA', (gray.shape[1], gray.shape[0]))
        img.putdata(list(zip(bump.flatten(), image[:,:,3].flatten())))
        img.save(output_path)
    else:
        # Save the image
        Image.fromarray(bump).save(output_path)
    
    return str(output_path)

def generate_metallic_map(image, output_dir, filename, has_alpha, settings):
    """Generate a metallic map from the input image.
    
    Creates a metallic map based on image brightness and color.
    """
    output_path = output_dir / f"{filename}_metallic.png"
    
    # Create a base metallic map (default to non-metallic)
    h, w = image.shape[:2]
    metallic = np.zeros((h, w), dtype=np.uint8)
    
    # If it's a color image, use color information to estimate metallic areas
    if len(image.shape) > 2 and image.shape[2] >= 3:
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2HSV)
        
        # Metallic surfaces often have high saturation and medium-high value
        # Create a mask for potentially metallic areas
        saturation = hsv[:,:,1]
        value = hsv[:,:,2]
        
        # Areas with high saturation and high value are more likely to be metallic
        potential_metallic = (saturation > 100) & (value > 150)
        
        # Adjust metallic map based on the mask
        metallic[potential_metallic] = 200  # Set to mostly metallic
        
        # Smooth the metallic map
        metallic = cv2.GaussianBlur(metallic, (5, 5), 0)
    
    # Add alpha channel if the original image has one
    if has_alpha:
        metallic_map = np.zeros((h, w, 2), dtype=np.uint8)
        metallic_map[:,:,0] = metallic
        metallic_map[:,:,1] = image[:,:,3]
        
        # Save the image with alpha
        img = Image.new('LA', (w, h))
        img.putdata(list(zip(metallic.flatten(), image[:,:,3].flatten())))
        img.save(output_path)
    else:
        # Save the image
        Image.fromarray(metallic).save(output_path)
    
    return str(output_path)

def generate_roughness_map(image, output_dir, filename, has_alpha, settings):
    """Generate a roughness map from the input image.
    
    Uses image variance and edge detection to estimate surface roughness.
    """
    output_path = output_dir / f"{filename}_roughness.png"
    
    # Convert to grayscale if it's a color image
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Calculate local variance as a measure of roughness
    # Higher variance = rougher surface
    kernel_size = 5
    mean, stddev = cv2.meanStdDev(blur, mask=None, ksize=(kernel_size, kernel_size))
    
    # Use Canny edge detection to find edges
    edges = cv2.Canny(blur, 50, 150)
    
    # Create base roughness map (mid-roughness by default)
    h, w = image.shape[:2]
    roughness = np.ones((h, w), dtype=np.uint8) * 128
    
    # Apply local variance to roughness map
    for y in range(0, h, kernel_size):
        for x in range(0, w, kernel_size):
            # Get the block
            block = blur[y:min(y+kernel_size, h), x:min(x+kernel_size, w)]
            if block.size > 0:
                # Calculate local variance
                local_stddev = np.std(block)
                # Map variance to roughness (0-255)
                local_roughness = min(255, int(local_stddev * 2))
                # Apply to roughness map
                roughness[y:min(y+kernel_size, h), x:min(x+kernel_size, w)] = local_roughness
    
    # Edges are typically rougher
    roughness[edges > 0] = 255
    
    # Smooth the roughness map
    roughness = cv2.GaussianBlur(roughness, (5, 5), 0)
    
    # Add alpha channel if the original image has one
    if has_alpha:
        roughness_map = np.zeros((h, w, 2), dtype=np.uint8)
        roughness_map[:,:,0] = roughness
        roughness_map[:,:,1] = image[:,:,3]
        
        # Save the image with alpha
        img = Image.new('LA', (w, h))
        img.putdata(list(zip(roughness.flatten(), image[:,:,3].flatten())))
        img.save(output_path)
    else:
        # Save the image
        Image.fromarray(roughness).save(output_path)
    
    return str(output_path)

def generate_specular_map(image, output_dir, filename, has_alpha, settings):
    """Generate a specular map from the input image.
    
    Creates a specular map based on image brightness and inverse of roughness.
    """
    output_path = output_dir / f"{filename}_specular.png"
    
    # Convert to grayscale if it's a color image
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Enhance contrast to better identify specular areas
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    
    # Bright areas are more likely to be specular
    # Create a mask for potentially specular areas
    _, bright_mask = cv2.threshold(enhanced, 200, 255, cv2.THRESH_BINARY)
    
    # Create base specular map (low specularity by default)
    h, w = image.shape[:2]
    specular = np.ones((h, w), dtype=np.uint8) * 50
    
    # Apply brightness to specular map
    specular = cv2.addWeighted(specular, 0.5, enhanced, 0.5, 0)
    
    # Bright areas get higher specularity
    specular[bright_mask > 0] = 255
    
    # Smooth the specular map
    specular = cv2.GaussianBlur(specular, (5, 5), 0)
    
    # Add alpha channel if the original image has one
    if has_alpha:
        specular_map = np.zeros((h, w, 2), dtype=np.uint8)
        specular_map[:,:,0] = specular
        specular_map[:,:,1] = image[:,:,3]
        
        # Save the image with alpha
        img = Image.new('LA', (w, h))
        img.putdata(list(zip(specular.flatten(), image[:,:,3].flatten())))
        img.save(output_path)
    else:
        # Save the image
        Image.fromarray(specular).save(output_path)
    
    return str(output_path)

def generate_ao_map(image, output_dir, filename, has_alpha, settings):
    """Generate an ambient occlusion map from the input image.
    
    Uses edge detection and depth estimation to create an ambient occlusion map.
    """
    output_path = output_dir / f"{filename}_ao.png"
    
    # Convert to grayscale if it's a color image
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Use height map as a base for AO calculation
    # Darker areas in height map are typically more occluded
    height_map = cv2.bitwise_not(blur)  # Invert for height
    
    # Detect edges for occlusion boundaries
    edges = cv2.Canny(blur, 50, 150)
    dilated_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    
    # Create base AO map (mostly unoccluded)
    h, w = image.shape[:2]
    ao = np.ones((h, w), dtype=np.uint8) * 220
    
    # Areas near edges are more likely to be occluded
    ao[dilated_edges > 0] = 180
    
    # Apply height information to AO
    # Lower areas (darker in height map) are more occluded
    ao = cv2.addWeighted(ao, 0.7, height_map, 0.3, 0)
    
    # Smooth the AO map
    ao = cv2.GaussianBlur(ao, (5, 5), 0)
    
    # Apply contrast enhancement for better visual results
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ao = clahe.apply(ao)
    
    # Add alpha channel if the original image has one
    if has_alpha:
        ao_map = np.zeros((h, w, 2), dtype=np.uint8)
        ao_map[:,:,0] = ao
        ao_map[:,:,1] = image[:,:,3]
        
        # Save the image with alpha
        img = Image.new('LA', (w, h))
        img.putdata(list(zip(ao.flatten(), image[:,:,3].flatten())))
        img.save(output_path)
    else:
        # Save the image
        Image.fromarray(ao).save(output_path)
    
    return str(output_path)

def generate_displacement_map(image, output_dir, filename, has_alpha, settings):
    """Generate a displacement map from the input image.
    
    Creates a high-detail displacement map using edge detection and filtering.
    """
    output_path = output_dir / f"{filename}_displacement.png"
    
    # Convert to grayscale if it's a color image
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply bilateral filter to preserve edges while smoothing
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Get the intensity factor from settings
    intensity = settings.get('heightIntensity', 1.0)
    
    # Apply intensity adjustment
    adjusted = np.clip(filtered.astype(np.float32) * intensity, 0, 255).astype(np.uint8)
    
    # Enhance contrast for better displacement details
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(adjusted)
    
    # Apply edge-preserving filter for smoother gradients
    displacement = cv2.edgePreservingFilter(enhanced, flags=1, sigma_s=60, sigma_r=0.4)
    
    # Add alpha channel if the original image has one
    if has_alpha:
        displacement_map = np.zeros((gray.shape[0], gray.shape[1], 2), dtype=np.uint8)
        displacement_map[:,:,0] = displacement
        displacement_map[:,:,1] = image[:,:,3]
        
        # Save the image with alpha
        img = Image.new('LA', (gray.shape[1], gray.shape[0]))
        img.putdata(list(zip(displacement.flatten(), image[:,:,3].flatten())))
        img.save(output_path)
    else:
        # Save the image
        Image.fromarray(displacement).save(output_path)
    
    return str(output_path)

def generate_curvature_map(image, output_dir, filename, has_alpha, settings):
    """Generate a curvature map from the input image.
    
    Uses second derivatives to estimate surface curvature.
    """
    output_path = output_dir / f"{filename}_curvature.png"
    
    # Convert to grayscale if it's a color image
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Calculate second derivatives using Laplacian
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    
    # Normalize to 0-255 range
    abs_laplacian = np.absolute(laplacian)
    normalized = cv2.normalize(abs_laplacian, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Enhance contrast for better curvature details
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)
    
    # Create curvature map
    h, w = image.shape[:2]
    curvature = np.ones((h, w), dtype=np.uint8) * 128  # Mid-gray base
    
    # Apply Laplacian to curvature map
    # Higher values (white) represent convex areas
    # Lower values (black) represent concave areas
    curvature = cv2.addWeighted(curvature, 0.5, enhanced, 0.5, 0)
    
    # Smooth the curvature map
    curvature = cv2.GaussianBlur(curvature, (5, 5), 0)
    
    # Add alpha channel if the original image has one
    if has_alpha:
        curvature_map = np.zeros((h, w, 2), dtype=np.uint8)
        curvature_map[:,:,0] = curvature
        curvature_map[:,:,1] = image[:,:,3]
        
        # Save the image with alpha
        img = Image.new('LA', (w, h))
        img.putdata(list(zip(curvature.flatten(), image[:,:,3].flatten())))
        img.save(output_path)
    else:
        # Save the image
        Image.fromarray(curvature).save(output_path)
    
    return str(output_path)

def generate_id_map(image, output_dir, filename, has_alpha, settings):
    """Generate an ID map from the input image.
    
    Uses segmentation techniques to create a colored ID map by region.
    """
    output_path = output_dir / f"{filename}_id.png"
    
    # Convert to RGB if it's grayscale
    if len(image.shape) == 2:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) > 2 and image.shape[2] >= 3:
        rgb_image = image[:,:,:3].copy()
    else:
        # Fallback
        h, w = image.shape[:2]
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Apply bilateral filter to smooth while preserving edges
    filtered = cv2.bilateralFilter(rgb_image, 9, 75, 75)
    
    # Convert to LAB color space for better segmentation
    lab = cv2.cvtColor(filtered, cv2.COLOR_RGB2LAB)
    
    # Apply K-means clustering for segmentation
    # Reshape the image to a 2D array of pixels
    pixel_values = lab.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 8  # Number of clusters/colors
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8 and reshape back to original image dimensions
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(rgb_image.shape)
    
    # Convert back to RGB
    id_map = cv2.cvtColor(segmented_image, cv2.COLOR_LAB2RGB)
    
    # Assign distinct colors to each segment for better visualization
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        # Generate a distinct color for each label
        color = np.array([
            (i * 37) % 255,  # R
            (i * 79) % 255,  # G
            (i * 151) % 255  # B
        ], dtype=np.uint8)
        
        # Create mask for this label
        mask = labels.flatten() == label
        mask = mask.reshape(rgb_image.shape[:2])
        
        # Apply color to the mask area
        for c in range(3):
            id_map[:,:,c][mask] = color[c]
    
    # Add alpha channel if the original image has one
    if has_alpha:
        id_map_with_alpha = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 4), dtype=np.uint8)
        id_map_with_alpha[:,:,:3] = id_map
        id_map_with_alpha[:,:,3] = image[:,:,3]
        id_map = id_map_with_alpha
    
    # Save the image
    Image.fromarray(id_map).save(output_path)
    return str(output_path)

def generate_alpha_mask(image, output_dir, filename, has_alpha, settings):
    """Generate an alpha mask from the input image.
    
    Extracts or creates an alpha mask for transparency.
    """
    output_path = output_dir / f"{filename}_alpha.png"
    
    h, w = image.shape[:2]
    
    if has_alpha:
        # Extract existing alpha channel
        alpha = image[:,:,3]
    else:
        # Create an alpha mask based on image brightness
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Threshold to create a binary mask
        _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
        
        # Smooth the edges
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    
    # Save the image
    Image.fromarray(alpha).save(output_path)
    return str(output_path)

def generate_opacity_map(image, output_dir, filename, has_alpha, settings):
    """Generate an opacity map from the input image.
    
    Creates an opacity map based on image brightness and existing alpha.
    """
    output_path = output_dir / f"{filename}_opacity.png"
    
    h, w = image.shape[:2]
    
    if has_alpha:
        # Use existing alpha channel as a base
        opacity = image[:,:,3].copy()
    else:
        # Create an opacity map based on image brightness
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Invert brightness for opacity (darker areas more opaque)
        opacity = cv2.bitwise_not(gray)
        
        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        opacity = clahe.apply(opacity)
        
        # Apply threshold to create more defined opacity areas
        _, opacity = cv2.threshold(opacity, 100, 255, cv2.THRESH_BINARY)
        
        # Smooth the edges
        opacity = cv2.GaussianBlur(opacity, (5, 5), 0)
    
    # Save the image
    Image.fromarray(opacity).save(output_path)
    return str(output_path)

def generate_emissive_map(image, output_dir, filename, has_alpha, settings):
    """Generate an emissive map from the input image.
    
    Creates an emissive map highlighting bright and saturated areas.
    """
    output_path = output_dir / f"{filename}_emissive.png"
    
    # Create a base black emissive map (non-emissive by default)
    h, w = image.shape[:2]
    emissive = np.zeros((h, w, 3), dtype=np.uint8)
    
    # If it's a color image, use color information to estimate emissive areas
    if len(image.shape) > 2 and image.shape[2] >= 3:
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2HSV)
        
        # Bright and saturated areas are more likely to be emissive
        saturation = hsv[:,:,1]
        value = hsv[:,:,2]
        
        # Create a mask for potentially emissive areas
        # High value (brightness) and high saturation suggest emissive properties
        potential_emissive = (saturation > 150) & (value > 200)
        
        # Extract colors from original image for emissive areas
        for c in range(3):
            emissive[:,:,c][potential_emissive] = image[:,:,c][potential_emissive]
        
        # Enhance brightness for emissive areas
        emissive_hsv = cv2.cvtColor(emissive, cv2.COLOR_RGB2HSV)
        emissive_hsv[:,:,2] = np.minimum(emissive_hsv[:,:,2] * 1.5, 255).astype(np.uint8)
        emissive = cv2.cvtColor(emissive_hsv, cv2.COLOR_HSV2RGB)
        
        # Apply Gaussian blur for glow effect
        emissive = cv2.GaussianBlur(emissive, (5, 5), 0)
    
    # Add alpha channel if the original image has one
    if has_alpha:
        emissive_with_alpha = np.zeros((h, w, 4), dtype=np.uint8)
        emissive_with_alpha[:,:,:3] = emissive
        emissive_with_alpha[:,:,3] = image[:,:,3]
        emissive = emissive_with_alpha
    
    # Save the image
    Image.fromarray(emissive).save(output_path)
    return str(output_path)

def generate_smoothness_map(image, output_dir, filename, has_alpha, settings):
    """Generate a smoothness map from the input image.
    
    Creates a smoothness map as the inverse of roughness.
    """
    output_path = output_dir / f"{filename}_smoothness.png"
    
    # Convert to grayscale if it's a color image
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image[:,:,:3], cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply bilateral filter to preserve edges while smoothing
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Calculate local variance as a measure of roughness
    # Lower variance = smoother surface
    kernel_size = 5
    mean, stddev = cv2.meanStdDev(filtered, mask=None, ksize=(kernel_size, kernel_size))
    
    # Create base smoothness map (mid-smoothness by default)
    h, w = image.shape[:2]
    smoothness = np.ones((h, w), dtype=np.uint8) * 128
    
    # Apply local variance to smoothness map
    for y in range(0, h, kernel_size):
        for x in range(0, w, kernel_size):
            # Get the block
            block = filtered[y:min(y+kernel_size, h), x:min(x+kernel_size, w)]
            if block.size > 0:
                # Calculate local variance
                local_stddev = np.std(block)
                # Map variance to smoothness (0-255)
                # Invert the roughness calculation: lower variance = higher smoothness
                local_smoothness = max(0, 255 - min(255, int(local_stddev * 2)))
                # Apply to smoothness map
                smoothness[y:min(y+kernel_size, h), x:min(x+kernel_size, w)] = local_smoothness
    
    # Smooth the smoothness map
    smoothness = cv2.GaussianBlur(smoothness, (5, 5), 0)
    
    # Add alpha channel if the original image has one
    if has_alpha:
        smoothness_map = np.zeros((h, w, 2), dtype=np.uint8)
        smoothness_map[:,:,0] = smoothness
        smoothness_map[:,:,1] = image[:,:,3]
        
        # Save the image with alpha
        img = Image.new('LA', (w, h))
        img.putdata(list(zip(smoothness.flatten(), image[:,:,3].flatten())))
        img.save(output_path)
    else:
        # Save the image
        Image.fromarray(smoothness).save(output_path)
    
    return str(output_path)

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print(json.dumps({
            'success': False,
            'error': 'Missing required arguments: image_path, selected_maps, settings'
        }))
        sys.exit(1)
    
    image_path = sys.argv[1]
    selected_maps = json.loads(sys.argv[2]) if len(sys.argv) > 2 else []
    settings = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
    
    # Process the image
    result = process_image(image_path, selected_maps, settings)
    
    # Return the result as JSON
    print(json.dumps(result))
