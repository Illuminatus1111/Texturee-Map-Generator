def pack_maps_into_channels(maps, output_dir, filename, has_alpha, settings):
    """Pack multiple maps into RGB channels of a single image.
    
    Typically packs AO in red, metallic in green, and smoothness in blue.
    """
    output_path = output_dir / f"{filename}_packed.png"
    
    # Get dimensions from the first map
    first_map = next(iter(maps.values()))
    h, w = first_map.shape[:2]
    
    # Create a base RGB image
    packed = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Default packing: AO in R, Metallic in G, Smoothness in B
    if 'ao' in maps:
        packed[:,:,0] = maps['ao']  # Red channel
    
    if 'metallic' in maps:
        packed[:,:,1] = maps['metallic']  # Green channel
    
    if 'smoothness' in maps:
        packed[:,:,2] = maps['smoothness']  # Blue channel
    elif 'roughness' in maps:
        # Invert roughness to get smoothness
        packed[:,:,2] = 255 - maps['roughness']  # Blue channel
    
    # Add alpha channel if needed
    if has_alpha:
        alpha = None
        if 'alpha' in maps:
            alpha = maps['alpha']
        elif 'opacity' in maps:
            alpha = maps['opacity']
        
        if alpha is not None:
            packed_with_alpha = np.zeros((h, w, 4), dtype=np.uint8)
            packed_with_alpha[:,:,:3] = packed
            packed_with_alpha[:,:,3] = alpha
            packed = packed_with_alpha
    
    # Save the image
    Image.fromarray(packed).save(output_path)
    return str(output_path)
