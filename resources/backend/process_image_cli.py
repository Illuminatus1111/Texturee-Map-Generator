#!/usr/bin/env python3
import sys
import os
import json
import numpy as np
import cv2
from PIL import Image
import argparse
from pathlib import Path
from channel_packing import pack_maps_into_channels

def main():
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

if __name__ == "__main__":
    main()
