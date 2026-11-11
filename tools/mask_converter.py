import os
import numpy as np
from PIL import Image

def class_to_rgb(mask_class, color_map):
    """
    color_map: dict {(R,G,B): [class_ids]}  ← format direct, pas d'inversion
    """
    mask_rgb = np.zeros((*mask_class.shape, 3), dtype=np.uint8)
    for color, class_ids in color_map.items():
        match = np.isin(mask_class, class_ids)
        mask_rgb[match] = np.array(color)
    return mask_rgb

if __name__ == '__main__':
    for d, r, f in os.walk('./data/base/mask_modif'):
        for file in f:
            img = Image.open(os.path.join(d, file))
            mask_array = np.array(img)
            inv_color_map = {
                (0, 0, 0): [1,5,6,7],  # Unlabelled
                (255, 255, 0): [2],  # Building
                (0, 64, 255): [4],  #water
                (240, 30, 44): [3],  # Road
            }
            converted_mask = class_to_rgb(mask_array, inv_color_map)
            Image.fromarray(converted_mask).save(os.path.join(d.replace("mask_modif", "mask_rgb"), file))
