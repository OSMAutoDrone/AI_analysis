import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
from PIL import Image

# ------------------------------------------------------------------ #
#  CONFIGURATION                                                       #
# ------------------------------------------------------------------ #
base_path        = r"./data/base"
img_input_path   = os.path.join(base_path, "img")
mask_input_path  = os.path.join(base_path, "mask")
img_output_path  = os.path.join(base_path, "../processed/img")
mask_output_path = os.path.join(base_path, "../processed/mask")

TARGET_RATIO     = (4, 3)       # largeur : hauteur
TARGET_SIZE      = (640, 480)   # (width, height) après resize
N_AUGMENTATIONS  = 0          # nb de variantes augmentées par image

# ------------------------------------------------------------------ #
#  PIPELINE ALBUMENTATIONS                                             #
# ------------------------------------------------------------------ #
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.4),
    A.VerticalFlip(p=0.4),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=15,
        border_mode=cv2.BORDER_REFLECT,
        p=0.5
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    A.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.3
    ),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2)
])

# ------------------------------------------------------------------ #
#  FONCTIONS                                                           #
# ------------------------------------------------------------------ #
def crop_to_ratio(image: np.ndarray, mask: np.ndarray, ratio: tuple) -> tuple:
    """Centre-crop image et masque vers le ratio cible."""
    h, w = image.shape[:2]
    target_w, target_h = ratio

    if w / h > target_w / target_h:
        # Image trop large → rogner les côtés
        new_w = int(h * target_w / target_h)
        x_start = (w - new_w) // 2
        image = image[:, x_start:x_start + new_w]
        mask  = mask[:, x_start:x_start + new_w]
    else:
        # Image trop haute → rogner le haut/bas
        new_h = int(w * target_h / target_w)
        y_start = (h - new_h) // 2
        image = image[y_start:y_start + new_h, :]
        mask  = mask[y_start:y_start + new_h, :]

    return image, mask


def resize_pair(image: np.ndarray, mask: np.ndarray, size: tuple) -> tuple:
    """Resize (width, height). Masque en NEAREST pour ne pas interpoler les classes."""
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    mask  = cv2.resize(mask,  size, interpolation=cv2.INTER_NEAREST)
    return image, mask


def save_pair(image: np.ndarray, mask: np.ndarray, img_out: str, mask_out: str):
    Image.fromarray(image).save(img_out,  quality=95, subsampling=0)
    Image.fromarray(mask).save(mask_out)


def process_pair(img_file: str):
    img_path  = os.path.join(img_input_path,  img_file)
    mask_file = img_file.replace(".jpg", ".png")
    msk_path  = os.path.join(mask_input_path, mask_file)

    if not os.path.exists(msk_path):
        print(f"  ⚠ Masque manquant pour {img_file}, ignoré.")
        return

    image = np.array(Image.open(img_path).convert("RGB"))
    mask  = np.array(Image.open(msk_path).convert("RGB"))

    # 1. Recadrage 4:3
    image, mask = crop_to_ratio(image, mask, TARGET_RATIO)

    # 2. Resize
    image, mask = resize_pair(image, mask, TARGET_SIZE)

    stem = img_file.replace(".jpg", "")

    # 3. Sauvegarde de la version originale (sans augmentation)
    save_pair(
        image, mask,
        os.path.join(img_output_path,  f"{stem}_orig.jpg"),
        os.path.join(mask_output_path, f"{stem}_orig.png"),
    )

    # 4. Augmentations
    for i in range(N_AUGMENTATIONS):
        augmented = augmentation_pipeline(image=image, mask=mask)
        save_pair(
            augmented["image"], augmented["mask"],
            os.path.join(img_output_path,  f"{stem}_aug{i:02d}.jpg"),
            os.path.join(mask_output_path, f"{stem}_aug{i:02d}.png"),
        )


# ------------------------------------------------------------------ #
#  MAIN                                                                #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    os.makedirs(img_output_path,  exist_ok=True)
    os.makedirs(mask_output_path, exist_ok=True)

    files = [f for f in os.listdir(img_input_path) if (f.endswith(".jpg") or f.endswith(".png"))]
    print(f"{len(files)} images trouvées, {N_AUGMENTATIONS} augmentations chacune "
          f"→ {len(files) * (N_AUGMENTATIONS + 1)} paires en sortie.")

    for img_file in tqdm(files):
        process_pair(img_file)

    print("Preprocessing terminé.")