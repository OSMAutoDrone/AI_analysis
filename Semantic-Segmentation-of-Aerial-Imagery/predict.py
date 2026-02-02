import rasterio
import numpy as np
import torch
from predictions import *
import segmentation_models_pytorch as smp
import utils
from PIL import Image

def load_geotiff(path):
    with rasterio.open(path) as src:
        img = src.read()
        transform = src.transform
        crs = src.crs
    return img, transform, crs

def preprocess(img_np):
    img_np = img_np.astype(np.float32)
    img_np /= 255.0

    img_tensor = torch.from_numpy(img_np).unsqueeze(0)

    return img_tensor


if __name__ == '__main__':
    img_np, transform, crs = load_geotiff("../data/testGeoImage02.tif")
    img_tensor = preprocess(img_np)
    img_vis = img_np.transpose(1, 2, 0)

    ENCODER = 'efficientnet-b4'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax2d'

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=6,
        activation=ACTIVATION,
    )
    model.load_state_dict(torch.load('./UNet_Model_Output/model_epoch_30', map_location=torch.device('cpu')))

    img = Image.open("../data/Patches/imgs/image0100.jpg")
    mask = Image.open("../data/Patches/masks/image0100.png")

    img_arr = np.array(img)
    mask_arr = np.array(mask)
    img_model = img_arr.transpose(2, 0, 1)
    img_tensor = preprocess(img_model)
    res = predict_mask(img_tensor, model, "cpu")
    predicted_mask = np.transpose(np.argmax(res[0].to('cpu'), axis=0, keepdims=True))
    utils.display(original_image=img_arr,mask=mask_arr, predicted_mask=predicted_mask)

