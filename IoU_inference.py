import segmentation_models_pytorch as smp
import utils
import torch
import numpy as np
import rasterio
import predictions
from auto_masking import masking_pipeline
from data_setup import MaskColorMap
import os
from tqdm import tqdm


def compute_IoU(img_path):
    osm_mask, _, _ = masking_pipeline(img_path)
    model = load_model()
    with rasterio.open(img_path) as src:
        data = src.read()  # (C, H, W)
        data = np.moveaxis(data, 0, -1)  # (H, W, C)

    img_tensor = utils.np_to_tensor(data.transpose(2, 0, 1))  # (1, C, H, W)
    res = predictions.predict_mask(img_tensor, model, "cpu")  # (1, C, H, W)
    predicted_mask = torch.argmax(res[0], dim=0)

    color_map = {color.value: idx for idx, color in enumerate(MaskColorMap)}
    osm_mask = utils.rgb_to_class(osm_mask, color_map)

    iou = utils.compute_iou(predicted_mask, osm_mask, "cpu", num_classes=4)
    return iou, predicted_mask, osm_mask


def load_model(model_path = './UNet_Model_Output/test_model02.pth'):
    ENCODER = 'efficientnet-b4'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'softmax2d'

    # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=4,
        activation=ACTIVATION,
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

if __name__ == '__main__':
    base_path = "data/source_OSM"
    ious_good = []
    ious_bad = []
    img_good = []
    img_bad = []
    for r, d, f in os.walk(base_path):
        for file in tqdm(f):
            iou, predicted_mask, osm_mask = compute_IoU(os.path.join(r, file))
            print(f"iou: {iou:.4f}")
            if os.path.exists(os.path.join(base_path.replace("source_OSM", "base/osm/img"), file.replace(".tif", ".jpg"))):
                ious_good.append(iou)
                img_good.append([predicted_mask, osm_mask])
            else:
                ious_bad.append(iou)
                img_bad.append([predicted_mask, osm_mask])

            # utils.display(mask=osm_mask, predicted_mask=predicted_mask)
    ious_good = np.array(ious_good)
    # img_good = np.array(img_good)
    print(f"Average good Ious: {np.mean(ious_good)}, covariance: {np.cov(ious_good)}")
    ious_bad = np.array(ious_bad)
    # img_bad = np.array(img_bad)
    print(f"Average bad Ious: {np.mean(ious_bad)}, covariance: {np.cov(ious_bad)}")
    bests_idx = np.argpartition(ious_good, -5)[-5:]
    worst_idx = np.argpartition(ious_bad, 5)[:5]
    for idx in bests_idx:
        utils.display(mask=img_good[idx][1], predicted_mask=img_good[idx][0], name=f"bests: {ious_good[idx]:.4f}")
    for idx in worst_idx:
        utils.display(mask=img_bad[idx][1], predicted_mask=img_bad[idx][0], name=f"worsts: {ious_bad[idx]:.4f}")

