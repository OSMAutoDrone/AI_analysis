import segmentation_models_pytorch as smp
import torch
import numpy as np
from auto_masking import masking_pipeline, osm_masking
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from preprocessing import TARGET_SIZE
import os
from tqdm import tqdm
from training import utils, predictions
from training.data_setup import MaskColorMap
from PIL import Image
import cv2
import mercantile


def compute_IoU(img_path, target_size=TARGET_SIZE):
    osm_mask, data, _ = masking_pipeline(img_path, target_size)  # data: (H, W, C)
    model = load_model()

    img_tensor = utils.np_to_tensor(data.transpose(2, 0, 1))  # (1, C, H, W)
    res = predictions.predict_mask(img_tensor, model, "cpu")  # (1, C, H, W)
    predicted_mask = torch.argmax(res[0], dim=0)

    color_map = {color.value: idx for idx, color in enumerate(MaskColorMap)}
    osm_mask = utils.rgb_to_class(osm_mask, color_map)

    iou = utils.compute_iou(predicted_mask, osm_mask, "cpu", num_classes=4)
    return iou, predicted_mask, osm_mask


def load_model(model_path = 'training/UNet_Model_Output/model_full_v02.pth'):
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

def Iou_test():
    base_path = "data/source_OSM"
    ious_good = []
    ious_bad = []
    img_good = []
    img_bad = []
    path_good = []
    path_bad = []
    threshold = 0.49
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for r, d, f in os.walk(base_path):
        f = np.array(f)
        idx = np.random.choice(f.shape[0], 200, replace=False)
        files = f[idx]
        for file in tqdm(f):
            qual = "good"
            iou, predicted_mask, osm_mask = compute_IoU(os.path.join(r, file))
            if os.path.exists(os.path.join(base_path.replace("source_OSM", "base/osm/img"), file.replace(".tif", ".jpg"))):
                ious_good.append(iou)
                img_good.append([predicted_mask, osm_mask])
                path_good.append(os.path.join(r, file))
                #if iou <= threshold:
                #    utils.display(name=f"good: {iou}",image=np.array(Image.open(os.path.join(r, file))) ,mask=osm_mask, predicted_mask=predicted_mask)
                #    fn =fn+1
                #else:
                #    tp = tp+1
            else:
                ious_bad.append(iou)
                img_bad.append([predicted_mask, osm_mask])
                path_bad.append(os.path.join(r, file))
                qual = "bad"
                #if iou > threshold:
                #    utils.display(name=f"bad: {iou}",image=np.array(Image.open(os.path.join(r, file))) ,mask=osm_mask, predicted_mask=predicted_mask)
                #    fp = fp+1
                #else:
                #    tn = tn+1
            print(f"{qual} iou: {iou:.4f}")
    ious_good = np.array(ious_good)
    ious_bad = np.array(ious_bad)

    avg_good = np.mean(ious_good)
    avg_bad = np.mean(ious_bad)
    print(f"Average good Ious: {avg_good}, covariance: {np.cov(ious_good)}")
    print(f"Average bad Ious: {avg_bad}, covariance: {np.cov(ious_bad)}")
    threshold = (avg_good+avg_bad)/2
    print(f"Threshold: {threshold}")
    for iou in ious_good:
        if iou <= threshold:
            fn =fn+1
        else:
            tp = tp+1
    for iou in ious_bad:
        if iou <= threshold:
            tn = tn+1
        else:
            fp = fp+1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"fp: {fp}, fn: {fn}, tp: {tp}, tn: {tn}")
    print(f"precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")

            # utils.display(mask=osm_mask, predicted_mask=predicted_mask)

    bests_idx = np.argpartition(ious_good, -5)[-5:]
    worst_idx = np.argpartition(ious_bad, 5)[:5]
    for idx in bests_idx:
        utils.display(name=f"bests: {ious_good[idx]:.4f}",image=np.array(Image.open(path_good[idx])), mask=img_good[idx][1], predicted_mask=img_good[idx][0])
    for idx in worst_idx:
        utils.display(name=f"worsts: {ious_bad[idx]:.4f}",image=np.array(Image.open(path_bad[idx])), mask=img_bad[idx][1], predicted_mask=img_bad[idx][0])

if __name__ == '__main__':
    model = load_model()
    z, x, y = 18, 78662, 93813
    img_path = f"data/flight_test_data/{z}/{x}/{y}.png"

    img = np.array(Image.open(img_path).convert("RGB"))
    height, width = img.shape[:2]

    # XYZ tile -> bounds in EPSG:3857 (web mercator, same CRS as the training tifs)
    tile_bounds = mercantile.xy_bounds(x, y, z)
    bounds = (tile_bounds.left, tile_bounds.bottom, tile_bounds.right, tile_bounds.top)
    transform = from_bounds(*bounds, width, height)
    print(f"Tile {z}/{x}/{y} bounds (EPSG:3857): {bounds}")

    osm_mask = osm_masking(bounds, CRS.from_epsg(3857), transform, height, width)

    img_tensor = utils.np_to_tensor(img.transpose(2, 0, 1))
    res = predictions.predict_mask(img_tensor, model, "cpu")
    predicted_mask = torch.argmax(res[0], dim=0)

    color_map = {color.value: idx for idx, color in enumerate(MaskColorMap)}
    iou = utils.compute_iou(predicted_mask, utils.rgb_to_class(osm_mask, color_map), "cpu", num_classes=4)
    utils.display(name=f"{z}/{x}/{y} IoU: {iou:.4f}", image=img, osm_mask=osm_mask, predicted_mask=predicted_mask)

