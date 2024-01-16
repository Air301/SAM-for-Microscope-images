import argparse
from tqdm import tqdm
import torch
import random
import matplotlib.pyplot as plt
from pathlib import Path
import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
import os
import glob
import sys
import numpy as np
from PIL import Image
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

sam_checkpoint = "../weights/edge_sam.pth"
model_type = "edge_sam"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)
predictor = SamPredictor(sam)

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes


def predict_and_save(img, crop_size, output_path, img_id, image_format):
    shape = img.shape[:2]
    xs = shape[0] // crop_size
    ys = shape[1] // crop_size
    if os.path.exists(output_path):
        pass
    else:
        os.makedirs(output_path)
    resized_img = cv2.resize(img, (shape[0] // xs, shape[1] // ys))
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=10,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )
    masks2 = mask_generator_2.generate(resized_img)

    if len(masks2) == 0:
        return

    sorted_anns = sorted(masks2, key=(lambda x: x['area']), reverse=True)
    # Use the original image shape for creating the mask
    img = np.empty((shape[0], shape[1], 4), dtype=np.uint8)
    img[:, :, 3] = 0

    for i, ann in enumerate(tqdm(sorted_anns)):
        # Upsample the segmentation mask to the original image shape
        m_upsampled = cv2.resize(ann['segmentation'].astype(np.uint8), (shape[1], shape[0]))

        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m_upsampled > 0] = color_mask

        # Save the batch
        if output_path:
            img_for_jpg = (img[:, :, 0:3] * 255).astype(np.uint8)
            save_jpg_path = os.path.join(output_path, f'{img_id}.jpg')
            plt.imsave(save_jpg_path.format(i+1), img_for_jpg)
        img = np.empty((shape[0], shape[1], 4), dtype=np.uint8)
        img[:, :, 3] = 0

    # Save the last batch if needed
    if save_jpg_path:
        img_for_jpg = (img[:, :, 0:3] * 255).astype(np.uint8)
        plt.imsave(save_jpg_path.format(i+1), img_for_jpg)

def run(source='', output='', imgsz=640):
    p = str(Path(source).absolute())  # os-agnostic absolute path
    if '*' in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        files = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')

    image_format = files[0].split('.')[-1].lower()
    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]

    for path in images:
        img_id = os.path.splitext(os.path.split(path)[1])[0]
        img = cv2.imread(path)
        # output_path = os.path.join(output)
        torch.cuda.empty_cache()  # 清空显存占用
        predict_and_save(img=img, crop_size=imgsz, output_path=output, img_id=img_id, image_format=image_format)
    return 0


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/home/pumbaa/Documents/OriginDataset/Moon/',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--output', type=str, default='/home/pumbaa/Documents/OriginDataset/Moon/output/',
                        help='path to cropped images')
    parser.add_argument('--imgsz', type=int, default='1024', help='cropped images size')
    opt = parser.parse_args()
    return opt


def main(opt):
    print('predict: ' + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
