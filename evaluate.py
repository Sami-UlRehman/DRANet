"""Evaluate the trained model on a paired low/high test set (e.g. LOL eval15).

Reports PSNR, SSIM (tf.image) and FSIM (image-similarity-measures).

    python evaluate.py --model ./saved_model --data_root ./eval15
"""
import argparse
import glob
import os
import cv2
import numpy as np
import tensorflow as tf
from image_similarity_measures import quality_metrics as qm

from predict import load_trained_model, IMG_SIZE


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data_root', required=True, help='folder with low/ and high/ subfolders')
    args = p.parse_args()

    model = load_trained_model(args.model)

    low_files = sorted(glob.glob(os.path.join(args.data_root, 'low', '*.png')))
    high_files = sorted(glob.glob(os.path.join(args.data_root, 'high', '*.png')))
    assert len(low_files) == len(high_files), 'low/high file counts differ'

    psnr_vals, ssim_vals, fsim_vals = [], [], []
    for lp, hp in zip(low_files, high_files):
        low = cv2.imread(lp)
        gt = cv2.imread(hp)

        x = cv2.resize(low, IMG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        y = np.clip(model.predict(x[None, ...], verbose=0)[0], 0, 1)

        gt_r = cv2.resize(gt, IMG_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

        psnr_vals.append(float(tf.image.psnr(gt_r, y, max_val=1.0)))
        ssim_vals.append(float(tf.image.ssim(gt_r, y, max_val=1.0)))
        fsim_vals.append(qm.fsim((gt_r * 255).astype(np.uint8), (y * 255).astype(np.uint8)))

    print(f'Images evaluated : {len(low_files)}')
    print(f'PSNR             : {np.mean(psnr_vals):.4f}')
    print(f'SSIM             : {np.mean(ssim_vals):.6f}')
    print(f'FSIM             : {np.mean(fsim_vals):.4f}')


if __name__ == '__main__':
    main()
