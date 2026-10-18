"""Enhance low-light image(s) with the trained SavedModel.

    python predict.py --model ./saved_model --input ./my_lowlight.png --output ./result.png
    python predict.py --model ./saved_model --input ./folder_of_images --output ./out_folder
"""
import argparse
import os
import glob
import cv2
import numpy as np
import tensorflow as tf

from model.attention import CBAM_Block
from model.blocks import DR_Block
from losses.losses import SSIMLoss

CUSTOM_OBJECTS = {'CBAM_Block': CBAM_Block, 'DR_Block': DR_Block, 'SSIMLoss': SSIMLoss}
IMG_SIZE = (256, 256)


def load_trained_model(path):
    return tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)


def enhance(model, image_bgr):
    resized = cv2.resize(image_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    y = model.predict(x[None, ...], verbose=0)[0]
    y = np.clip(y * 255.0, 0, 255).astype(np.uint8)
    return cv2.resize(y, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='path to SavedModel folder (.pb + variables)')
    p.add_argument('--input', required=True, help='image file or folder of images')
    p.add_argument('--output', required=True, help='output file or folder')
    args = p.parse_args()

    model = load_trained_model(args.model)

    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        files = [f for e in exts for f in glob.glob(os.path.join(args.input, e))]
        for f in sorted(files):
            out = os.path.join(args.output, os.path.basename(f))
            cv2.imwrite(out, enhance(model, cv2.imread(f)))
            print(f'saved: {out}')
    else:
        img = cv2.imread(args.input)
        cv2.imwrite(args.output, enhance(model, img))
        print(f'saved: {args.output}')


if __name__ == '__main__':
    main()
