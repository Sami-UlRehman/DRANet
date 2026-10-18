"""Train the Dense-Residual + CBAM low-light enhancement network.

Example (LOL dataset layout: <root>/low/*.png and <root>/high/*.png):
    python train.py --data_root ./our485 --epochs 250 --batch_size 16         --out ./saved_model
"""
import argparse
import tensorflow as tf

from model.architecture import build_model
from losses.losses import SSIMLoss

IMG_SIZE = (256, 256)


def _resize_pair(low, high):
    low = tf.image.resize(low, IMG_SIZE)
    high = tf.image.resize(high, IMG_SIZE)
    return low, high


def make_dataset(data_root, batch_size):
    """Zip (low, high) PNG folders into a batched, normalized dataset."""
    low_files = tf.data.Dataset.list_files(f'{data_root}/low/*.png', shuffle=False)
    high_files = tf.data.Dataset.list_files(f'{data_root}/high/*.png', shuffle=False)
    paired = tf.data.Dataset.zip((low_files, high_files))

    def image_map(low_path, high_path):
        low = tf.io.decode_png(tf.io.read_file(low_path), channels=3)
        high = tf.io.decode_png(tf.io.read_file(high_path), channels=3)
        low = tf.image.resize(low, IMG_SIZE)
        high = tf.image.resize(high, IMG_SIZE)
        return tf.cast(low, tf.float32) / 255.0, tf.cast(high, tf.float32) / 255.0

    n_samples = len(list(low_files))
    return (paired.map(image_map, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(batch_size)
                  .prefetch(tf.data.AUTOTUNE)), n_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True, help='dataset root with low/ and high/ folders')
    p.add_argument('--epochs', type=int, default=250)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--out', default='DenseResidualAttentionCBAMLABLOSSDRB_model')
    args = p.parse_args()

    dataset, n_samples = make_dataset(args.data_root, args.batch_size)
    steps_per_epoch = (n_samples // args.batch_size) + 1

    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss=[SSIMLoss])
    model.fit(dataset, epochs=args.epochs, steps_per_epoch=steps_per_epoch)
    model.save(args.out)
    print(f'Model saved to: {args.out}')


if __name__ == '__main__':
    main()
