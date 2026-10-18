"""Custom losses: per-channel normalized LAB loss + combined SSIMLoss.

SSIMLoss = MAE + (1 - SSIM) + LAB_loss
"""
import tensorflow as tf
import tensorflow_io as tfio


def _norm_channel(c):
    """Per-image min-max normalization of a single channel map."""
    cmin = tf.reduce_min(c, axis=[1, 2], keepdims=True)
    cmax = tf.reduce_max(c, axis=[1, 2], keepdims=True)
    return (c - cmin) / (cmax - cmin + 1e-8)


def labloss(y_true, y_pred):
    """MAE computed in LAB color space, each channel per-image normalized."""
    lab_true = tfio.experimental.color.rgb_to_lab(y_true)
    lab_pred = tfio.experimental.color.rgb_to_lab(y_pred)

    L1, a1, b1 = tf.unstack(value=lab_true, axis=3)
    L2, a2, b2 = tf.unstack(value=lab_pred, axis=3)

    g = tf.stack(values=[_norm_channel(L1), _norm_channel(a1), _norm_channel(b1)], axis=3)
    h = tf.stack(values=[_norm_channel(L2), _norm_channel(a2), _norm_channel(b2)], axis=3)

    return tf.reduce_mean(tf.abs(g - h))


def SSIMLoss(y_true, y_pred):
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    return mae_loss + ssim_loss + labloss(y_true, y_pred)
