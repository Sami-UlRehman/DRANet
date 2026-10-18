"""Full network architecture.

Four-stage multi-scale encoder/decoder-style pipeline:
    conv_block -> DR_Block (+local residual) -> CBAM attention -> dense concat
filter schedule: 16 -> 32 -> 64 -> 32, finished by a 3-filter conv, a global
residual add with the input image, and a sigmoid output.
Input/output: 256 x 256 x 3, values in [0, 1].
"""
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Add, Concatenate
from tensorflow.keras import backend as K
import tensorflow as tf

from .blocks import conv_block, DR_Block
from .attention import CBAM_Block


def build_model(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape, name='RGB_input')

    # Stage 1 (16 filters)
    x = skip1 = conv_block(inputs, 16)
    x = DR_Block(filters_no=16)(x)
    x = Add()([x, skip1])
    x = Concatenate(axis=3)([CBAM_Block(K.int_shape(skip1)[3])(skip1), x])

    # Stage 2 (32 filters)
    x = skip2 = conv_block(x, 32)
    x = DR_Block(filters_no=32)(x)
    x = Add()([x, skip2])
    x = Concatenate(axis=3)([CBAM_Block(K.int_shape(skip2)[3])(skip2), x])

    # Stage 3 (64 filters)
    x = skip3 = conv_block(x, 64)
    x = DR_Block(filters_no=64)(x)
    x = Add()([x, skip3])
    x = Concatenate(axis=3)([CBAM_Block(K.int_shape(skip3)[3])(skip3), x])

    # Stage 4 (32 filters)
    x = skip4 = conv_block(x, 32)
    x = DR_Block(filters_no=32)(x)
    x = Add()([x, skip4])
    x = Concatenate(axis=3)([CBAM_Block(K.int_shape(skip4)[3])(skip4), x])

    # Output head: 3-filter conv + global residual + sigmoid
    x = conv_block(x, 3)
    x = Add()([inputs, x])
    outputs = tf.nn.sigmoid(x, name='sigmoid_output')

    return Model(inputs, outputs)


if __name__ == '__main__':
    model = build_model()
    model.summary()
