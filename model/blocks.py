"""Dense Residual Block (DR_Block) and basic conv block."""
from tensorflow.keras.layers import Conv2D, Add, Concatenate, ReLU, BatchNormalization


def conv_block(input_tensor, filters_no=16, kernel_value=(3, 3), BN=True):
    """Conv -> ReLU -> (BatchNorm)."""
    conv = Conv2D(filters=filters_no, kernel_size=kernel_value,
                  strides=(1, 1), padding='same')(input_tensor)
    conv = ReLU()(conv)
    if BN:
        return BatchNormalization()(conv)
    return conv


class DR_Block(tf.keras.layers.Layer):
    """Dense Residual Block.

    Five 3x3 convolutions in a dense (concatenative) pattern; the output of the
    last conv is added to the first conv's output as a local residual.
    """

    def __init__(self, filters_no, kernel_value=(3, 3), **kwargs):
        super(DR_Block, self).__init__(**kwargs)
        self.filters_no = filters_no
        self.kernel_value = kernel_value
        self.conv1 = Conv2D(filters=filters_no, kernel_size=kernel_value,
                            strides=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=filters_no, kernel_size=kernel_value,
                            strides=(1, 1), padding='same', activation='relu')
        self.conv3 = Conv2D(filters=filters_no, kernel_size=kernel_value,
                            strides=(1, 1), padding='same', activation='relu')
        self.conv4 = Conv2D(filters=filters_no, kernel_size=kernel_value,
                            strides=(1, 1), padding='same', activation='relu')
        self.conv5 = Conv2D(filters=filters_no, kernel_size=kernel_value,
                            strides=(1, 1), padding='same', activation='relu')
        self.add = Add()

    def call(self, input_tensor):
        conv1 = self.conv1(input_tensor)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(Concatenate(axis=3)([conv1, conv2]))
        conv4 = self.conv4(Concatenate(axis=3)([conv1, conv2, conv3]))
        conv5 = self.conv5(Concatenate(axis=3)([conv1, conv2, conv3, conv4]))
        return self.add([conv5, conv1])

    def get_config(self):
        config = super(DR_Block, self).get_config()
        config.update({'filters_no': self.filters_no,
                       'kernel_value': self.kernel_value})
        return config
