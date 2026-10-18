"""CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018).

Channel Attention Module (CAM) -> Spatial Attention Module (SAM), applied as a
residual refinement on the input feature map.
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Conv2D, Add, Multiply, Concatenate, Reshape, Flatten, ReLU


class CBAM_Block(Layer):
    """CBAM attention block with residual connection.

    Args:
        f_channels: number of channels of the input feature map.
        reduction_ratio: channel reduction ratio for the shared MLP in CAM.
    """

    def __init__(self, f_channels, reduction_ratio=0.5, **kwargs):
        super(CBAM_Block, self).__init__(**kwargs)
        self.f_channels = f_channels
        self.reduction_ratio = reduction_ratio

        self.mlp1_maxpool = Dense(units=int(f_channels * reduction_ratio))
        self.mlp2_maxpool = Dense(units=f_channels)
        self.mlp1_maxpool_act = ReLU()
        self.mlp2_maxpool_act = ReLU()

        self.mlp1_avgpool = Dense(units=int(f_channels * reduction_ratio))
        self.mlp2_avgpool = Dense(units=f_channels)
        self.mlp1_avgpool_act = ReLU()
        self.mlp2_avgpool_act = ReLU()

        self.reshape_maxpool = Reshape(target_shape=(1, 1, f_channels))
        self.reshape_avgpool = Reshape(target_shape=(1, 1, f_channels))

        self.channel_addition = Add()
        self.channel_refined = Multiply()

        self.concat = Concatenate(axis=3)
        self.conv = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')

        self.spatial_refined = Multiply()

    def CAM(self, input_tensor):
        # Channel Attention Module
        Gmaxpool = tf.math.reduce_max(input_tensor=input_tensor, axis=[1, 2], keepdims=True)
        Gavgpool = tf.math.reduce_mean(input_tensor=input_tensor, axis=[1, 2], keepdims=True)
        Gmaxpool = Flatten()(Gmaxpool)
        Gavgpool = Flatten()(Gavgpool)

        mlp_maxpool = self.mlp1_maxpool_act(self.mlp1_maxpool(Gmaxpool))
        mlp_maxpool = self.mlp2_maxpool_act(self.mlp2_maxpool(mlp_maxpool))

        mlp_avgpool = self.mlp1_avgpool_act(self.mlp1_avgpool(Gavgpool))
        mlp_avgpool = self.mlp2_avgpool_act(self.mlp2_avgpool(mlp_avgpool))

        reshape_maxpool = self.reshape_maxpool(mlp_maxpool)
        reshape_avgpool = self.reshape_avgpool(mlp_avgpool)

        channel_addition = self.channel_addition([reshape_maxpool, reshape_avgpool])
        channel_attention = tf.nn.sigmoid(x=channel_addition)

        return self.channel_refined([input_tensor, channel_attention])

    def SAM(self, channel_refined):
        # Spatial Attention Module
        channel_maxpool = tf.math.reduce_max(input_tensor=channel_refined, axis=3, keepdims=True)
        channel_meanpool = tf.math.reduce_mean(input_tensor=channel_refined, axis=3, keepdims=True)
        concat = self.concat([channel_maxpool, channel_meanpool])

        conv = self.conv(concat)
        spatial_attention = tf.nn.sigmoid(x=conv)

        return self.spatial_refined([channel_refined, spatial_attention])

    def call(self, input_tensor, training=None):
        cam_tensor = self.CAM(input_tensor)
        sam_tensor = self.SAM(cam_tensor)
        refined_feature = Add()([input_tensor, sam_tensor])
        return refined_feature

    def get_config(self):
        config = super(CBAM_Block, self).get_config()
        config.update({'f_channels': self.f_channels,
                       'reduction_ratio': self.reduction_ratio})
        return config
