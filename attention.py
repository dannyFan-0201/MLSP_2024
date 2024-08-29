import tensorflow as tf
from keras.regularizers import l2

class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, channels, reduction):
        super(ChannelAttention, self).__init__()
        self.avg_pool = tf.keras.layers.GlobalAveragePooling3D()
        self.max_pool = tf.keras.layers.GlobalMaxPooling3D()
        self.fc_avg = tf.keras.Sequential([
            tf.keras.layers.Dense(channels // reduction, kernel_regularizer=l2(0.01), use_bias=True, bias_initializer='zeros'),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(channels, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

        ])
        self.fc_max = tf.keras.Sequential([
            tf.keras.layers.Dense(channels // reduction, kernel_regularizer=l2(0.01), use_bias=True, bias_initializer='zeros'),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Dense(channels, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        ])
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)

        channel_attention_avg = self.fc_avg(avg_pool)
        channel_attention_max = self.fc_max(max_pool)
        channel_attention = self.sigmoid(0.5*(channel_attention_max - channel_attention_avg))


        return tf.expand_dims(tf.expand_dims(tf.expand_dims(channel_attention, 1), 1), 1)


class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = tf.keras.layers.Conv3D(32, kernel_size=(3, 3, 3), padding='same', activation='sigmoid')

    def call(self, x):
        max_pool = tf.reduce_max(x, axis=4)
        avg_pool = tf.reduce_mean(x, axis=4)
        max_pool = tf.expand_dims(max_pool, axis=-1)
        avg_pool = tf.expand_dims(avg_pool, axis=-1)
        combined = tf.concat([max_pool, avg_pool], axis=4)
        spatial_attention = self.conv(combined)
        return spatial_attention

# 定義CBAM模塊
class CBAMModule(tf.keras.layers.Layer):
    def __init__(self, channels, reduction, **kwargs):
        super(CBAMModule, self).__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def call(self, x):
        channel_attention = self.channel_attention(x)
        spatial_attention = self.spatial_attention(x)

        result = tf.multiply(x, channel_attention)
        result = tf.multiply(result, spatial_attention)

        return result

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
            "reduction": self.reduction,
        })
        return config

