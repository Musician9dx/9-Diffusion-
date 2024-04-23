import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, MultiHeadAttention, Multiply, Conv2D, Add, Concatenate, MaxPooling2D, LayerNormalization, Layer, UpSampling2D, Reshape, Flatten, Conv2DTranspose
from tensorflow.keras import Model
from keras_nlp.layers import TokenAndPositionEmbedding, StartEndPacker
from tensorflow import GradientTape
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow import GradientTape
from blocks import VAE_ResidualBlock,VAE_AttentionBlock,Time_Embedding,Cross_Attention_Block


class UNetResidualBlockDownSampler(Layer):

    def __init__(self, input_dim, output_dim):
        super(UNetResidualBlockDownSampler, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.normalization_1 = LayerNormalization()
        self.normalization_2 = LayerNormalization()
        self.conv_2d_1 = Conv2D(filters=self.input_dim, kernel_size=3, strides=(1, 1), padding="same",
                                activation="silu")
        self.conv_2d_2 = Conv2D(filters=self.output_dim, kernel_size=3, strides=(1, 1), padding="same",
                                activation="silu")
        self.conv_2d_3 = Conv2D(filters=self.output_dim, kernel_size=3, strides=(1, 1), padding="same",
                                activation="silu")

        self.add = Add()

    def call(self, x):

        input_dim = x.shape[-1]

        residue = x

        x = self.normalization_1(x)
        x = self.conv_2d_1(x)
        x = self.normalization_2(x)
        x = self.conv_2d_2(x)

        output_dim = x.shape[-1]

        if input_dim == output_dim:

            return x

        else:

            residue = self.conv_2d_3(residue)
            x = self.add([residue, x])
            return x


class UNetResidualBlockUpSampler(Layer):

    def __init__(self, input_dim, output_dim):
        super(UNetResidualBlockUpSampler, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.normalization_1 = LayerNormalization()
        self.normalization_2 = LayerNormalization()
        self.conv_2d_1 = Conv2D(filters=self.input_dim, kernel_size=3, strides=(1, 1), padding="same",
                                activation="silu")
        self.conv_2d_2 = Conv2D(filters=self.output_dim, kernel_size=3, strides=(1, 1), padding="same",
                                activation="silu")
        self.conv_2d_3 = Conv2D(filters=self.output_dim, kernel_size=3, strides=(1, 1), padding="same",
                                activation="silu")

        self.add = Add()

    def call(self, x):

        input_dim = x.shape[-1]

        residue = x

        x = self.normalization_1(x)
        x = self.conv_2d_1(x)
        x = self.normalization_2(x)
        x = self.conv_2d_2(x)

        output_dim = x.shape[-1]

        if input_dim == output_dim:

            return x

        else:

            residue = self.conv_2d_3(residue)
            x = self.add([residue, x])
            return x


class UNetResidualConnection(Layer):

    def __init__(self, filters, ipDim):
        super(UNetResidualConnection, self).__init__()

        self.ln = LayerNormalization()
        self.conv_1 = Conv2D(filters, 3, (1, 1), padding="same", activation="silu")
        self.conv_2 = Conv2D(4 * filters, 3, (1, 1), padding="same", activation="silu")
        self.conv_3 = Conv2D(filters, 3, (1, 1), padding="same", activation="silu")

    def call(self, x):
        x = self.ln(x)

        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)

        return x


class UNet(Layer):

    def __init__(self):
        super(UNet, self).__init__()

        self.downSampler_1 = UNetResidualBlockDownSampler(8, 12)
        self.downSampler_2 = UNetResidualBlockDownSampler(12, 12)
        self.downSampler_3 = UNetResidualBlockDownSampler(12, 12)

        self.conv_1 = Conv2D(12, kernel_size=3, strides=(1, 1), padding="valid", activation="relu")

        self.downSampler_4 = UNetResidualBlockDownSampler(12, 24)
        self.downSampler_5 = UNetResidualBlockDownSampler(24, 24)
        self.downSampler_6 = UNetResidualBlockDownSampler(24, 24)

        self.conv_2 = Conv2D(24, kernel_size=3, strides=(1, 1), padding="valid", activation="relu")

        self.upsampler_1 = UNetResidualBlockUpSampler(24, 12)
        self.upsampler_2 = UNetResidualBlockUpSampler(12, 12)
        self.upsampler_3 = UNetResidualBlockUpSampler(12, 12)

        self.upsampling_1 = Conv2DTranspose(12, 3, (1, 1), "valid")

        self.upsampler_4 = UNetResidualBlockUpSampler(12, 8)
        self.upsampler_5 = UNetResidualBlockUpSampler(8, 8)
        self.upsampler_6 = UNetResidualBlockUpSampler(8, 8)

        self.upsampling_2 = Conv2DTranspose(8, 3, (1, 1), "valid")
        self.add = Add()

    def build(self):
        self.attention = [VAE_AttentionBlock(12, 4) for i in range(4)]

    def call(self, x):
        x = self.downSampler_1(x)
        x = self.downSampler_2(x)
        x = self.downSampler_3(x)
        x = self.conv_1(x)

        y1 = x

        x = self.downSampler_4(x)
        x = self.downSampler_5(x)
        x = self.downSampler_6(x)
        x = self.conv_2(x)

        for layer in self.attention:
            x = layer(x, x)

        x = self.upsampler_1(x)
        x = self.upsampler_2(x)
        x = self.upsampler_3(x)
        x = self.upsampling_1(x)

        # Residual Connection

        x = self.add([x, y1])

        x = self.upsampler_4(x)
        x = self.upsampler_5(x)
        x = self.upsampler_6(x)
        x = self.upsampling_2(x)

        return x


