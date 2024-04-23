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
from vaeComponents import VAE_ResidualBlock,VAE_AttentionBlock


class VAE_Decoder(Layer):

    def __init__(self):
        super(VAE_Decoder, self).__init__()
        self.residual_block_1 = VAE_ResidualBlock(128, 128)
        self.residual_block_2 = VAE_ResidualBlock(128, 128)
        self.residual_block_3 = VAE_ResidualBlock(256, 256)
        self.residual_block_4 = VAE_ResidualBlock(256, 256)
        self.residual_block_5 = VAE_ResidualBlock(512, 512)
        self.residual_block_6 = VAE_ResidualBlock(512, 512)
        self.residual_block_7 = VAE_ResidualBlock(512, 512)
        self.residual_block_8 = VAE_ResidualBlock(512, 512)
        self.residual_block_9 = VAE_ResidualBlock(512, 512)
        self.residual_block_10 = VAE_ResidualBlock(512, 512)

        self.conv_2d_1de = Conv2D(512, kernel_size=3, strides=(1, 1), padding="same", activation="relu")
        self.upsampling_2 = UpSampling2D((2, 2))
        self.upsampling_3 = UpSampling2D((2, 2))
        self.upsampling_4 = UpSampling2D((2, 2))
        self.upsampling_5 = UpSampling2D((2, 2))
        self.conv_2d_6de = Conv2D(3, kernel_size=3, strides=(1, 1), padding="same", activation="relu")

        self.attention_block = VAE_AttentionBlock(8, 4)
        self.normalization = LayerNormalization()

    def call(self, x):
        x = self.normalization(x)
        x = self.conv_2d_1de(x)

        x = self.upsampling_2(x)
        x = self.residual_block_10(x)

        x = self.upsampling_3(x)

        x = self.residual_block_7(x)
        x = self.residual_block_8(x)
        x = self.residual_block_9(x)

        x = self.upsampling_4(x)

        x = self.residual_block_5(x)
        x = self.residual_block_6(x)

        x = self.upsampling_5(x)

        x = self.residual_block_3(x)
        x = self.residual_block_4(x)

        x = self.residual_block_1(x)
        x = self.residual_block_2(x)
        x = self.conv_2d_6de(x)

        return x