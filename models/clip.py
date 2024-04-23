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

class CLIP_NLP_Layer(Layer):

    def __init__(self, numHead, numEmbed):
        super(CLIP_NLP_Layer, self).__init__()

        self.numHead = numHead
        self.numEmbed = numEmbed

        self.normalization_1 = LayerNormalization()
        self.normalization_2 = LayerNormalization()

        self.attention_block = VAE_AttentionBlock(self.numHead, 64)

        self.dense_1 = Dense(self.numEmbed, activation="linear")
        self.dense_2 = Dense(self.numEmbed, activation="linear")

        self.add = Add()

    def call(self, x):
        residue = x

        x = self.normalization_1(x)

        x = self.attention_block(x, x)

        x = self.dense_1(x)
        x = self.add([residue, x])
        x = self.dense_2(x)

        return x


class CLIP_NLP(Layer):

    def __init__(self, seqLength, embedDim, numHead, outputDim, reshape_dim):
        super(CLIP_NLP, self).__init__()

        self.embedDim = embedDim
        self.numHead = numHead
        self.outputDim = outputDim
        self.startEndPackers = StartEndPacker(seqLength)
        self.embedding = TokenAndPositionEmbedding(100, seqLength, embedDim)
        self.flat = Flatten()
        self.linear_3 = Dense(self.outputDim, activation="linear")
        self.linear_4 = Dense(self.outputDim, activation="linear")
        self.reshape = Reshape(reshape_dim)

    def build(self):
        self.clipLayers = [CLIP_NLP_Layer(self.numHead, self.embedDim) for i in range(12)]

    def call(self, x):
        x = self.embedding(x)
        x = tf.expand_dims(x, 0)

        for layer in self.clipLayers:
            x = layer(x)

        x = self.flat(x)

        x = self.linear_3(x)
        x = self.linear_4(x)
        x = self.reshape(x)

        return x