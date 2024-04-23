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

class Diffusion(Model):

    def __init__(self,

                 encoder,
                 clip,
                 timeEmbedding,
                 crossattention,
                 unet,
                 decoder,
                 mse,
                 adam
                 ):
        super(Diffusion, self).__init__()

        self.encoder = encoder()
        self.clip = clip(10, 128, 12, 512, (8, 8, 8))
        self.timeEmb = timeEmbedding(128, 512, (8, 8, 8))
        self.ca1 = crossattention(12)
        self.ca2 = crossattention(12)

        self.unet1 = unet()
        self.unet2 = unet()
        self.unet3 = unet()

        self.decoder = decoder()

        self.conv_ip = Conv2D(8, 3, (1, 1), padding="same", activation="relu")

        self.cost_function = mse()
        self.optimizer = adam(0.000001)

    def call(self, context, latent, schedule):
        latent = self.conv_ip(latent)

        context = self.clip(context)
        latent = self.encoder(latent)
        time = self.timeEmb(schedule)

        caat1 = self.ca1(context, time)
        spatial_representation = self.ca2(caat1, latent)

        x = self.unet1(spatial_representation)
        x = self.unet2(x)
        x = self.unet3(x)

        x = self.decoder(x)

        return x

    def sampleData(self):
        
        # Data Base Connecters
        
        time = # Data Base Connecters
        latent = # Data Base Connecters
        context = # Data Base Connecters
        target = # Data Base Connecters

        return (

            time,
            latent,
            context,
            target
        )

    def loss_function(self, yreal, ypred):
        loss = self.cost_function(yreal, ypred)

        return loss

    def fit(self, steps):
        for i in range(steps):
            time, latent, context, target = self.sampleData()

            with GradientTape() as tape:
                op = self(context, latent, time)

                loss = self.loss_function(latent, target)

                print(loss)

                grads = tape.gradient(loss, self.trainable_variables)

            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

