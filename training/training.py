
from models.clip import CLIP_NLP
from models.encoder import CLIP_VAE_Encoder
from models.decoder import VAE_Decoder
from models.unet import UNet
from models.blocks import Time_Embedding,Cross_Attention_Block
from models.diffusion import  Diffusion
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



diff=Diffusion(CLIP_VAE_Encoder,CLIP_NLP,Time_Embedding,Cross_Attention_Block,UNet,VAE_Decoder,MeanSquaredError,Adam)

diff.fit(10)