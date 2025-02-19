# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:10:44 2024

@author: hhmso
"""

import os
import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv3D, Activation, MaxPooling3D, BatchNormalization, UpSampling3D, Concatenate
from tensorflow.keras.layers import Dense, Reshape, Add, Multiply, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
# from tensorflow.keras.losses import BinaryCrossentropy
# from tensorflow.keras.optimizers import Adadelta

import dataClean
import dataRead

class UNetTraining(Model):
    
    def __init__(self, input_shape, learning_rate = 0.0001):
        
        super(UNetTraining, self).__init__()
        # tf.random.set_seed(42)
        self.model = self.build_model(input_shape)
        self.model.compile(loss = "binary_crossentropy", optimizer = "Adam", metrics = ["acc"])
        
        
    def conv_block(self, inputs, num_filter):
    
        x = Conv3D(num_filter, (3, 3, 3), padding = "same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = Conv3D(num_filter, (3, 3, 3), padding = "same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        return x
    
    
    def encoder_block(self, inputs, num_filter):
    
        x = self.conv_block(inputs, num_filter)
        p = MaxPooling3D(pool_size = (2, 2, 2), padding = 'same')(x)
        
        return x, p
    
    
    def decoder_block(self, inputs, skip, num_filters):
        
        up = UpSampling3D(size = (2, 2, 2))(inputs)
        
        if skip.shape[3] == 1:
            skip = Conv3D(num_filters*up.shape[3]*2, (3, 3, 3), padding = "same")(skip)
                    
        else:
            skip = Conv3D(16*up.shape[3], (3, 3, 3), padding = "same")(skip)
        
        skip = BatchNormalization()(skip)
        skip = Activation("relu")(skip)
        skip = Reshape((up.shape[1], up.shape[2], up.shape[3], num_filters*2))(skip)
    
        concat = Concatenate()([up, skip])
        conv = self.conv_block(concat, num_filters)
        
        return conv
    
    
    def sCSELayer(self, x, reduction_ratio = 2):
        
        channels = x.shape[-1]
        squeeze_channels  = GlobalAveragePooling2D()(x)
        squeeze_channels  = Reshape((1, 1, channels))(squeeze_channels)
        
        excitation_channels = Dense(channels // reduction_ratio, activation='relu')(squeeze_channels)
        excitation_channels = Dense(channels, activation='sigmoid')(excitation_channels)
        excitation_channels = Reshape((1, 1, channels))(excitation_channels)
        
        channel_wise = Multiply()([x, excitation_channels])
            
        spatial_excitation = Conv2D(1, kernel_size=1, activation='sigmoid')(x)
        spatial_weights = Multiply()([spatial_excitation, tf.reduce_mean(x, axis = -1, keepdims = True)])
        output = Multiply()([x, spatial_weights])
        
        output = Add()([channel_wise, output])
        
        return output
    
    
    def build_model(self, input_shape):
    
        inputs = Input(input_shape)
        
        """ Encoder """
        s1, p1 = self.encoder_block(inputs, 16)
        s2, p2 = self.encoder_block(p1, 32)
        s3, p3 = self.encoder_block(p2, 64)
        s4, p4 = self.encoder_block(p3, 128)
        s5, p5 = self.encoder_block(p4, 256)
        
        """ Bridge """
        b1 = self.conv_block(p5, 512)
        
        """ Decoder """
        d1 = self.decoder_block(b1, s5, 256)
        d2 = self.decoder_block(d1, s4, 128)
        d3 = self.decoder_block(d2, s3, 64)
        d4 = self.decoder_block(d3, s2, 32)
        d5 = self.decoder_block(d4, s1, 16)
        
        x = Conv3D(1, (1, 1, 1), activation = "relu", padding = "same")(d5)
        x = Reshape((x.shape[1], x.shape[2], x.shape[3]))(x)
        
        x = self.sCSELayer(x, reduction_ratio = 2)
        
        output = Conv2D(1, 1, padding = "same", activation = "sigmoid")(x)
        
        model = Model(inputs = inputs, outputs = output, name = "Unet3D")
        
        return model
    
    
    def create_dir(self, name):
        
        token = name.split()
        path = ""
        for i in token:
            path = path.join("//" + i)
            if os.path.exists(i):
                os.mkdir(name)
        


if __name__ == "__main__":
    
       
    stride = 128
    num_aug = 1
    fillna = True
    addSlope = True
    normalization = True
    path = r"..\..\Data\ModelData"
    dst_crs = 'EPSG:4326' 
    bands = [3, 4, 5, 6, 10] 

    img = dataRead.readTiffImage(path, bands, normalization = normalization) 
    image, _ = img.getStackedData(dst_crs = dst_crs, max_search_distance = 10, fillna = fillna, addSlop = addSlope)
    mask, _ = img.createMask(image)
    image, mask = img.createStride(image, mask, filters = stride) 
    image, mask = image.astype(np.float16), mask.astype(np.uint8)
    
    img = dataClean.dataPre()
    image, mask = img.clearZeros(image[:, :, :, :], mask[:, :, :])
    image1, mask1 = img.augment(image, mask, batch_size = 10, num_aug = num_aug, forD = False)
    
    # del(stride, num_aug, fillna, addSlope, normalization, path, bands, dst_crs, img)    
    
    unet = UNetTraining((128, 128, 2, 3), learning_rate = 0.0001)
    
    # unet.create_dir("../../output/dsf")
    # checkpoint_path = ""
    # log_file = ""
    
    # model = unet.model   
    
    # callbacks = [
    #                 ModelCheckpoint(filepath = checkpoint_path + "weights.{epoch:02d}-{val_loss:.2f}.h5", save_weights_only = True, save_best_only = True, verbose = 1),
    #                 ReduceLROnPlateau(monitor = "val_loss", factor = 0.1, patience = 4),
    #                 CSVLogger(log_file + "Model_log.csv"),
    #                 EarlyStopping(monitor = "val_loss", patience = 20, restore_best_weights = True)
    #             ]
    
#     model.fit(x_train, y_train, epochs = 10, validation_split = 0.3, callbacks = callbacks, verbose = 1)

#     model.save()