# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:10:44 2024

@author: hhmso
"""

import os
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, BatchNormalization, UpSampling2D, Concatenate
from tensorflow.keras.layers import Dense, Reshape, Add, Multiply, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

import dataClean
import dataRead

class UNetTraining(Model):
    
    
    def __init__(self, input_shape, learning_rate = 0.0001):
        
        super(UNetTraining, self).__init__()
        tf.random.set_seed(42)
        self.learning_rate = learning_rate
        self.model = self.build_model(input_shape)
        self.model.compile(loss = BinaryCrossentropy(), optimizer = Adam(self.learning_rate), metrics = ["acc"])
        
        
    def conv_block(self, inputs, num_filter):
    
        x = Conv2D(num_filter, (3, 3), padding = "same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = Conv2D(num_filter, (3, 3), padding = "same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        return x
    
    
    def encoder_block(self, inputs, num_filter):
    
        x = self.conv_block(inputs, num_filter)
        p = MaxPooling2D(pool_size = (2, 2), padding = 'same')(x)
        
        return x, p
    
    
    def decoder_block(self, inputs, skip, num_filters):
        
        up = UpSampling2D(size = (2, 2))(inputs)
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
        
        # x = self.sCSELayer(d5, reduction_ratio = 2)
        
        outputs = Conv2D(1, (1, 1), activation = "sigmoid", padding = "same")(d5)
                
        model = Model(inputs = inputs, outputs = outputs, name = "Unet3D")
        
        return model
    
    
    def create_dir(self, folder_name, subfolder_names):
        
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
            for subfolder_name in subfolder_names:
                os.makedirs(os.path.join(folder_name, subfolder_name))
        


if __name__ == "__main__":
           
    # stride = 128
    # num_aug = 10
    # fillna = True
    # addSlope = True
    # normalization = True
    # path = r"..\..\Data\ModelData"
    # dst_crs = 'EPSG:4326' 
    # bands = [3, 4, 5, 6, 10] 
    # folder_name = "..\..\Outputs"
    # subfolder_names = ["Logs", "ModelCheckPoint"]
    # train_percent = 0.7
    # val_percent = 0.15
    # test_percent = 0.15

    # img = dataRead.readTiffImage(path, bands, normalization = normalization) 
    # image, _ = img.getStackedData(dst_crs = dst_crs, max_search_distance = 10, fillna = fillna, addSlop = addSlope)
    # mask, _ = img.createMask(image)
    # image, mask = img.createStride(image, mask, filters = stride) 
    # image, mask = image.astype(np.float32), mask.astype(np.uint8)
    
    # img = dataClean.dataPre()
    # image, mask = img.clearZeros(image[:, :, :, :], mask[:, :, :])
    # # image, mask = img.augment(image, mask, batch_size = 10, num_aug = num_aug, forD = False)
    
    # del(stride, num_aug, fillna, addSlope, normalization, path, bands, dst_crs, img)    
    
    unet = UNetTraining((128, 128, 3), learning_rate = 0.0001)
    model = unet.model
    model.summary()
        
    # unet.create_dir(folder_name, subfolder_names)
    # checkpoint_path = os.path.join(folder_name, subfolder_names[1])
    # log_file = os.path.join(folder_name, subfolder_names[0])
    
    # model = unet.model   
    
    # callbacks = [
    #                 ModelCheckpoint(filepath = checkpoint_path + "weights.{epoch:02d}-{val_loss:.2f}-{acc:.2f}.h5", save_weights_only = True, save_best_only = True, verbose = 1),
    #                 ReduceLROnPlateau(monitor = "val_loss", factor = 0.1, patience = 4),
    #                 CSVLogger(log_file + "Model_log.csv"),
    #                 EarlyStopping(monitor = "val_loss", patience = 20, restore_best_weights = True)
    #             ]
    
    # del unet, folder_name, subfolder_names
    
    # x_train, x_test, y_train, y_test = train_test_split(image, mask, test_size = (1 - train_percent), random_state = 42)

    # model.fit(x_train, y_train, epochs = 10, steps_per_epoch = len(x_train), validation_split = 0.3, callbacks = callbacks, verbose = 1)

    # model.save()
    
    # pred = model.predict()
  
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # axs[0].imshow(x_test[1, :, :, 1], cmap='gray')
    # axs[0].set_title('Image 1')
    
    # axs[1].imshow(y_test[1, :, :], cmap='gray')
    # axs[1].set_title('Image 2')
    
    # axs[2].imshow(pred[1, :, :, 0], cmap='gray')
    # axs[2].set_title('Image 3')
    
    # plt.tight_layout()
    # plt.show()   