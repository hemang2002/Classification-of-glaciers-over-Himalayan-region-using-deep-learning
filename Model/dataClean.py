# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 01:44:53 2024

@author: hhmso
"""

import numpy as np
from tqdm import tqdm

import dataRead

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.simplefilter("ignore")


class dataPre:
    
    
    def __init__(self, rotation_range = 90, width_shift_range = 0.0, height_shift_range = 0.0, 
                 shear_range = 0.0, zoom_range = 0.0, horizontal_flip = True, vertical_flip = True, 
                 fill_mode = 'nearest'):
        
        self.datagen_image = ImageDataGenerator(
            rotation_range = rotation_range,
            width_shift_range = width_shift_range,
            height_shift_range = height_shift_range,
            shear_range = shear_range,
            zoom_range = zoom_range,
            horizontal_flip = horizontal_flip,
            vertical_flip = vertical_flip,
            fill_mode = fill_mode
        )
        
        self.datagen_label = ImageDataGenerator(
            rotation_range = rotation_range,
            width_shift_range = width_shift_range,
            height_shift_range = height_shift_range,
            shear_range = shear_range,
            zoom_range = zoom_range,
            horizontal_flip = horizontal_flip,
            vertical_flip = vertical_flip,
            fill_mode = fill_mode
        )
       
    
    def augment(self, image_datas, label_datas, batch_size = 10, num_aug = 10, forD = False):
        
        if forD == True:
            
            print("Data Agument:\n")
            
            image_datas = np.reshape(image_datas, (image_datas.shape[0], 1, image_datas.shape[1], image_datas.shape[2], image_datas.shape[3]))
            label_datas = np.reshape(label_datas, (label_datas.shape[0], 1, label_datas.shape[1], label_datas.shape[2], 1))
                      
            augmented_images = []    
            augmented_labels = []
            for image_data, label_data in tqdm(zip(image_datas, label_datas), desc = "Data augment:"):
                
                seed = np.random.randint(0, 2**8 - 1)
                
                augmented_image = []    
                augmented_label = []
                for image, label in zip(self.datagen_image.flow(image_data, shuffle = False, seed = seed), 
                                              self.datagen_label.flow(label_data, shuffle = False, seed = seed)):
                    
                    augmented_label.append(label)
                    augmented_image.append(image)
                    
                    if len(augmented_image)  >= num_aug:  
                        break
                    
                augmented_labels.extend(augmented_label)
                augmented_images.extend(augmented_image)
            
            augmented_images = np.vstack(augmented_images)
            # augmented_images = np.reshape(augmented_images, (augmented_images.shape[0], augmented_images.shape[1], augmented_images.shape[2], 2, int(augmented_images.shape[3]/2)))
          
            augmented_labels = np.vstack(augmented_labels)
            augmented_labels = augmented_labels.squeeze(axis = 3)
            
            print()
            
            return augmented_images, augmented_labels
        
        else:
            
            augmented_images = np.reshape(augmented_images, (augmented_images.shape[0], augmented_images.shape[1], augmented_images.shape[2], 2, int(augmented_images.shape[3]/2)))
            
            return augmented_images, label_datas
            

    def clearZeros(self, images, masks):
        
        print("Croping Image:\n")
        
        corr_img = []
        corr_mask = []
        
        for image, mask in tqdm(zip(images, masks), desc = "Remove Frame with zero:"):
            
            if not np.any(image == 0.):
                
                corr_img.append(image)
                corr_mask.append(mask)
        
        corr_img = np.array(corr_img)
        corr_mask = np.array(corr_mask)
        
        print()
        
        return corr_img, corr_mask
       
    
#%% 

# import matplotlib.pyplot as plt

# if __name__ == "__main__":
        
#     path = r"..\..\Data\ModelData"
#     dst_crs = 'EPSG:4326' 
#     bands = [3, 4, 5, 6, 10] 

#     img = dataRead.readTiffImage(path, bands, normalization = True) 
#     image, _ = img.getStackedData(dst_crs = dst_crs, max_search_distance = 10, fillna = True, addSlop = True)
#     mask, _ = img.createMask(image)
#     images, masks = img.createStride(image, mask, filters = 128) 
    
#     del(path, bands, dst_crs)
    
#     img = dataPre()
#     images1, masks1 = img.clearZeros(images[:, :, :, :], masks[:, :, :])
#     images2, masks2 = img.augment(images1, masks1, batch_size = 10, num_aug = 2)
