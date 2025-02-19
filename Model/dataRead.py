# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 10:11:44 2024

@author: hhmso
"""

import os
import re 
from tqdm import tqdm

import numpy as np
import math

import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.warp import reproject, calculate_default_transform
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from rasterio.features import rasterize

import warnings
warnings.simplefilter("ignore")

class readTiffImage:
    
    
    def __init__(self, path = [], bands = [], normalization = False):
        
        self.meta_image = {}
        self.meta_mask = {}
        self.files, self.slop_path, self.meta_path, self.shape_path  = self.getFilePath(path)
        self.bands = bands
        self.normalization = normalization
    
    
    def getFilePath(self, path):
        
        shape = []
        tiff = []
        slop = []
        meta = []
        
        for dirpath, dirnames, filenames in os.walk(path):
            
            for filename in filenames:
                
                if filename.endswith(".shp"): 
                    shape.append(os.path.join(dirpath, filename))
                    
                if filename.lower().endswith(".tif") or filename.lower().endswith(".tiff"): 
                    
                    if "slope" in dirpath.split("\\")[-1].lower():
                        slop.append(os.path.join(dirpath, filename))
                        
                    if "landsat" in dirpath.split("\\")[-1].lower():
                        tiff.append(os.path.join(dirpath, filename))
                    
                if filename.endswith("MTL.txt"): 
                    meta.append(os.path.join(dirpath, filename))
                    
        return tiff, slop, meta, shape
    
    
    def getMetaData(self):
        
        metadata = {}
        with open(self.meta_path[0], 'r') as f:
        
            for line in f.readlines():
                if line != "END\n" and line != "END":
                    key, value = line.strip().split('=')
                    metadata[key.strip()] = value.strip()
        f = None
        
        return metadata
    
    
    def getReflectance(self, band, metadata):
        
        if band <= 9 and band > 0:
            add = float(metadata["REFLECTANCE_ADD_BAND_" + str(band)])
            multi = float(metadata["REFLECTANCE_MULT_BAND_" + str(band)])
        
        elif band == 10:        
            add = float(metadata["TEMPERATURE_ADD_BAND_ST_B" + str(band)])
            multi = float(metadata["TEMPERATURE_MULT_BAND_ST_B" + str(band)])
            
        else:        
            add = 0.0
            multi = 1.0
                
        return add, multi
    
    
    def reprojectBand(self, src, dst_crs = 'EPSG:4326' , fillna = True, max_search_distance = 10, src_band = None):
            
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
          
        dst_profile = src.profile.copy()
        dst_profile.update({
            'crs': dst_crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height
        })
    
        if src_band == None:
            src_band = src.read(1)
        
        dst_band = np.empty((dst_height, dst_width), dtype = src_band.dtype)
        reproject(
            source = src_band,
            destination = dst_band,
            src_transform = src.transform,
            src_crs = src.crs,
            dst_transform = dst_transform,
            dst_crs = dst_crs,
            resampling = Resampling.nearest
        )
            
        if fillna: 
            
            mask = np.where(dst_band == src.nodata, 0, 1)
            filled_data = fillnodata(dst_band, mask, max_search_distance = max_search_distance, smoothing_iterations = 0)
            
            return filled_data, dst_profile, mask
        
        else:
            return dst_band, dst_profile
    
    
    def getSlop(self, image_data, mask_image):
        
        src = rasterio.open(self.slop_path[0])
        data = src.read(1)
        meta = src.meta.copy()
        
        # slope = data * mask_image
        slope = (1 - np.all(image_data[:, :, :] == 0, axis = 2)) * data
        slope[slope == -9999.0] = 0.0 
        slope[slope == -0.0] = 0.0
        
        if self.normalization == True:
            slope, _ = self.min_max_normalization(slope)
            
        print("\n->Reading Slope.")
        
        return slope, meta  


    def min_max_normalization(self, image):
        
        min_val = np.min(image)
        max_val = np.max(image)
        
        mask = image != 0
        normalized_image = ((image - min_val) / (max_val - min_val)) * mask
        normalized_image[normalized_image == -9999.0] = 0.0 
        normalized_image[normalized_image == -0.0] = 0.0
        
        return normalized_image, mask
    

    def getStackedData(self, dst_crs = 'EPSG:4326', max_search_distance = 10, fillna = True, addSlop = True):
        
        print("\n->Reading data")
        
        self.files.sort()
        
        metadata = self.getMetaData()
                
        if addSlop == True:
            num_band = len(self.bands) + 1
                        
        else:
            num_band = len(self.bands)
            
        cnt = 0
        for file in tqdm(self.files, desc = "\nprocessing Bands: "):
            
            band = re.findall(r'\d+', file.split("\\")[-1])[-1]
            
            band = int(band) if band.isdigit() else band
            
            if band in self.bands:
                
                print("\nprocessing Band: ", str(band))
                
                add, multi = self.getReflectance(band, metadata)
                    
                img = rasterio.open(file)
                reProj, meta_reProj, reProj_mask = self.reprojectBand(img, dst_crs, fillna, max_search_distance)
                
                if cnt == 0:
                    image = np.zeros((reProj.shape[0], reProj.shape[1], num_band))
                            
                if self.normalization == True:
                    image[:, :, cnt], mask_image = self.min_max_normalization(((reProj * multi) + add) * reProj_mask)
                    
                else:
                    image[:, :, cnt] = ((reProj * multi) + add) * reProj_mask
                    
                cnt = cnt + 1
                
        if addSlop == True:
            image[:, :, -1], _ = self.getSlop(image, mask_image)  
                    
        for i in ["driver", "dtype", "nodata", "width", "height", "count", "crs", "transform"]:
            self.meta_image[i] = meta_reProj[i]
            
        self.meta_image["count"] = num_band
        self.meta_image["dtype"] = np.float32
        self.meta_image["nodata"] = np.nan
        
        return image, self.meta_image


    def createMask(self, image_data, dst_crs = 'EPSG:4326'):
        
        print("\n->Creating Mask")
        
        gdfs = []
        for i in tqdm(self.shape_path, desc = "Reading Shape file:"):
            
            gdf = gpd.read_file(i)
            gdf = gdf.to_crs(dst_crs)
            gdfs.append(gdf)
        
        gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index = True))
        
        self.meta_mask = self.meta_image.copy()
        burned = rasterize(
            [(geom, 1) for geom in gdf.geometry],
            out_shape = (self.meta_mask["height"], self.meta_mask["width"]),
            fill = 0,
            transform = self.meta_mask["transform"],
            dtype = rasterio.uint8
        )
        
        self.meta_mask["count"] = 1
        self.meta_mask["nodata"] = 0
        
        mask = (1 - np.all(image_data[:,:,:] == 0, axis = 2).astype(np.uint8)) * burned
        
        return mask, self.meta_mask 

    
    def createStride(self, image, mask, filters):
                
        row = math.ceil((image.shape[0] / filters))
        column = math.ceil((image.shape[1] / filters))
        cnt = 0
        
        images = np.empty((row * column, filters, filters, image.shape[2]))
        masks = np.empty((row * column, filters, filters))
        
        for i in tqdm(range(0, row), desc = "Creating stride images:"):
            for j in range(0, column):
                
                if i + 1 == row and j + 1 == column:
                    images[cnt, :, :, :] = image[-filters :,  -filters: , :]
                    masks[cnt, :, :] = mask[-filters :,  -filters:]
                    
                elif i + 1 == row:
                    images[cnt, :, :, :] = image[-filters:, j * filters: (j + 1) * filters, :]
                    masks[cnt, :, :] = mask[-filters:, j * filters: (j + 1) * filters]
                    
                elif j + 1 == column:
                    images[cnt, :, :, :] = image[i * filters: (i + 1) * filters, -filters :, :]
                    masks[cnt, :, :] = mask[i * filters: (i + 1) * filters, -filters:]
                    
                else:   
                    images[cnt, :, :, :] = image[i * filters: (i + 1) * filters, j * filters: (j + 1) * filters, :]
                    masks[cnt, :, :] = mask[i * filters: (i + 1) * filters, j * filters: (j + 1) * filters]
                    
                cnt = cnt + 1
                
        images[images == -0.0] = 0.0 
        masks[masks == -0.0] = 0.0
        
        return images, masks
    
    
#%%

# import matplotlib.pyplot as plt

# if __name__ == "__main__":

#     path = r"..\..\Data\ModelData"
#     dst_crs = 'EPSG:4326' 
#     bands = [3, 4, 5, 6, 10] 

#     img = readTiffImage(path, bands, normalization = False)    
    
#     image, meta = img.getStackedData(dst_crs = dst_crs, max_search_distance = 10, fillna = True, addSlop = True)
   
#     xyz = r"C:\Users\hhmso\Desktop\Hemang\Project\Masters\sem_2\MR & RE\Project_research\Data\InputData\abc1.tif"
#     with rasterio.open(xyz, "w", **meta) as f:
#         f.write(image)
    
#     mask, mask_meta = img.createMask(image)
    
#     xyz = r"C:\Users\hhmso\Desktop\Hemang\Project\Masters\sem_2\MR & RE\Project_research\Data\InputData\abc_mask.tif"
#     with rasterio.open(xyz, "w", **mask_meta) as f:
#         f.write(mask, 1)
    
#     images, masks = img.createStride(image, mask, filters = 128)    
    
#     plt.imshow(mask)
#     plt.show()
   
#     for i in range(0, image.shape[-1]):
#         print(i)
#         plt.imshow(image[:, :, i])
#         plt.show()