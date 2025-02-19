# Classification-of-glaciers-over-Himalayan-region
Developed U-Net and CNN models with cSE enhancements for classifying Himalayan glaciers using Landsat 8 data.  This improved glacial feature recognition and enabled precise change detection.

This repository contains the implementation of a **deep learning-based glacier classification framework** using **Landsat 8 imagery**. The project aims to enhance the accuracy of glacier classification by leveraging **3D Convolutional Neural Networks (CNNs)** and the **U-Net architecture** with **Spatial and Channel Squeeze & Excitation (scSE) layers**.

## **🌍 Project Overview**  
Glaciers are critical indicators of climate change, and accurate classification is essential for **climate research, water resource management, and natural hazard assessment**. This project automates the classification of glaciers using **deep learning techniques**, addressing the limitations of traditional methods like manual interpretation and supervised classification.

## **📌 Key Features**  
✅ **Landsat 8 Imagery Processing**: Multispectral data preprocessing, including band selection and rasterization.  
✅ **Deep Learning Model**: Implementation of **U-Net with scSE layers** for improved classification accuracy.  
✅ **Data Augmentation & Preprocessing**: Includes **reflectance correction**, **thermal infrared adjustments**, and **topographic data integration**.  
✅ **Automated Glacier Detection**: Reduces reliance on manual classification techniques for a more scalable and efficient solution.  

## **🛠 Methodology**  
### 1. **Data Collection & Preprocessing**  
   - Utilizes **Landsat 8 bands** (Green, Red, Near-Infrared, SWIR, Thermal Infrared).  
   - **Masking** using **Randolph Glacier Inventory 6.0** to define glacier regions.  
   - **Spatial resampling** for data consistency.  

### 2. **Deep Learning Model**  
   - **U-Net architecture** with **convolution layers** for spatial feature extraction.  
   - **scSE Layers** to enhance feature representation.  
   - **Training & Optimization** using advanced loss functions.  

### 3. **Evaluation & Results**  
   - Model performance tested on **real-world glacier datasets**.  
   - **Generalization validation** using diverse environmental conditions.  
   - Recommendations for future improvements in classification accuracy.
